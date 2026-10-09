"""
pages/5_🎬_Video_Analysis.py
============================
Video Analysis Page — Frame-by-frame pothole detection with
multiple false-positive suppression techniques:

  1. HIGH confidence threshold (0.40 default, tunable)
  2. IoU/NMS threshold to remove duplicate boxes
  3. MIN BOX SIZE filter — ignores tiny specks (< 0.5% of frame)
  4. TEMPORAL SMOOTHING — a detection only counts if it appears
     in N consecutive frames (default: 3). Eliminates one-frame glitches.
  5. ASPECT RATIO GUARD — potholes are roughly square/oval;
     extremely thin/tall boxes (banners, poles, shadows) are rejected.
  6. EDGE GUARD — ignores detections near frame borders where
     lens distortion causes noise.
  7. Summary statistics + annotated output video download.
"""

import os
import tempfile
import datetime
import time
import collections

import cv2
import numpy as np
import streamlit as st

from common import (
    BASE_DIR, UPLOADS_DIR, load_model, inject_custom_css, draw_boxes, to_rgb
)
try:
    from common import INFER_IMGSZ
except ImportError:
    INFER_IMGSZ = 960

st.set_page_config(
    page_title="Video Analysis | Pothole AI",
    page_icon="🎬",
    layout="wide"
)
inject_custom_css()

# ── Page Header ───────────────────────────────────────────────────────────────
st.title("🎬 Video Pothole Analysis")
st.markdown(
    "Upload a dashcam or road survey video. "
    "The AI analyses every frame with **false-positive suppression** "
    "to avoid flagging shadows, road markings, and reflections."
)

# ── Sidebar Controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Detection Settings")

    conf = st.slider(
        "Confidence Threshold", 0.10, 1.00, 0.40, 0.05,
        help="Minimum confidence to accept a detection. 0.40 recommended for video (higher = fewer false positives)."
    )
    iou_thresh = st.slider(
        "IoU / NMS Threshold", 0.10, 0.90, 0.40, 0.05,
        help="Lower = suppress overlapping/duplicate boxes more aggressively."
    )

    st.markdown("---")
    st.markdown("### 🛡️ False-Positive Filters")

    min_area_pct = st.slider(
        "Min Detection Size (% of frame)", 0.1, 5.0, 0.5, 0.1,
        help="Ignore detections smaller than this % of the frame area. Removes tiny noise."
    )
    temporal_n = st.slider(
        "Temporal Smoothing (frames)", 1, 10, 3, 1,
        help="A pothole must be detected in this many consecutive frames to count. Eliminates one-frame glitches. Set to 1 to disable."
    )
    aspect_min = st.slider(
        "Min Aspect Ratio (W/H)", 0.2, 1.0, 0.35, 0.05,
        help="Reject very tall/thin boxes (poles, shadows). Potholes are roughly square."
    )
    aspect_max = st.slider(
        "Max Aspect Ratio (W/H)", 1.0, 5.0, 4.0, 0.25,
        help="Reject very wide/flat boxes. Potholes should not be extremely wide strips."
    )
    edge_margin_pct = st.slider(
        "Edge Guard Margin (% of frame)", 0, 15, 5, 1,
        help="Ignore detections whose centre falls within this % of the frame edge."
    )

    st.markdown("---")
    st.markdown("### 🎞️ Processing")
    process_every = st.slider(
        "Analyse every Nth frame", 1, 10, 2, 1,
        help="Skip frames to speed up processing. Every 2nd frame is fine for dashcam footage."
    )
    max_frames = st.number_input(
        "Max frames to process (0 = all)", 0, 10000, 0, 100,
        help="Limit total frames for faster previews."
    )
    save_output = st.checkbox("Save annotated output video", value=True)

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded_video = st.file_uploader(
    "Upload road video",
    type=["mp4", "avi", "mov", "mkv", "webm"],
    help="Dashcam footage, phone video, or drone survey video."
)

# ── Helper: is detection valid? ───────────────────────────────────────────────
def is_valid_detection(x1, y1, x2, y2, score, frame_h, frame_w,
                       min_area_pct, aspect_min, aspect_max, edge_margin_pct):
    """Returns True only if the box passes all false-positive filters."""
    bw = x2 - x1
    bh = y2 - y1

    # 1. Minimum size guard
    area_pct = (bw * bh) / (frame_w * frame_h) * 100.0
    if area_pct < min_area_pct:
        return False, "too_small"

    # 2. Aspect ratio guard (W/H)
    if bh == 0:
        return False, "zero_height"
    ar = bw / bh
    if ar < aspect_min or ar > aspect_max:
        return False, "bad_aspect"

    # 3. Edge guard (centre of box must not be near frame border)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    margin_x = frame_w * edge_margin_pct / 100.0
    margin_y = frame_h * edge_margin_pct / 100.0
    if cx < margin_x or cx > frame_w - margin_x:
        return False, "edge_x"
    if cy < margin_y or cy > frame_h - margin_y:
        return False, "edge_y"

    return True, "ok"


# ── Draw boxes with filter reason overlay ────────────────────────────────────
def draw_video_boxes(frame, boxes_data, confirmed_ids):
    """Draw only confirmed (temporally-stable) boxes."""
    ann = frame.copy()
    h, w = ann.shape[:2]
    scale = max(h, w) / 1000.0
    thickness = max(2, int(scale * 2.2))
    font_scale = max(0.45, scale * 0.5)

    for box_id, (x1, y1, x2, y2, score, label) in boxes_data.items():
        if box_id not in confirmed_ids:
            continue
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Color by confidence
        if score >= 0.65:
            color = (30, 30, 235)
        elif score >= 0.40:
            color = (20, 160, 245)
        else:
            color = (40, 200, 60)

        overlay = ann.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.22, ann, 0.78, 0, ann)
        cv2.rectangle(ann, (x1, y1), (x2, y2), color, thickness)

        text = f"{label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                       max(1, int(thickness / 2)))
        tag_h = int(th * 1.6)
        tag_y1 = max(0, y1 - tag_h)
        cv2.rectangle(ann, (x1, tag_y1), (x1 + int(tw + scale * 10), y1), color, -1)
        cv2.putText(ann, text, (x1 + int(scale * 4), y1 - int(th * 0.2)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                    max(1, int(thickness / 2)))
    return ann


# ── Main processing ───────────────────────────────────────────────────────────
if uploaded_video is not None:
    model = load_model()

    # Write uploaded video to a temp file (OpenCV needs a path)
    suffix = os.path.splitext(uploaded_video.name)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_video.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in        = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s    = total_frames / fps_in

    st.info(
        f"📹 **{uploaded_video.name}**  |  "
        f"{frame_w}×{frame_h}  |  {fps_in:.1f} fps  |  "
        f"{total_frames} frames  |  {duration_s:.1f}s"
    )

    frames_to_process = total_frames if max_frames == 0 else min(max_frames, total_frames)
    frames_analysed   = max(1, frames_to_process // process_every)

    st.markdown("---")

    # Output video writer setup
    out_path = None
    out_writer = None
    if save_output:
        ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"video_detected_{ts_str}.mp4"
        out_path = os.path.join(UPLOADS_DIR, out_name)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(out_path, fourcc, fps_in, (frame_w, frame_h))

    # ── UI placeholders ───────────────────────────────────────────────────────
    col_prev, col_stats = st.columns([3, 2])
    with col_prev:
        st.markdown("#### Live Preview")
        frame_placeholder = st.empty()
    with col_stats:
        st.markdown("#### Live Stats")
        stats_placeholder = st.empty()

    progress_bar  = st.progress(0)
    status_text   = st.empty()

    # ── Temporal smoothing state ──────────────────────────────────────────────
    # Maps a simple box-grid-cell to consecutive-frame count
    # We use a 10x10 grid overlay — if a cell has a detection N frames in a row → confirmed
    GRID_COLS, GRID_ROWS = 10, 10
    cell_streak   = collections.defaultdict(int)   # cell -> consecutive hit count
    confirmed_cells = set()                         # cells with streak >= temporal_n

    # ── Counters ──────────────────────────────────────────────────────────────
    frame_idx          = 0
    processed_count    = 0
    total_detections   = 0
    confirmed_count    = 0
    filtered_small     = 0
    filtered_aspect    = 0
    filtered_edge      = 0
    frames_with_det    = 0
    pothole_frame_nos  = []
    conf_history       = []

    t_start = time.time()

    while frame_idx < frames_to_process:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Skip frames for speed
        if (frame_idx % process_every) != 0:
            if out_writer is not None:
                out_writer.write(frame)
            continue

        processed_count += 1
        progress = frame_idx / frames_to_process
        progress_bar.progress(min(progress, 1.0))
        status_text.text(
            f"Frame {frame_idx}/{frames_to_process}  |  "
            f"Analysed: {processed_count}  |  "
            f"Confirmed potholes: {confirmed_count}  |  "
            f"Elapsed: {time.time()-t_start:.1f}s"
        )

        # ── YOLO inference ────────────────────────────────────────────────────
        results = model.predict(frame, imgsz=INFER_IMGSZ, conf=conf, iou=iou_thresh, verbose=False)

        # Grid cells hit this frame (for temporal smoothing)
        hit_cells_this_frame = set()
        raw_boxes = {}    # id -> (x1,y1,x2,y2,score,label)
        confirmed_ids = set()

        for i, box in enumerate(results[0].boxes):
            score = float(box.conf[0].item())
            if score < conf:
                continue

            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            cls_id = int(box.cls[0].item())
            label  = results[0].names.get(cls_id, "pothole")

            total_detections += 1

            # ── Apply false-positive filters ──────────────────────────────────
            valid, reason = is_valid_detection(
                x1, y1, x2, y2, score,
                frame_h, frame_w,
                min_area_pct, aspect_min, aspect_max, edge_margin_pct
            )

            if not valid:
                if reason == "too_small":
                    filtered_small += 1
                elif reason == "bad_aspect":
                    filtered_aspect += 1
                elif reason in ("edge_x", "edge_y"):
                    filtered_edge += 1
                continue

            # Map detection centre to grid cell for temporal tracking
            cx = int((x1 + x2) / 2 / frame_w * GRID_COLS)
            cy = int((y1 + y2) / 2 / frame_h * GRID_ROWS)
            cell = (cx, cy)
            hit_cells_this_frame.add(cell)

            raw_boxes[i] = (x1, y1, x2, y2, score, label)

        # ── Temporal smoothing ────────────────────────────────────────────────
        # Increment streak for cells hit this frame, reset others
        all_cells = set(cell_streak.keys()) | hit_cells_this_frame
        for cell in all_cells:
            if cell in hit_cells_this_frame:
                cell_streak[cell] += 1
                if cell_streak[cell] >= temporal_n:
                    confirmed_cells.add(cell)
            else:
                cell_streak[cell] = 0
                confirmed_cells.discard(cell)

        # Determine which raw boxes belong to confirmed cells
        for box_id, (x1, y1, x2, y2, score, label) in raw_boxes.items():
            cx = int((x1 + x2) / 2 / frame_w * GRID_COLS)
            cy = int((y1 + y2) / 2 / frame_h * GRID_ROWS)
            if (cx, cy) in confirmed_cells:
                confirmed_ids.add(box_id)
                conf_history.append(score)

        if confirmed_ids:
            confirmed_count += len(confirmed_ids)
            frames_with_det += 1
            pothole_frame_nos.append(frame_idx)

        # ── Draw + output ─────────────────────────────────────────────────────
        ann_frame = draw_video_boxes(frame, raw_boxes, confirmed_ids)

        # Overlay HUD on frame
        hud_lines = [
            f"Frame: {frame_idx}",
            f"Confirmed: {len(confirmed_ids)}",
            f"Conf >= {conf:.2f}",
        ]
        for li, txt in enumerate(hud_lines):
            cv2.putText(ann_frame, txt, (10, 25 + li * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

        if out_writer is not None:
            out_writer.write(ann_frame)

        # Show preview every ~15 processed frames to avoid UI lag
        if processed_count % 5 == 0 or processed_count == 1:
            rgb = cv2.cvtColor(ann_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb, use_container_width=True,
                                    caption=f"Frame {frame_idx}")

            avg_c = float(np.mean(conf_history)) if conf_history else 0.0
            stats_placeholder.markdown(f"""
| Metric | Value |
|---|---|
| Frames Processed | {processed_count} |
| Confirmed Potholes | {confirmed_count} |
| Frames With Potholes | {frames_with_det} |
| Avg Confidence | {avg_c:.1%} |
| Filtered (tiny) | {filtered_small} |
| Filtered (shape) | {filtered_aspect} |
| Filtered (edge) | {filtered_edge} |
""")

    cap.release()
    if out_writer is not None:
        out_writer.release()

    os.unlink(tmp_path)   # clean up temp file

    progress_bar.progress(1.0)
    status_text.text("✅ Analysis complete!")

    # ── Final Summary ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Analysis Summary")

    avg_conf_final = float(np.mean(conf_history)) if conf_history else 0.0
    pothole_rate   = frames_with_det / max(processed_count, 1) * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Frames Analysed",     processed_count)
    c2.metric("Frames With Potholes", frames_with_det)
    c3.metric("Pothole Frame Rate",   f"{pothole_rate:.1f}%")
    c4.metric("Avg Confidence",       f"{avg_conf_final:.1%}")
    c5.metric("Filtered Out",         filtered_small + filtered_aspect + filtered_edge,
              delta=f"-{filtered_small+filtered_aspect+filtered_edge} false positives removed",
              delta_color="inverse")

    # Breakdown of filters
    st.markdown("#### 🛡️ False-Positive Filter Breakdown")
    fc1, fc2, fc3 = st.columns(3)
    fc1.metric("Removed: Too Tiny",   filtered_small,   help="Box area < min size threshold")
    fc2.metric("Removed: Wrong Shape", filtered_aspect, help="Aspect ratio outside pothole range")
    fc3.metric("Removed: Frame Edge",  filtered_edge,   help="Detection near frame border (distortion zone)")

    # Road condition assessment
    st.markdown("#### 🛣️ Road Condition Assessment")
    if pothole_rate > 30:
        st.error(f"🚨 **CRITICAL** — Potholes detected in {pothole_rate:.1f}% of analysed frames. Immediate repair required.")
    elif pothole_rate > 15:
        st.warning(f"⚠️ **HIGH** — Potholes detected in {pothole_rate:.1f}% of frames. Schedule repair soon.")
    elif pothole_rate > 5:
        st.info(f"🟡 **MODERATE** — Potholes detected in {pothole_rate:.1f}% of frames. Monitor condition.")
    elif pothole_rate > 0:
        st.success(f"✅ **LOW** — Minor surface damage detected in {pothole_rate:.1f}% of frames.")
    else:
        st.success("✅ **CLEAR** — No confirmed potholes detected in this video.")

    # Pothole hotspot timeline
    if pothole_frame_nos:
        st.markdown("#### 📈 Pothole Timeline (frame numbers)")
        timeline_data = {"Frame Number": pothole_frame_nos}
        import pandas as pd
        df = pd.DataFrame(timeline_data)
        df["Time (s)"] = (df["Frame Number"] / fps_in).round(2)
        st.dataframe(df, use_container_width=True, height=200)

        # Timestamps for worst segments
        timestamps = [f"{int(f/fps_in//60):02d}:{int(f/fps_in%60):02d}" for f in pothole_frame_nos[:10]]
        st.caption(f"First pothole hotspots at: {', '.join(timestamps)}")

    # Download annotated video
    if save_output and out_path and os.path.exists(out_path):
        st.markdown("---")
        with open(out_path, "rb") as vf:
            st.download_button(
                label="⬇️ Download Annotated Video",
                data=vf,
                file_name=os.path.basename(out_path),
                mime="video/mp4",
            )

else:
    st.info("⬆️ Upload a dashcam or road survey video above to begin analysis.")

    # Tips section
    with st.expander("💡 Tips for best results", expanded=True):
        st.markdown("""
**To reduce false positives (detecting non-potholes):**
- 🔼 **Raise the Confidence Threshold** — start at 0.40, go higher if needed
- 🔼 **Raise Temporal Smoothing** — set to 3–5 frames: only flags *persistent* detections
- 🔽 **Lower Min Detection Size** to 0.5%+ — filters tiny shadow artifacts
- 🔽 **Raise Edge Guard Margin** — removes detections at frame corners

**Common false positives in road videos and their fixes:**
| False Positive | Cause | Fix |
|---|---|---|
| Road markings (lines, arrows) | Low confidence, usually elongated | Raise aspect ratio min + confidence |
| Shadow patches | Ephemeral (1 frame) | Increase temporal smoothing |
| Puddle reflections | Small + near edges | Min size + edge guard |
| Drain covers / manholes | Visually similar | Raise confidence ≥ 0.45 |
| Frame blur / lens distortion | Near edges | Edge guard margin |

**Best video settings:**
- Dashcam: confidence 0.40, temporal 3, min-size 0.8%
- Slow survey (< 20 km/h): confidence 0.35, temporal 2, min-size 0.5%
""")
