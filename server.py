"""
server.py
==========
Flask Web Server & REST API for the HTML/CSS/JS Pothole AI System.
Serves dedicated multi-page frontend and provides endpoints for
YOLOv8 inference, GPS extraction, and SQLite database storage.
"""

import os
import json
import base64
import sqlite3
import datetime
import subprocess
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
import imageio_ffmpeg
from flask import Flask, render_template, request, jsonify, send_from_directory
from ultralytics import YOLO

from common import (
    BASE_DIR, MODEL_PATH, DB_PATH, UPLOADS_DIR, SAMPLES_DIR, INFER_IMGSZ,
    load_model, draw_boxes, to_rgb, build_detections,
    overall_severity, extract_gps_from_exif, reverse_geocode, get_pothole_image,
    load_reports, search_reports, update_status, delete_report, save_report, init_db
)

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "web", "templates"),
    static_folder=os.path.join(BASE_DIR, "web", "static")
)
app.config['TEMPLATES_AUTO_RELOAD'] = True

import time

ALLOWED_STATUSES = ["Reported", "In Progress", "Repaired"]
# The Streamlit app / older rows use a different vocabulary; map it at read time (DB rows are not rewritten)
_LEGACY_STATUS = {"Under Review": "In Progress", "Fixed": "Repaired", "Critical": "Reported"}
STARTED_AT = time.time()
LAST_INFERENCE_MS = None


def norm_status(s):
    s = (s or "Reported").strip()
    s = _LEGACY_STATUS.get(s, s)
    return s if s in ALLOWED_STATUSES else "Reported"


def decorate_report(r):
    """Add web-ready fields (image_url, timestamp, normalized status) to a report dict."""
    raw_img = get_pothole_image(r.get("image_path"), r.get("road_name", ""), r.get("address", ""))
    if raw_img:
        rel = os.path.relpath(raw_img, BASE_DIR).replace('\\', '/')
        r["image_url"] = f"/{rel}"
    else:
        r["image_url"] = "/sample_images/sample_city_pothole_annotated.jpg"
    # False when the UI is showing a generic sample photo instead of this report's own evidence
    r["has_own_image"] = not r["image_url"].startswith("/sample_images/")
    r["timestamp"] = r.get("ts")
    r["status"] = norm_status(r.get("status"))
    return r


# Initialize Database and Load Model on Startup
init_db()
model = load_model()
class_names = model.names

# ── Static File Routes for Images ─────────────────────────────────────────────
@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(UPLOADS_DIR, filename)

@app.route('/sample_images/<path:filename>')
def serve_sample(filename):
    return send_from_directory(SAMPLES_DIR, filename)

# ── Page HTML Routes (Separate Pages) ─────────────────────────────────────────
@app.route('/')
def page_home():
    return render_template('index.html')

@app.route('/detect')
def page_detect():
    return render_template('detect.html')

@app.route('/video')
def page_video():
    return render_template('video.html')

@app.route('/heatmap')
def page_heatmap():
    return render_template('heatmap.html')

@app.route('/reports')
def page_reports():
    return render_template('reports.html')

@app.route('/road-search')
def page_road_search():
    return render_template('road-search.html')

@app.route('/bulk')
def page_bulk():
    return render_template('bulk.html')

# ── REST API Endpoints ────────────────────────────────────────────────────────
@app.route('/api/health', methods=['GET'])
def api_health():
    con = sqlite3.connect(DB_PATH)
    try:
        journal = con.execute("PRAGMA journal_mode").fetchone()[0]
        records = con.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
    finally:
        con.close()
    try:
        device = str(model.device)
    except Exception:
        device = "unknown"
    return jsonify({
        "status": "ok",
        "uptime_sec": int(time.time() - STARTED_AT),
        "model": {
            "file": os.path.basename(MODEL_PATH),
            "classes": list(class_names.values()),
            "device": device,
            "last_inference_ms": LAST_INFERENCE_MS,
        },
        "db": {"file": os.path.basename(DB_PATH), "records": records,
               "journal_mode": str(journal).upper(), "sqlite": sqlite3.sqlite_version},
        "basemap": {"key_source": "env" if os.environ.get("CARTO_API_KEY") else "built-in default"},
    })


@app.route('/api/stats', methods=['GET'])
def api_stats():
    reports = load_reports()
    total_potholes = sum(r["num_potholes"] for r in reports)
    
    sev_counts = {}
    stat_counts = {}
    for r in reports:
        s = r.get("severity", "Low")
        sev_counts[s] = sev_counts.get(s, 0) + 1
        st = norm_status(r.get("status"))
        stat_counts[st] = stat_counts.get(st, 0) + 1

    return jsonify({
        "total_reports": len(reports),
        "total_potholes": total_potholes,
        "severity_counts": sev_counts,
        "status_counts": stat_counts
    })


@app.route('/api/reports', methods=['GET'])
def api_get_reports():
    q = request.args.get('q', '').strip()
    limit = request.args.get('limit', type=int)
    
    reports = search_reports(q) if q else load_reports()
    if limit:
        reports = reports[:limit]

    # Resolve web accessible image URLs
    for r in reports:
        decorate_report(r)

    return jsonify({"reports": reports})


@app.route('/api/reports', methods=['POST'])
def api_save_report():
    data = request.json or {}
    try:
        lat = float(data.get("lat", 18.5204))
        lon = float(data.get("lon", 73.8567))
    except (TypeError, ValueError):
        return jsonify({"error": "lat/lon must be numbers"}), 400
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return jsonify({"error": "lat/lon out of range"}), 400
    address = data.get("address", "")
    road_name = data.get("road_name", "")
    detections = data.get("detections", [])
    num_potholes = len(detections)
    
    confs = [d["confidence"] for d in detections]
    areas = [d["area_pct"] for d in detections]
    avg_conf = float(np.mean(confs)) if confs else 0.0
    max_area = float(max(areas)) if areas else 0.0
    sev = overall_severity(confs)
    img_rel = data.get("saved_image_rel", "")

    save_report(lat, lon, address, road_name, num_potholes, avg_conf, max_area, sev, detections, image_path=img_rel)
    return jsonify({"status": "ok", "message": "Report saved successfully"})


@app.route('/api/reports/<int:report_id>/status', methods=['POST', 'PUT'])
def api_update_status(report_id):
    data = request.json or {}
    new_status = data.get("status")
    if not new_status:
        return jsonify({"error": "Missing status parameter"}), 400
    new_status = _LEGACY_STATUS.get(new_status, new_status)
    if new_status not in ALLOWED_STATUSES:
        return jsonify({"error": f"Invalid status. Use one of: {', '.join(ALLOWED_STATUSES)}"}), 400
    
    update_status(report_id, new_status)
    return jsonify({"status": "ok", "id": report_id, "new_status": new_status})


@app.route('/api/reports/<int:report_id>', methods=['DELETE'])
def api_delete_report(report_id):
    delete_report(report_id)
    return jsonify({"status": "ok", "id": report_id, "message": "Report deleted successfully"})


@app.route('/api/seed-db', methods=['POST'])
def api_seed_db():
    try:
        from scrape_india_data import get_db, fetch_seed
        con = get_db()
        count = fetch_seed(con)
        con.close()
        return jsonify({"status": "ok", "count": count, "message": f"Successfully seeded {count} verified Indian road reports."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/search', methods=['GET'])
def api_search_road():
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({"reports": []})

    results = search_reports(query)
    for r in results:
        decorate_report(r)

    return jsonify({"query": query, "reports": results})


@app.route('/api/reverse-geocode', methods=['GET'])
@app.route('/api/reverse_geocode', methods=['GET'])
def api_reverse_geocode():
    """Reverse-geocodes coordinates to Street / Area / City format."""
    try:
        lat = float(request.args.get('lat', 18.5204))
        lon = float(request.args.get('lon', 73.8567))
        road_name, address = reverse_geocode(lat, lon)
        return jsonify({
            "status": "ok",
            "lat": lat,
            "lon": lon,
            "road_name": road_name,
            "address": address
        })
    except Exception as e:
        return jsonify({"error": str(e), "road_name": "Road Corridor", "address": ""}), 400


@app.route('/api/detect', methods=['POST'])
def api_detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    conf = float(request.form.get('conf', 0.20))
    road_name = (request.form.get('road_name') or '').strip()
    fallback_lat = float(request.form.get('fallback_lat', 18.5204))
    fallback_lon = float(request.form.get('fallback_lon', 73.8567))

    # 1. Read EXIF GPS
    file.seek(0)
    exif_lat, exif_lon = extract_gps_from_exif(file)
    file.seek(0)

    if exif_lat is not None and exif_lon is not None:
        lat, lon = exif_lat, exif_lon
        gps_source = "EXIF"
        auto_road, auto_addr = reverse_geocode(lat, lon)
        road_name = auto_road or road_name or f"Corridor ({lat:.4f}, {lon:.4f})"
        address = auto_addr if auto_addr else f"{road_name} ({lat:.4f}, {lon:.4f})"
    else:
        lat, lon = fallback_lat, fallback_lon
        gps_source = "FALLBACK"
        # If road_name is empty or a generic placeholder, auto-reverse geocode
        generic_markers = ("survey", "corridor", "detecting", "resolving", "pending", "fallback", "auto", "road corridor")
        if not road_name or any(m in road_name.lower() for m in generic_markers):
            auto_road, auto_addr = reverse_geocode(lat, lon)
            road_name = auto_road or "Street / Area / City"
            address = auto_addr if auto_addr else f"{road_name} ({lat:.4f}, {lon:.4f})"
        else:
            address = f"{road_name} ({lat:.4f}, {lon:.4f})"

    # 2. Decode Image
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({"error": "Invalid or corrupt image format"}), 400

    h, w = img_bgr.shape[:2]

    # 3. YOLO Inference. Ultralytics expects OpenCV BGR arrays and converts to RGB itself,
    #    so pass img_bgr directly (passing RGB swaps colour channels and ruins detections).
    infer_sz = 1280 if max(h, w) >= 1600 else INFER_IMGSZ
    global LAST_INFERENCE_MS
    _t0 = time.perf_counter()
    results = model.predict(img_bgr, imgsz=infer_sz, conf=conf, verbose=False)
    LAST_INFERENCE_MS = round((time.perf_counter() - _t0) * 1000, 1)
    ann_bgr = draw_boxes(img_bgr, results, conf)
    dets = build_detections(results, conf, class_names, h, w)

    confs = [d["confidence"] for d in dets]
    areas = [d["area_pct"] for d in dets]
    avg_conf = float(np.mean(confs)) if confs else 0.0
    max_area = float(max(areas)) if areas else 0.0
    sev = overall_severity(confs)

    # 4. Save high-resolution annotated image to uploads
    ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_filename = f"detect_{ts_str}_{len(dets)}potholes.jpg"
    saved_path = os.path.join(UPLOADS_DIR, saved_filename)
    cv2.imwrite(saved_path, ann_bgr)
    saved_rel = os.path.join("uploads", saved_filename).replace('\\', '/')

    # 5. Fast Web-Optimized Previews (prevents browser choking on 25MB base64)
    max_web_dim = 1400
    if max(h, w) > max_web_dim:
        scale = max_web_dim / max(h, w)
        preview_orig = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        preview_ann = cv2.resize(ann_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        preview_orig = img_bgr
        preview_ann = ann_bgr

    _, orig_buf = cv2.imencode('.jpg', preview_orig, [cv2.IMWRITE_JPEG_QUALITY, 85])
    _, ann_buf = cv2.imencode('.jpg', preview_ann, [cv2.IMWRITE_JPEG_QUALITY, 85])
    orig_b64 = "data:image/jpeg;base64," + base64.b64encode(orig_buf).decode('utf-8')
    ann_b64 = "data:image/jpeg;base64," + base64.b64encode(ann_buf).decode('utf-8')

    return jsonify({
        "lat": lat,
        "lon": lon,
        "gps_source": gps_source,
        "road_name": road_name,
        "address": address,
        "detections": dets,
        "avg_conf": avg_conf,
        "max_area_pct": max_area,
        "severity": sev,
        "saved_image_rel": saved_rel,
        "image_url": f"/{saved_rel}",
        "original_data_url": orig_b64,
        "annotated_data_url": ann_b64
    })


@app.route('/api/bulk-detect', methods=['POST'])
def api_bulk_detect():
    files = request.files.getlist('images')
    if not files or len(files) == 0:
        return jsonify({"error": "No image files provided"}), 400

    conf = float(request.form.get('conf', 0.20))
    corridor_name = request.form.get('road_name', 'Survey Corridor')
    base_lat = float(request.form.get('lat', 18.5204))
    base_lon = float(request.form.get('lon', 73.8567))

    results = []
    total_cavities = 0
    high_count = 0

    for idx, file in enumerate(files):
        if not file.filename:
            continue
        file.seek(0)
        exif_lat, exif_lon = extract_gps_from_exif(file)
        file.seek(0)

        if exif_lat is not None and exif_lon is not None:
            cur_lat, cur_lon = exif_lat, exif_lon
            gps_src = "EXIF"
        else:
            # No EXIF GPS: use the coordinates the user supplied as-is (no invented offsets)
            cur_lat, cur_lon = base_lat, base_lon
            gps_src = "FALLBACK"

        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue

        h, w = img_bgr.shape[:2]
        yolo_res = model.predict(img_bgr, imgsz=INFER_IMGSZ, conf=conf, verbose=False)
        dets = build_detections(yolo_res, conf, class_names, h, w)
        ann_bgr = draw_boxes(img_bgr, yolo_res, conf)

        confs = [d["confidence"] for d in dets]
        areas = [d["area_pct"] for d in dets]
        avg_c = float(np.mean(confs)) if confs else 0.0
        max_a = float(max(areas)) if areas else 0.0
        sev = overall_severity(confs)

        saved_rel = None
        if dets:  # only keep evidence for frames that actually contain potholes
            ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            saved_filename = f"bulk_{ts_str}_{idx}_{len(dets)}potholes.jpg"
            saved_path = os.path.join(UPLOADS_DIR, saved_filename)
            cv2.imwrite(saved_path, ann_bgr)
            saved_rel = os.path.join("uploads", saved_filename).replace('\\', '/')

        road_label = f"{corridor_name} · {file.filename}"
        addr_label = f"{road_label} ({cur_lat:.4f}, {cur_lon:.4f})"

        if dets:  # never store empty (0-pothole) frames as reports
            save_report(
                cur_lat, cur_lon, addr_label, road_label,
                len(dets), avg_c, max_a, sev, dets,
                image_path=saved_rel
            )

        total_cavities += len(dets)
        if sev == "High":
            high_count += 1

        results.append({
            "index": idx + 1,
            "filename": file.filename,
            "road_name": road_label,
            "lat": round(cur_lat, 5),
            "lon": round(cur_lon, 5),
            "gps_source": gps_src,
            "potholes_count": len(dets),
            "avg_conf": round(avg_c, 3),
            "max_area_pct": round(max_a, 2),
            "severity": sev,
            "image_url": f"/{saved_rel}" if saved_rel else None,
            "saved": bool(dets)
        })

    return jsonify({
        "status": "ok",
        "processed_count": len(results),
        "total_cavities": total_cavities,
        "critical_count": high_count,
        "results": results
    })


@app.route('/api/detect-frame', methods=['POST'])
def api_detect_frame():
    """Ultra-fast frame detection for real-time video playback and streaming."""
    conf = float(request.form.get('conf', 0.20))
    file = request.files.get('frame')
    if not file:
        if request.is_json:
            data = request.get_json() or {}
            conf = float(data.get('conf', conf))
            b64_str = data.get('frame', '')
            if ',' in b64_str:
                b64_str = b64_str.split(',', 1)[1]
            try:
                file_bytes = base64.b64decode(b64_str)
                img_bgr = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
            except Exception as e:
                return jsonify({"error": f"Base64 decode failed: {str(e)}"}), 400
        else:
            return jsonify({"error": "No frame provided"}), 400
    else:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return jsonify({"error": "Invalid frame data"}), 400

    h, w = img_bgr.shape[:2]
    results = model.predict(img_bgr, imgsz=INFER_IMGSZ, conf=conf, verbose=False)
    dets = build_detections(results, conf, class_names, h, w)
    confs = [d["confidence"] for d in dets]

    return jsonify({
        "status": "ok",
        "count": len(dets),
        "detections": dets,
        "max_conf": float(max(confs)) if confs else 0.0,
        "severity": overall_severity(confs) if confs else "None"
    })


@app.route('/api/detect-video', methods=['POST'])
def api_detect_video():
    conf = float(request.form.get('conf', 0.20))
    sample_rate = max(1, min(30, int(request.form.get('sample_rate', 1))))
    road_name = request.form.get('road_name', 'Survey Dashcam Corridor')
    lat = float(request.form.get('lat', 18.5204))
    lon = float(request.form.get('lon', 73.8567))
    sample_mode = request.form.get('sample_mode', 'false').lower() == 'true'

    ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = os.path.join(UPLOADS_DIR, f"temp_frames_{ts_str}")
    os.makedirs(temp_dir, exist_ok=True)

    # 1. Resolve Input Video File
    has_custom_upload = ('video' in request.files and request.files['video'].filename != '')
    if sample_mode:
        input_video_path = os.path.join(SAMPLES_DIR, 'sample_dashcam.mp4')
        orig_rel = "sample_images/sample_dashcam.mp4"
    elif has_custom_upload:
        file = request.files['video']
        _, ext = os.path.splitext(file.filename or "video.mp4")
        if not ext:
            ext = ".mp4"
        saved_input_name = f"input_dashcam_{ts_str}{ext}"
        input_video_path = os.path.join(UPLOADS_DIR, saved_input_name)
        file.save(input_video_path)
        orig_rel = os.path.join("uploads", saved_input_name).replace('\\', '/')
    else:
        return jsonify({"error": "No video file provided in custom mode. Please upload a video file or click 'Test Sample'."}), 400

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        return jsonify({"error": f"Failed to open video file: {input_video_path}"}), 400

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Process up to 300 frames for fast, responsive interactive feedback
    max_frames_to_process = min(total_frames, 300)
    frame_idx = 0
    all_confs = []
    occurrences = []
    start_time = datetime.datetime.now()

    cached_results = None
    cached_boxes = []

    while frame_idx < max_frames_to_process:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # Only run heavy neural inference every sample_rate frames
        should_infer = (frame_idx % sample_rate == 0) or (cached_results is None)
        if should_infer:
            cached_results = model.predict(frame_bgr, imgsz=INFER_IMGSZ, conf=conf, verbose=False)
            cached_boxes = cached_results[0].boxes

        # Draw boxes using current or cached detection
        ann_bgr = draw_boxes(frame_bgr, cached_results, conf)

        # Telemetry HUD
        hud_text = f"FRAME {frame_idx + 1}/{max_frames_to_process} | STEP: {sample_rate}x | CAVITIES: {len(cached_boxes)} | YOLOv8"
        cv2.putText(ann_bgr, hud_text, (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.imwrite(os.path.join(temp_dir, f"f_{frame_idx:04d}.jpg"), ann_bgr)

        if len(cached_boxes) > 0 and should_infer:
            box_confs = [float(b.conf[0]) for b in cached_boxes]
            all_confs.extend(box_confs)
            occurrences.append({
                "frame": frame_idx + 1,
                "timestamp_sec": round(frame_idx / fps, 2),
                "timestamp_display": f"{int((frame_idx/fps)//60):02d}:{int((frame_idx/fps)%60):02d}",
                "count": len(cached_boxes),
                "max_conf": round(max(box_confs), 3),
                "severity": overall_severity(box_confs)
            })

        frame_idx += 1

    cap.release()

    # 3. Assemble H.264 MP4 using imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    out_video_name = f"annotated_dashcam_{ts_str}.mp4"
    out_video_path = os.path.join(UPLOADS_DIR, out_video_name)

    cmd = [
        ffmpeg_exe, '-y',
        '-framerate', str(fps),
        '-i', os.path.join(temp_dir, 'f_%04d.jpg'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        out_video_path
    ]
    subprocess.run(cmd, check=True)

    # Clean up temp frames
    for f in os.listdir(temp_dir):
        try:
            os.remove(os.path.join(temp_dir, f))
        except Exception:
            pass
    try:
        os.rmdir(temp_dir)
    except Exception:
        pass

    duration_sec = round(frame_idx / fps, 1)
    proc_duration = round((datetime.datetime.now() - start_time).total_seconds(), 2)
    sev = overall_severity(all_confs) if all_confs else "None"
    avg_conf = float(np.mean(all_confs)) if all_confs else 0.0
    max_conf = float(max(all_confs)) if all_confs else 0.0

    out_rel = os.path.join("uploads", out_video_name).replace('\\', '/')

    return jsonify({
        "status": "ok",
        "annotated_video_url": f"/{out_rel}",
        "original_video_url": f"/{orig_rel}",
        "total_frames": frame_idx,
        "fps": round(fps, 1),
        "duration_sec": duration_sec,
        "occurrences_count": len(occurrences),
        "occurrences": occurrences,
        "avg_conf": round(avg_conf, 3),
        "max_conf": round(max_conf, 3),
        "severity": sev,
        "road_name": road_name,
        "lat": lat,
        "lon": lon,
        "processing_time_sec": proc_duration
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"=================================================")
    print(f"  POTHOLE AI HTML/CSS/JS SERVER RUNNING")
    print(f"  URL:   http://localhost:{port}")
    print(f"  Model: {os.path.basename(MODEL_PATH)} ({MODEL_PATH})")
    print(f"=================================================")
    app.run(host='0.0.0.0', port=port, debug=False)
