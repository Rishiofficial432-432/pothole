"""
pages/1_🔍_Detect.py
====================
Dedicated Page 1: Single image/photo upload, AI inference, and civic registration.
"""

import os
import datetime
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from common import (
    BASE_DIR, UPLOADS_DIR, load_model, draw_boxes, to_rgb,
    build_detections, overall_severity, analyze, save_report,
    extract_gps_from_exif, geocode_address, reverse_geocode, get_ip_geolocation, inject_custom_css
)
try:
    from common import INFER_IMGSZ
except ImportError:
    INFER_IMGSZ = 960

try:
    from streamlit_js_eval import get_geolocation
    JS_GEO_AVAILABLE = True
except ImportError:
    JS_GEO_AVAILABLE = False

st.set_page_config(page_title="Detect Potholes | Pothole AI", page_icon="🔍", layout="wide")
inject_custom_css()

# ── Sidebar Controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Detection Settings")
    conf = st.slider("Confidence Threshold", 0.10, 1.00, 0.35, 0.05)

    st.markdown("---")
    st.markdown("### 📍 Location Tag")
    loc_mode = st.radio(
        "Location Method",
        ["📡 Live Browser GPS", "📍 Search Address / Street", "⚙️ Manual Coordinates"],
        index=0,
        horizontal=False,
        help="Live Browser GPS queries your device's exact location and auto-resolves Street / Area / City."
    )

    ip_lat, ip_lon, ip_city, ip_road = get_ip_geolocation()

    if loc_mode == "📡 Live Browser GPS":
        browser_coords = None
        if JS_GEO_AVAILABLE:
            browser_coords = get_geolocation(component_key="p1_browser_gps")

        if browser_coords and "coords" in browser_coords:
            c = browser_coords["coords"]
            c_lat = float(c["latitude"])
            c_lon = float(c["longitude"])
            acc = round(float(c.get("accuracy", 10)))
            auto_road, auto_addr = reverse_geocode(c_lat, c_lon)
            st.session_state["det_lat"] = c_lat
            st.session_state["det_lon"] = c_lon
            st.session_state["street_area_city"] = auto_road
            st.success(f"📡 Browser GPS Live (±{acc}m): `{c_lat:.5f}, {c_lon:.5f}`")
            st.info(f"🏛️ **Auto-Resolved:** **{auto_road}**")
        else:
            c_lat = st.session_state.get("det_lat", ip_lat)
            c_lon = st.session_state.get("det_lon", ip_lon)
            auto_road, auto_addr = reverse_geocode(c_lat, c_lon)
            st.session_state["street_area_city"] = auto_road
            st.info(f"📡 Browser GPS Active — Position: `{c_lat:.5f}, {c_lon:.5f}`")
            st.write(f"🏛️ **Auto-Resolved:** **{auto_road}**")
            st.caption("ℹ️ Click 'Allow' when your browser prompts for location permission.")

        c1, c2 = st.columns(2)
        with c1:
            c_lat = st.number_input("Latitude", value=c_lat, format="%.5f", key="p1_bgps_lat")
        with c2:
            c_lon = st.number_input("Longitude", value=c_lon, format="%.5f", key="p1_bgps_lon")

    elif loc_mode == "📍 Search Address / Street":
        init_val = st.session_state.get("street_area_city", ip_road)
        addr_input = st.text_input("Street / Area / City", value=init_val, placeholder="e.g. Raopura Road, Vadodara")
        geo = None
        if addr_input.strip():
            with st.spinner("Locating..."):
                geo = geocode_address(addr_input)
        if geo:
            c_lat, c_lon = geo[0], geo[1]
            st.session_state["det_lat"] = c_lat
            st.session_state["det_lon"] = c_lon
            st.session_state["street_area_city"] = geo[3] or addr_input
            st.success(f"Located: {c_lat:.4f}, {c_lon:.4f}")
        else:
            if addr_input.strip():
                st.warning("⚠️ Address not found in map database. Kept live Browser GPS coordinates instead:")
            c_lat = st.session_state.get("det_lat", ip_lat)
            c_lon = st.session_state.get("det_lon", ip_lon)
            auto_road, _ = reverse_geocode(c_lat, c_lon)
            st.session_state["street_area_city"] = auto_road
            st.info(f"📍 **Browser GPS Coordinates:** `{c_lat:.5f}, {c_lon:.5f}` — **{auto_road}**")

        c1, c2 = st.columns(2)
        with c1:
            c_lat = st.number_input("Latitude", value=c_lat, format="%.5f", key="p1_search_lat")
        with c2:
            c_lon = st.number_input("Longitude", value=c_lon, format="%.5f", key="p1_search_lon")

    else:
        c1, c2 = st.columns(2)
        with c1:
            c_lat = st.number_input("Latitude", value=st.session_state.get("det_lat", ip_lat), format="%.5f", key="p1_man_lat")
        with c2:
            c_lon = st.number_input("Longitude", value=st.session_state.get("det_lon", ip_lon), format="%.5f", key="p1_man_lon")
        auto_road, _ = reverse_geocode(c_lat, c_lon)
        st.session_state["street_area_city"] = auto_road
        st.caption(f"📍 Location: **{auto_road}**")

    if st.button("📡 Refresh Live Browser GPS", key="p1_btn_refresh_gps"):
        st.session_state.pop("det_lat", None)
        st.session_state.pop("det_lon", None)
        st.session_state.pop("street_area_city", None)
        st.rerun()

# ── Page Header ───────────────────────────────────────────────────────────────
st.title("🔍 Pothole Detection & Road Inspection")
st.markdown("Upload a road photo to detect surface damage, measure cavity dimensions, and log reports to the civic registry.")

model = load_model()
class_names = model.names

uploaded = st.file_uploader(
    "Choose a road photo...",
    type=["jpg", "jpeg", "png", "webp"],
    help="Smartphone photos with GPS location enabled will be tagged automatically!"
)

if uploaded is not None:
    # 1. Check EXIF GPS
    uploaded.seek(0)
    exif_lat, exif_lon = extract_gps_from_exif(uploaded)
    uploaded.seek(0)

    if exif_lat is not None and exif_lon is not None:
        lat, lon = exif_lat, exif_lon
        st.success(f"📍 GPS coordinates extracted from camera EXIF: **{lat:.5f}, {lon:.5f}**")
        with st.spinner("Resolving Street / Area / City via GPS..."):
            auto_road, auto_addr = reverse_geocode(lat, lon)
        road_name = auto_road
        address = auto_addr
        st.session_state["street_area_city"] = road_name
        st.info(f"🏛️ **Auto-Identified Location (Street / Area / City):** **{road_name}** (`{address}`)")
    else:
        lat, lon = c_lat, c_lon
        with st.spinner("Resolving Street / Area / City from coordinates..."):
            auto_road, auto_addr = reverse_geocode(lat, lon)
        road_name = auto_road or st.session_state.get("street_area_city", "Street / Area / City")
        address = auto_addr or f"{road_name} ({lat:.5f}, {lon:.5f})"
        st.session_state["street_area_city"] = road_name
        st.info(f"📍 **Auto-Identified Location (Street / Area / City):** **{road_name}** (`{lat:.5f}, {lon:.5f}`)")

    # 2. Decode Image
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("Could not decode the uploaded image.")
        st.stop()

    h, w = img_bgr.shape[:2]
    
    # 3. YOLO Inference
    results = model.predict(img_bgr, imgsz=INFER_IMGSZ, conf=conf, verbose=False)
    ann_bgr = draw_boxes(img_bgr, results, conf)
    dets = build_detections(results, conf, class_names, h, w)

    # 4. Canvas Comparison
    col1, col2 = st.columns(2)
    with col1:
        st.image(to_rgb(img_bgr), caption="Original Road Photo", use_container_width=True)
    with col2:
        st.image(to_rgb(ann_bgr), caption=f"AI Annotated Detections (conf ≥ {conf:.2f})", use_container_width=True)

    # 5. Metrics
    confs = [d["confidence"] for d in dets]
    areas = [d["area_pct"]   for d in dets]
    sev   = overall_severity(confs)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Potholes Detected", len(dets))
    m2.metric("Avg Confidence",   f"{np.mean(confs):.1%}" if confs else "—")
    m3.metric("Max Area %",       f"{max(areas):.2f}%" if areas else "—")
    m4.metric("Road Severity",    sev)

    if dets:
        st.dataframe(dets, use_container_width=True)
    else:
        st.warning("No potholes detected above the current confidence threshold.")

    if st.button("🔬 Run Detailed Diagnostic Analysis", type="secondary"):
        analyze(img_bgr, results, conf, class_names)

    # 6. Save Report
    st.divider()
    st.markdown("### 📍 Location: Street / Area / City")
    user_road_input = st.text_input(
        "Street / Area / City",
        value=road_name,
        help="Auto-filled from GPS coordinates. Visible and editable before saving to database."
    )
    if user_road_input and user_road_input.strip():
        road_name = user_road_input.strip()
        st.session_state["street_area_city"] = road_name

    st.caption(f"Full Resolved Address: *{address}*  |  GPS: `{lat:.5f}, {lon:.5f}`")
    st.write(f"**Verified Target Road:** **{road_name}**")
    st.caption("Saves the annotated image evidence and publishes coordinates to the city heatmap.")
    
    if st.button("💾 Save Report with Photo Evidence", type="primary", disabled=len(dets) == 0):
        avg_conf = float(np.mean(confs)) if confs else 0.0
        max_area = float(max(areas))     if areas else 0.0

        ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_name = f"detect_{ts_str}_{len(dets)}potholes.jpg"
        saved_rel = os.path.join("uploads", saved_name)
        cv2.imwrite(os.path.join(BASE_DIR, saved_rel), ann_bgr)

        save_report(lat, lon, address, road_name, len(dets), avg_conf, max_area, sev, dets, image_path=saved_rel)
        st.success(f"✅ Report recorded! {len(dets)} pothole(s) at {road_name or address} archived with photo.")
        st.balloons()
else:
    st.info("⬆️ Drag and drop or browse a road photograph above to start detection.")
