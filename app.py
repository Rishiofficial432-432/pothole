import os
import json
import sqlite3
import datetime
from io import BytesIO

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ── optional geocoding dep ────────────────────────────────────────────────────
try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut
    GEO_AVAILABLE = True
except ImportError:
    GEO_AVAILABLE = False

# ── optional map deps ─────────────────────────────────────────────────────────
try:
    import folium
    from folium.plugins import HeatMap, MarkerCluster
    from streamlit_folium import st_folium
    MAP_AVAILABLE = True
except ImportError:
    MAP_AVAILABLE = False

try:
    import pandas as pd
    PD_AVAILABLE = True
except ImportError:
    PD_AVAILABLE = False

from common import reverse_geocode, get_ip_geolocation
try:
    from streamlit_js_eval import get_geolocation
    JS_GEO_AVAILABLE = True
except ImportError:
    JS_GEO_AVAILABLE = False

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
_model_env = os.environ.get("POTHOLE_MODEL", "best.pt")
MODEL_PATH = _model_env if os.path.isabs(_model_env) else os.path.join(BASE_DIR, _model_env)
DB_PATH    = os.path.join(BASE_DIR, "potholes.db")

# ── CARTO Map Config ──────────────────────────────────────────────────────────
CARTO_API_KEY = os.environ.get("CARTO_API_KEY", "cb1_42v7_1_2dc6571c05f28bc6841e688e")
CARTO_DARK_TILES = f"https://{{s}}.basemaps.cartocdn.com/rastertiles/dark_all/{{z}}/{{x}}/{{y}}.png?key={CARTO_API_KEY}"
CARTO_ATTR = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'

def create_folium_map(location, zoom_start=13):
    """Creates a Folium map using authenticated CARTO Dark Matter basemap (watermark-free)."""
    m = folium.Map(location=location, zoom_start=zoom_start, tiles=None)
    folium.TileLayer(
        tiles=CARTO_DARK_TILES,
        attr=CARTO_ATTR,
        name="CartoDB Dark Matter",
        subdomains="abcd",
        max_zoom=20
    ).add_to(m)
    return m

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pothole Detector & Reporter",
    page_icon="🛣️",
    layout="wide",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stMetricValue"] { font-size: 2rem; }
  .report-card {
    background: #1e1e2e; border-radius: 12px; padding: 1rem;
    margin-bottom: 0.6rem; border-left: 4px solid #7c3aed;
  }
  .high    { border-color: #ef4444; }
  .medium  { border-color: #f59e0b; }
  .low     { border-color: #22c55e; }
  .badge {
    display: inline-block; border-radius: 6px; padding: 2px 10px;
    font-size: 0.78rem; font-weight: bold; margin-left: 6px;
  }
  .badge-reported   { background:#3b82f6; color:#fff; }
  .badge-critical   { background:#ef4444; color:#fff; }
  .badge-review     { background:#f59e0b; color:#000; }
  .badge-fixed      { background:#22c55e; color:#000; }
  .search-result {
    background: #16213e; border-radius: 10px; padding: 1rem;
    margin-bottom: 0.5rem; border-left: 4px solid #3b82f6;
  }
  .bulk-card {
    background: #0d1b2a; border-radius: 10px; padding: 0.8rem;
    margin-bottom: 0.4rem; border-left: 4px solid #7c3aed;
    font-size: 0.9rem;
  }
  .bulk-ok   { border-color: #22c55e; }
  .bulk-skip { border-color: #6b7280; }
  .bulk-err  { border-color: #ef4444; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# GPS EXIF HELPER
# ─────────────────────────────────────────────────────────────────────────────
def _dms_to_decimal(dms, ref):
    """Convert EXIF GPS DMS tuple to decimal degrees."""
    try:
        d, m, s = dms
        # Each value may be an IFDRational or a tuple (num, denom)
        def to_float(v):
            if hasattr(v, 'numerator'):
                return v.numerator / v.denominator if v.denominator else 0.0
            if isinstance(v, tuple):
                return v[0] / v[1] if v[1] else 0.0
            return float(v)
        dec = to_float(d) + to_float(m) / 60 + to_float(s) / 3600
        if ref in ("S", "W"):
            dec = -dec
        return dec
    except Exception:
        return None


def extract_gps_from_exif(img_file) -> tuple:
    """
    Try to read GPS lat/lon from JPEG/PNG EXIF.
    Returns (lat, lon) or (None, None).
    """
    try:
        from PIL.ExifTags import TAGS, GPSTAGS
        img = Image.open(img_file)
        exif_raw = img._getexif()  # returns dict or None
        if not exif_raw:
            return None, None
        # Map tag IDs to names
        exif = {TAGS.get(k, k): v for k, v in exif_raw.items()}
        gps_info_raw = exif.get("GPSInfo")
        if not gps_info_raw:
            return None, None
        gps = {GPSTAGS.get(k, k): v for k, v in gps_info_raw.items()}
        lat = _dms_to_decimal(gps.get("GPSLatitude"), gps.get("GPSLatitudeRef", "N"))
        lon = _dms_to_decimal(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef", "E"))
        return lat, lon
    except Exception:
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────────────────────────────
STATUS_OPTIONS = ["Reported", "Critical", "Under Review", "In Progress", "Fixed"]

def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            ts           TEXT NOT NULL,
            lat          REAL NOT NULL,
            lon          REAL NOT NULL,
            address      TEXT,
            road_name    TEXT,
            num_potholes INTEGER,
            avg_conf     REAL,
            max_area_pct REAL,
            severity     TEXT,
            status       TEXT DEFAULT 'Reported',
            detections   TEXT,
            image_path   TEXT
        )
    """)
    # Migrate existing DB if columns missing
    for col, definition in [
        ("road_name", "TEXT"),
        ("status", "TEXT DEFAULT 'Reported'"),
        ("image_path", "TEXT"),
    ]:
        try:
            con.execute(f"ALTER TABLE reports ADD COLUMN {col} {definition}")
        except Exception:
            pass
    con.commit()
    con.close()


def save_report(lat, lon, address, road_name, num, avg_conf, max_area, sev, detections, image_path=None):
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        INSERT INTO reports
          (ts, lat, lon, address, road_name, num_potholes, avg_conf,
           max_area_pct, severity, status, detections, image_path)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.datetime.now().isoformat(timespec="seconds"),
        lat, lon, address, road_name, num,
        round(avg_conf, 4), round(max_area, 4),
        sev, "Reported", json.dumps(detections), image_path,
    ))
    con.commit()
    con.close()


def load_reports():
    con = sqlite3.connect(DB_PATH)
    rows = con.execute("""
        SELECT id, ts, lat, lon, address, road_name, num_potholes,
               avg_conf, max_area_pct, severity, status, detections, image_path
        FROM reports ORDER BY ts DESC
    """).fetchall()
    con.close()
    cols = ["id", "ts", "lat", "lon", "address", "road_name", "num_potholes",
            "avg_conf", "max_area_pct", "severity", "status", "detections", "image_path"]
    return [dict(zip(cols, r)) for r in rows]


def search_reports(query: str):
    """Fuzzy search by address or road_name (case-insensitive LIKE)."""
    q = f"%{query.strip()}%"
    con = sqlite3.connect(DB_PATH)
    rows = con.execute("""
        SELECT id, ts, lat, lon, address, road_name, num_potholes,
               avg_conf, max_area_pct, severity, status, detections, image_path
        FROM reports
        WHERE lower(address) LIKE lower(?)
           OR lower(road_name) LIKE lower(?)
        ORDER BY ts DESC
    """, (q, q)).fetchall()
    con.close()
    cols = ["id", "ts", "lat", "lon", "address", "road_name", "num_potholes",
            "avg_conf", "max_area_pct", "severity", "status", "detections", "image_path"]
    return [dict(zip(cols, r)) for r in rows]


def get_pothole_image(image_path: str, road_name: str = "", address: str = ""):
    """Returns a valid file path to display for the pothole report."""
    if image_path:
        full_p = os.path.join(BASE_DIR, image_path) if not os.path.isabs(image_path) else image_path
        if os.path.exists(full_p):
            return full_p
    # Fallback to realistic annotated sample based on road type
    name_str = f"{road_name} {address}".lower()
    if any(k in name_str for k in ["nh", "expressway", "highway", "bypass"]):
        sample_p = os.path.join(BASE_DIR, "sample_images", "sample_highway_pothole_annotated.jpg")
    else:
        sample_p = os.path.join(BASE_DIR, "sample_images", "sample_city_pothole_annotated.jpg")
    if os.path.exists(sample_p):
        return sample_p
    return None


def update_status(report_id: int, new_status: str):
    con = sqlite3.connect(DB_PATH)
    con.execute("UPDATE reports SET status=? WHERE id=?", (new_status, report_id))
    con.commit()
    con.close()


@st.cache_data(show_spinner=False, ttl=300)
def geocode_address(address: str):
    """Return (lat, lon, display_name, road) or None via Nominatim."""
    if not GEO_AVAILABLE or not address.strip():
        return None
    try:
        geolocator = Nominatim(user_agent="pothole_reporter_app")
        loc = geolocator.geocode(address, timeout=5)
        if loc:
            raw = loc.raw.get("address", {})
            road = (raw.get("road") or raw.get("pedestrian")
                    or raw.get("footway") or raw.get("street") or "")
            return loc.latitude, loc.longitude, loc.address, road
    except Exception:
        pass
    return None


def status_badge(status: str) -> str:
    cls = {"Reported": "badge-reported", "Critical": "badge-critical",
           "Under Review": "badge-review", "Fixed": "badge-fixed"}.get(status, "badge-reported")
    icon = {"Reported": "📋", "Critical": "🚨", "Under Review": "🔍", "Fixed": "✅"}.get(status, "📋")
    return f'<span class="badge {cls}">{icon} {status}</span>'


init_db()

# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def draw_boxes(img_bgr, results, conf):
    ann = img_bgr.copy()
    names = results[0].names
    for box in results[0].boxes:
        if box.conf[0].item() < conf:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        label = names[int(box.cls[0].item())]
        score = box.conf[0].item()
        color = (0, 200, 80)
        cv2.rectangle(ann, (x1, y1), (x2, y2), color, 3)
        text = f"{label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(ann, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
        cv2.putText(ann, text, (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    return ann


def to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def severity(score):
    if score >= 0.80:
        return "High"
    if score >= 0.60:
        return "Medium"
    return "Low"


def overall_severity(scores):
    if not scores:
        return "None"
    return severity(float(np.mean(scores)))


def build_detections(results, conf, class_names, img_h, img_w):
    rows = []
    for b in results[0].boxes:
        s = b.conf[0].item()
        if s < conf:
            continue
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        rows.append({
            "class":      class_names[int(b.cls[0].item())],
            "confidence": round(s, 3),
            "severity":   severity(s),
            "bbox":       [round(v, 1) for v in [x1, y1, x2, y2]],
            "area_px":    round((x2 - x1) * (y2 - y1)),
            "area_pct":   round((x2 - x1) * (y2 - y1) / (img_h * img_w) * 100, 3),
        })
    return rows


def analyze(img_bgr, results, conf, class_names):
    """Original detailed analysis panel."""
    h, w = img_bgr.shape[:2]
    boxes = results[0].boxes
    rows = []
    for b in boxes:
        score = b.conf[0].item()
        if score < conf:
            continue
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        rows.append({
            "Class":            class_names[int(b.cls[0].item())],
            "Confidence":       round(score, 3),
            "Severity":         severity(score),
            "Width_px":         round(x2 - x1, 1),
            "Height_px":        round(y2 - y1, 1),
            "Area_px":          round((x2 - x1) * (y2 - y1)),
            "Area_%_of_image":  round((x2 - x1) * (y2 - y1) / (w * h) * 100, 2),
        })

    st.subheader("Analysis")
    if not rows:
        st.info("No detections above the threshold, so nothing to analyze.")
        return

    if PD_AVAILABLE:
        import pandas as pd
        df = pd.DataFrame(rows)
        confs = df["Confidence"].tolist()
        avg_area_pct = df["Area_%_of_image"].mean()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total potholes", len(df))
        m2.metric("Avg confidence", f"{np.mean(confs):.3f}")
        m3.metric("Avg area (% img)", f"{avg_area_pct:.2f}%")
        biggest = df.loc[df["Area_px"].idxmax()]
        m4.metric("Largest pothole size", f"{int(biggest['Area_px'])} px")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Confidence distribution**")
            st.bar_chart(df["Confidence"].value_counts().sort_index())
        with c2:
            st.markdown("**Severity breakdown**")
            st.bar_chart(df["Severity"].value_counts())

        st.markdown("**Threshold sensitivity** (detections kept at each cutoff)")
        sens = []
        for thr in np.arange(0.05, 1.0, 0.05):
            sens.append({"Threshold": round(thr, 2),
                         "Detections": int((df["Confidence"] >= thr).sum())})
        st.line_chart(pd.DataFrame(sens).set_index("Threshold"))

        sector = pd.cut(
            df["Area_%_of_image"],
            bins=[0, 0.5, 2, 5, 100],
            labels=["Very small", "Small", "Medium", "Large"],
        )
        df["Size bucket"] = sector
        st.markdown("**Pothole size buckets**")
        st.dataframe(df, use_container_width=True)
        st.markdown("**Top 5 most confident detections**")
        st.dataframe(df.nlargest(5, "Confidence"), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# APP HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("🛣️ Pothole Detection & Community Reporter")
st.caption("Detect potholes, tag their GPS location, and build a city-wide heatmap for civic action.")

model      = load_model()
class_names = model.names

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    conf = st.slider("Confidence threshold", 0.05, 1.0, 0.35, 0.05,
                     help="Lower = detect more potholes (may add false positives). 0.35–0.45 is recommended for v3.")
    iou_thresh = st.slider("IoU threshold (NMS)", 0.1, 0.9, 0.45, 0.05,
                            help="Lower = suppress overlapping/duplicate boxes. Reduces hallucination.")
    st.info(f"Model classes: {list(class_names.values())}")

    st.divider()
    st.header("📍 Location Tag")
    loc_mode = st.radio(
        "Location Method",
        ["📡 Live Browser GPS", "📍 Search Address / Street", "⚙️ Manual Coordinates"],
        index=0,
        horizontal=False,
        help="Live Browser GPS queries your device's exact location and auto-resolves Street / Area / City."
    )

    road_name_sidebar = ""
    ip_lat, ip_lon, ip_city, ip_road = get_ip_geolocation()

    if loc_mode == "📡 Live Browser GPS":
        browser_coords = None
        if JS_GEO_AVAILABLE:
            browser_coords = get_geolocation(component_key="app_browser_gps")

        if browser_coords and "coords" in browser_coords:
            c = browser_coords["coords"]
            lat = float(c["latitude"])
            lon = float(c["longitude"])
            acc = round(float(c.get("accuracy", 10)))
            auto_road, auto_addr = reverse_geocode(lat, lon)
            road_name_sidebar = auto_road
            address = auto_addr
            st.session_state["det_lat"] = lat
            st.session_state["det_lon"] = lon
            st.session_state["street_area_city"] = road_name_sidebar
            st.success(f"📡 Browser GPS Live (±{acc}m): `{lat:.5f}, {lon:.5f}`")
            st.info(f"🏛️ **Auto-Resolved:** **{road_name_sidebar}**")
        else:
            lat = st.session_state.get("det_lat", ip_lat)
            lon = st.session_state.get("det_lon", ip_lon)
            auto_road, auto_addr = reverse_geocode(lat, lon)
            road_name_sidebar = auto_road
            address = auto_addr
            st.session_state["street_area_city"] = road_name_sidebar
            st.info(f"📡 Browser GPS Active — Position: `{lat:.5f}, {lon:.5f}`")
            st.write(f"🏛️ **Auto-Resolved:** **{road_name_sidebar}**")
            st.caption("ℹ️ Click 'Allow' when your browser prompts for location permission.")

        c1, c2 = st.columns(2)
        with c1:
            lat = st.number_input("Latitude", value=lat, format="%.5f", key="bgps_lat")
        with c2:
            lon = st.number_input("Longitude", value=lon, format="%.5f", key="bgps_lon")

    elif loc_mode == "📍 Search Address / Street":
        init_val = st.session_state.get("street_area_city", ip_road)
        address_query = st.text_input("Street / Area / City",
                                      value=init_val,
                                      placeholder="e.g. Raopura Road, Vadodara or MG Road, Pune")
        geo_result = None
        if address_query.strip():
            with st.spinner("Locating..."):
                geo_result = geocode_address(address_query)

        if geo_result:
            lat, lon, full_addr, road_name_sidebar = geo_result
            st.session_state["det_lat"] = lat
            st.session_state["det_lon"] = lon
            st.session_state["street_area_city"] = road_name_sidebar
            address = full_addr
            st.success(f"📍 {full_addr[:60]}..." if len(full_addr) > 60 else f"📍 {full_addr}")
            st.caption(f"Road: **{road_name_sidebar}**  |  {lat:.5f}, {lon:.5f}")
        else:
            if address_query.strip():
                st.warning("⚠️ Address not found in map database. Kept live Browser GPS coordinates instead:")
            lat = st.session_state.get("det_lat", ip_lat)
            lon = st.session_state.get("det_lon", ip_lon)
            auto_road, auto_addr = reverse_geocode(lat, lon)
            road_name_sidebar = auto_road
            address = auto_addr
            st.session_state["street_area_city"] = road_name_sidebar
            st.info(f"📍 **Browser GPS Coordinates:** `{lat:.5f}, {lon:.5f}` — **{road_name_sidebar}**")

        c1, c2 = st.columns(2)
        with c1:
            lat = st.number_input("Latitude", value=st.session_state.get("det_lat", ip_lat), format="%.5f", key="search_lat")
        with c2:
            lon = st.number_input("Longitude", value=st.session_state.get("det_lon", ip_lon), format="%.5f", key="search_lon")

    else:
        c1, c2 = st.columns(2)
        with c1:
            lat = st.number_input("Latitude", value=st.session_state.get("det_lat", ip_lat), format="%.5f", key="man_lat")
        with c2:
            lon = st.number_input("Longitude", value=st.session_state.get("det_lon", ip_lon), format="%.5f", key="man_lon")
        auto_road, auto_addr = reverse_geocode(lat, lon)
        road_name_sidebar = auto_road
        address = auto_addr
        st.session_state["street_area_city"] = road_name_sidebar
        st.caption(f"📍 Location: **{road_name_sidebar}**")

    if st.button("📡 Refresh Live Browser GPS", key="btn_refresh_gps"):
        st.session_state.pop("det_lat", None)
        st.session_state.pop("det_lon", None)
        st.session_state.pop("street_area_city", None)
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_detect, tab_map, tab_reports, tab_search, tab_bulk = st.tabs(
    ["🔍 Detect", "🗺️ Heatmap", "📋 Reports", "🔎 Search Road", "🗂️ Bulk Process"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DETECTION
# ══════════════════════════════════════════════════════════════════════════════
with tab_detect:
    st.subheader("Upload a road image")
    uploaded = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        # 1. Check EXIF GPS
        uploaded.seek(0)
        exif_lat, exif_lon = extract_gps_from_exif(uploaded)
        uploaded.seek(0)

        if exif_lat is not None and exif_lon is not None:
            active_lat, active_lon = exif_lat, exif_lon
            with st.spinner("Resolving Street / Area / City from photo GPS..."):
                auto_road, auto_addr = reverse_geocode(active_lat, active_lon)
            active_road = auto_road
            active_addr = auto_addr
            st.session_state["street_area_city"] = active_road
            st.success(f"📍 GPS coordinates from EXIF: **{active_lat:.5f}, {active_lon:.5f}** — Auto-Identified Location: **{active_road}**")
        else:
            active_lat, active_lon = lat, lon
            if road_name_sidebar:
                active_road = road_name_sidebar
                active_addr = address or f"{active_road} ({active_lat:.5f}, {active_lon:.5f})"
            else:
                with st.spinner("Resolving Street / Area / City from coordinates..."):
                    auto_road, auto_addr = reverse_geocode(active_lat, active_lon)
                active_road = auto_road
                active_addr = auto_addr
            st.session_state["street_area_city"] = active_road
            st.info(f"📍 **Auto-Identified Location (Street / Area / City):** **{active_road}** (`{active_lat:.5f}, {active_lon:.5f}`)")

        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is None:
            st.error("Could not read the uploaded image.")
            st.stop()

        h, w    = img_bgr.shape[:2]
        results = model.predict(img_bgr, conf=conf, iou=iou_thresh, verbose=False)
        ann_bgr = draw_boxes(img_bgr, results, conf)
        dets    = build_detections(results, conf, class_names, h, w)

        col1, col2 = st.columns(2)
        with col1:
            st.image(to_rgb(img_bgr), caption="Original", use_container_width=True)
        with col2:
            st.image(to_rgb(ann_bgr),
                     caption=f"Detections (conf ≥ {conf})",
                     use_container_width=True)

        confs = [d["confidence"] for d in dets]
        areas = [d["area_pct"]   for d in dets]
        sev   = overall_severity(confs)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Potholes",         len(dets))
        m2.metric("Avg Confidence",   f"{np.mean(confs):.3f}" if confs else "—")
        m3.metric("Avg Area %",       f"{np.mean(areas):.2f}%" if areas else "—")
        m4.metric("Overall Severity", sev)

        if dets:
            st.dataframe(dets, use_container_width=True)
        else:
            st.warning("No potholes detected above the confidence threshold.")

        if st.button("Run Analysis", type="primary"):
            analyze(img_bgr, results, conf, class_names)

        # ── save report ───────────────────────────────────────────────────────
        st.divider()
        st.markdown("### 📍 Location: Street / Area / City")
        user_loc_name = st.text_input(
            "Street / Area / City",
            value=active_road,
            help="Auto-filled from GPS coordinates. Visible and editable before saving to database."
        )
        if user_loc_name and user_loc_name.strip():
            active_road = user_loc_name.strip()
            st.session_state["street_area_city"] = active_road

        st.caption(f"Full Resolved Address: *{active_addr}*  |  GPS: `{active_lat:.5f}, {active_lon:.5f}`")
        st.write(f"**Verified Target Road:** **{active_road}**")

        if st.button("💾 Save report to heatmap & database", type="primary",
                     disabled=len(dets) == 0):
            avg_conf = float(np.mean(confs)) if confs else 0.0
            max_area = float(max(areas))     if areas else 0.0

            # Save annotated image to uploads
            os.makedirs(os.path.join(BASE_DIR, "uploads"), exist_ok=True)
            ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            saved_name = f"detect_{ts_str}_{len(dets)}potholes.jpg"
            saved_rel = os.path.join("uploads", saved_name)
            cv2.imwrite(os.path.join(BASE_DIR, saved_rel), ann_bgr)

            save_report(active_lat, active_lon, active_addr, active_road,
                        len(dets), avg_conf, max_area, sev, dets,
                        image_path=saved_rel)
            st.success(
                f"✅ Report saved! {len(dets)} pothole(s) at {active_road} "
                "recorded with detection photograph."
            )
            st.balloons()
    else:
        st.info("⬆️  Upload a road photograph to get started.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
with tab_map:
    reports = load_reports()

    if not MAP_AVAILABLE:
        st.warning(
            "Map libraries not installed.\n\n"
            "Run:  `pip install folium streamlit-folium`  then restart the app."
        )
    elif not reports:
        st.info("No reports yet. Detect potholes and save a report to see the map.")
    else:
        st.subheader(f"🗺️ Community Pothole Heatmap  —  {len(reports)} reports")

        heat_data = [
            [r["lat"], r["lon"],
             min(r["num_potholes"] * r["avg_conf"], 10)]
            for r in reports
        ]
        lats = [r["lat"] for r in reports]
        lons = [r["lon"] for r in reports]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        center_lat = np.mean(lats)
        center_lon = np.mean(lons)

        m = create_folium_map(location=[center_lat, center_lon], zoom_start=5)
        if len(reports) > 1 and (max_lat - min_lat > 0.001 or max_lon - min_lon > 0.001):
            m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]], padding=[30, 30])

        HeatMap(
            heat_data, radius=25, blur=20, max_zoom=16,
            gradient={0.2: "blue", 0.5: "lime", 0.8: "orange", 1.0: "red"},
            name="Heatmap"
        ).add_to(m)

        mc = MarkerCluster(name="Incident Pins").add_to(m)

        for r in reports:
            sev_color = {"High": "red", "Medium": "orange",
                         "Low": "green", "None": "gray"}.get(r["severity"], "blue")
            loc_label = r.get("road_name") or r.get("address") or "Road Segment"
            popup_html = (
                f"<div style='font-family: sans-serif; font-size: 12px; color: #1e293b; min-width: 200px;'>"
                f"<b style='font-size: 13px;'>{loc_label}</b><br>"
                f"Potholes: <b style='color: #dc2626;'>{r['num_potholes']}</b><br>"
                f"Severity: <b>{r['severity']}</b><br>"
                f"Confidence: <b>{r['avg_conf']:.1%}</b><br>"
                f"Status: <b>{r.get('status', 'Reported')}</b><br>"
                f"<small style='color: #64748b;'>{r['ts']}</small>"
                f"</div>"
            )
            folium.Marker(
                location=[r["lat"], r["lon"]],
                icon=folium.Icon(color=sev_color, icon="exclamation-sign", prefix="glyphicon"),
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"#{r['id']} {loc_label}: {r['num_potholes']} pothole(s) [{r['severity']}]",
            ).add_to(mc)

        st_folium(m, use_container_width=True, height=560)

        if PD_AVAILABLE:
            df_r = pd.DataFrame(reports)
            st.divider()
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Total reports",       len(reports))
            s2.metric("Total potholes",      int(df_r["num_potholes"].sum()))
            s3.metric("High severity",       int((df_r["severity"] == "High").sum()))
            s4.metric("Avg potholes/report", f"{df_r['num_potholes'].mean():.1f}")
            st.markdown("**Severity breakdown across all reports**")
            st.bar_chart(df_r["severity"].value_counts())

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — REPORTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_reports:
    reports = load_reports()
    st.subheader(f"📋 Saved Reports  ({len(reports)} total)")

    if not reports:
        st.info("No reports saved yet.")
    else:
        for r in reports:
            sev_class = r["severity"].lower()
            loc = r["address"] or f"{r['lat']:.5f}, {r['lon']:.5f}"
            road_disp = f" • Road: <b>{r['road_name']}</b>" if r.get("road_name") else ""
            st.markdown(f"""
<div class="report-card {sev_class}">
  <b>#{r['id']} — {loc}</b>{road_disp} &nbsp;|&nbsp; {r['ts']}{status_badge(r.get('status','Reported'))}<br>
  🕳️ Potholes: <b>{r['num_potholes']}</b> &nbsp;
  🎯 Avg Conf: <b>{r['avg_conf']:.3f}</b> &nbsp;
  ⚠️ Severity: <b>{r['severity']}</b>
</div>
""", unsafe_allow_html=True)
            r_img = get_pothole_image(r.get("image_path"), r.get("road_name", ""), r.get("address", ""))
            if r_img and os.path.exists(r_img):
                with st.expander(f"📸 View Pothole Photo (Report #{r['id']})", expanded=False):
                    st.image(r_img, caption=f"Report #{r['id']} — {r.get('road_name') or loc}", use_container_width=True)

            new_status = st.selectbox(
                f"Update status for report #{r['id']}",
                STATUS_OPTIONS,
                index=STATUS_OPTIONS.index(r.get("status", "Reported")) if r.get("status", "Reported") in STATUS_OPTIONS else 0,
                key=f"status_{r['id']}",
                label_visibility="collapsed",
            )
            if new_status != r.get("status", "Reported"):
                update_status(r["id"], new_status)
                st.rerun()

        if st.button("🗑️ Clear ALL reports", type="secondary"):
            con = sqlite3.connect(DB_PATH)
            con.execute("DELETE FROM reports")
            con.commit()
            con.close()
            st.success("All reports cleared.")
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ROAD SEARCH
# ══════════════════════════════════════════════════════════════════════════════
with tab_search:
    st.subheader("🔎 Search Any Road or Location")
    st.caption("Type a road name, area, or city to instantly see all pothole reports and their current status.")

    query = st.text_input(
        "Search",
        placeholder="e.g.  MG Road  /  Baner  /  Pune  /  NH-48",
        label_visibility="collapsed",
    )

    if query.strip():
        results = search_reports(query)

        if not results:
            st.warning(f"🔍 No pothole reports found for **‘{query}’**.\n\nTry a broader keyword or detect & report potholes first.")
        else:
            # ── Road-level summary ───────────────────────────────────────────────────
            total_ph  = sum(r["num_potholes"] for r in results)
            avg_conf  = float(np.mean([r["avg_conf"] for r in results]))
            sev_list  = [r["severity"] for r in results]
            high_c    = sev_list.count("High")
            fixed_c   = sum(1 for r in results if r.get("status") == "Fixed")
            open_c    = len(results) - fixed_c
            overall   = ("Critical" if high_c >= len(results) // 2
                         else overall_severity([r["avg_conf"] for r in results]))

            st.markdown(f"### 🛣️ Road Report: **{query}**")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Reports",  len(results))
            c2.metric("Total Potholes", total_ph)
            c3.metric("Avg Confidence", f"{avg_conf:.3f}")
            c4.metric("Open Issues",    open_c)
            c5.metric("Fixed",          fixed_c)

            # Road condition badge
            cond_color = {
                "High": "🔴", "Critical": "🚨",
                "Medium": "🟠", "Low": "🟢", "None": "⚫"
            }.get(overall, "⚫")
            st.markdown(
                f"**Overall Road Condition:** {cond_color} `{overall}`  |  "
                f"Last reported: `{results[0]['ts']}`"
            )

            # ── Visual Pothole Evidence Gallery (No Map) ─────────────────────────
            st.divider()
            st.markdown(f"### 📸 Pothole Visual Evidence on **{query}** ({len(results)} Report{'s' if len(results) > 1 else ''})")
            st.caption("AI-detected pothole imagery with bounding boxes, severity assessment, and verification status.")

            for r in results:
                sev_cls  = r["severity"].lower()
                loc      = r["address"] or f"{r['lat']:.5f}, {r['lon']:.5f}"
                road_tag = f" • <b>{r['road_name']}</b>" if r.get("road_name") else ""
                cur_stat = r.get("status", "Reported")
                img_path = get_pothole_image(r.get("image_path"), r.get("road_name", ""), r.get("address", ""))

                with st.container():
                    st.markdown(f"""
<div class="search-result {sev_cls}">
  <span style="font-size: 1.15rem; font-weight: 700;">Report #{r['id']}</span> — {loc}{road_tag} {status_badge(cur_stat)}
</div>
""", unsafe_allow_html=True)

                    c_img, c_meta = st.columns([1.2, 1.0])

                    with c_img:
                        if img_path and os.path.exists(img_path):
                            st.image(
                                img_path,
                                caption=f"Pothole Evidence #{r['id']} — {r.get('road_name') or loc} ({r['num_potholes']} detected)",
                                use_container_width=True
                            )
                        else:
                            st.info("📷 Photographic evidence is being processed.")

                    with c_meta:
                        st.markdown(f"""
- 🛣️ **Road Segment:** `{r.get('road_name') or 'N/A'}`
- 📍 **Location:** {loc}
- 🧭 **GPS Coordinates:** `{r['lat']:.5f}, {r['lon']:.5f}`
- 🕳️ **Potholes Detected:** **{r['num_potholes']}**
- 🎯 **AI Confidence:** **{r['avg_conf']:.2%}**
- ⚠️ **Severity Rating:** **{r['severity']}**
- 🗓️ **Recorded On:** `{r['ts']}`
                        """)

                        st.write("**Civic Status Action:**")
                        upd_c1, upd_c2 = st.columns([2, 1])
                        with upd_c1:
                            new_st = st.selectbox(
                                "Change status",
                                STATUS_OPTIONS,
                                index=STATUS_OPTIONS.index(cur_stat) if cur_stat in STATUS_OPTIONS else 0,
                                key=f"search_status_{r['id']}",
                                label_visibility="collapsed",
                            )
                        with upd_c2:
                            if st.button("Update", key=f"upd_{r['id']}", use_container_width=True):
                                update_status(r["id"], new_st)
                                st.success(f"Updated #{r['id']} to '{new_st}'")
                                st.rerun()

                    st.markdown("<hr style='margin: 0.8rem 0; border: none; border-top: 1px solid #2d3748;'>", unsafe_allow_html=True)

            # ── Status breakdown chart ────────────────────────────────────────
            if PD_AVAILABLE and len(results) > 1:
                st.divider()
                df_s = pd.DataFrame(results)
                ch1, ch2 = st.columns(2)
                with ch1:
                    st.markdown("**Current Status Breakdown**")
                    st.bar_chart(df_s["status"].value_counts())
                with ch2:
                    st.markdown("**Severity Breakdown**")
                    st.bar_chart(df_s["severity"].value_counts())
    else:
        st.info("🔍 Type a road name or area above to search for pothole reports.\n\n"
                "**Examples:** `MG Road` • `NH-48` • `Baner, Pune` • `Delhi` • `Koramangala`")
        if not load_reports():
            st.caption("No data in the database yet. Go to **🔍 Detect** tab, upload images, and save reports first.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — BULK PROCESS
# ══════════════════════════════════════════════════════════════════════════════
with tab_bulk:
    st.subheader("🗂️ Bulk Process — Upload Many Images at Once")
    st.caption(
        "Upload dozens or hundreds of road photos. GPS is read automatically from photo EXIF "
        "(smartphone photos embed GPS). If photos have no GPS, set a fallback location below."
    )

    # ─ Settings row ───────────────────────────────────────────────────────────
    bk1, bk2, bk3, bk4 = st.columns(4)
    with bk1:
        bulk_conf = st.slider("Min confidence", 0.05, 1.0, 0.45, 0.05, key="bulk_conf")
    with bk2:
        skip_no_gps = st.checkbox("Skip images without GPS", value=False)
    with bk3:
        fallback_lat = st.number_input("Fallback Latitude",  value=18.5204,
                                       format="%.5f", key="blat")
    with bk4:
        fallback_lon = st.number_input("Fallback Longitude", value=73.8567,
                                       format="%.5f", key="blon")

    bulk_address_input = st.text_input(
        "Fallback address / area label (used when no EXIF GPS)",
        placeholder="e.g. Baner Road, Pune",
        key="bulk_addr",
    )

    # Auto-geocode fallback address if provided
    bulk_road_name = ""
    if bulk_address_input.strip() and GEO_AVAILABLE:
        geo = geocode_address(bulk_address_input)
        if geo:
            fallback_lat, fallback_lon = geo[0], geo[1]
            bulk_road_name = geo[3]
            st.caption(f"📍 Geocoded: {geo[2][:70]} | Road: **{bulk_road_name}**")

    # ─ File uploader ────────────────────────────────────────────────────────────
    bulk_files = st.file_uploader(
        "Upload road images (any number)",
        type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="bulk_uploader",
    )

    if bulk_files:
        st.info(f"📂 {len(bulk_files)} image(s) selected. Click **Run Bulk Detection** to start.")
        if st.button("🚀 Run Bulk Detection", type="primary", key="run_bulk"):

            # Counters
            saved_count  = 0
            skipped_gps  = 0
            no_pothole   = 0
            error_count  = 0
            saved_latlons = []

            progress_bar = st.progress(0, text="Starting...")
            log_area     = st.empty()
            log_lines    = []

            for idx, f in enumerate(bulk_files):
                pct  = int((idx + 1) / len(bulk_files) * 100)
                name = f.name
                progress_bar.progress(pct, text=f"Processing {idx+1}/{len(bulk_files)}: {name}")

                try:
                    raw = f.read()
                    # ―― GPS from EXIF ――――――――――――――――――――――――――――――――――――――――
                    from io import BytesIO as _BIO
                    exif_lat, exif_lon = extract_gps_from_exif(_BIO(raw))
                    gps_source = "EXIF"

                    if exif_lat is None or exif_lon is None:
                        if skip_no_gps:
                            skipped_gps += 1
                            log_lines.append(
                                f'<div class="bulk-card bulk-skip">⏩ <b>{name}</b> — No GPS in EXIF, skipped.</div>'
                            )
                            log_area.markdown("\n".join(log_lines[-20:]), unsafe_allow_html=True)
                            continue
                        # use fallback
                        exif_lat, exif_lon = fallback_lat, fallback_lon
                        gps_source = "fallback"

                    # ―― Decode image ――――――――――――――――――――――――――――――――――――――――
                    arr     = np.asarray(bytearray(raw), dtype=np.uint8)
                    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img_bgr is None:
                        raise ValueError("cv2 could not decode image")

                    h, w = img_bgr.shape[:2]

                    # ―― YOLO inference ――――――――――――――――――――――――――――――――――――――
                    res  = model.predict(img_bgr, conf=bulk_conf, iou=iou_thresh, verbose=False)
                    dets = build_detections(res, bulk_conf, class_names, h, w)

                    if not dets:
                        no_pothole += 1
                        log_lines.append(
                            f'<div class="bulk-card bulk-skip">🟢 <b>{name}</b> — '
                            f'No potholes detected (conf≥{bulk_conf}). GPS={gps_source}</div>'
                        )
                        log_area.markdown("\n".join(log_lines[-20:]), unsafe_allow_html=True)
                        continue

                    # ―― Reverse-geocode road name ―――――――――――――――――――――――――――
                    r_road = bulk_road_name
                    addr   = bulk_address_input
                    if gps_source == "EXIF" and GEO_AVAILABLE:
                        geo_r = geocode_address(f"{exif_lat:.5f} {exif_lon:.5f}")
                        if geo_r:
                            addr   = geo_r[2][:80]
                            r_road = geo_r[3]

                    # ―― Save to DB ―――――――――――――――――――――――――――――――――――――――――
                    confs_b = [d["confidence"] for d in dets]
                    areas_b = [d["area_pct"]   for d in dets]
                    sev_b   = overall_severity(confs_b)

                    # Save annotated bulk image to uploads
                    os.makedirs(os.path.join(BASE_DIR, "uploads"), exist_ok=True)
                    ann_b = draw_boxes(img_bgr, res, bulk_conf)
                    b_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    clean_name = os.path.splitext(os.path.basename(name))[0]
                    b_rel = os.path.join("uploads", f"bulk_{b_ts}_{saved_count}_{clean_name}.jpg")
                    cv2.imwrite(os.path.join(BASE_DIR, b_rel), ann_b)

                    save_report(
                        exif_lat, exif_lon, addr, r_road,
                        len(dets),
                        float(np.mean(confs_b)),
                        float(max(areas_b)),
                        sev_b, dets,
                        image_path=b_rel,
                    )
                    saved_count += 1
                    saved_latlons.append((exif_lat, exif_lon, len(dets), sev_b))

                    log_lines.append(
                        f'<div class="bulk-card bulk-ok">✅ <b>{name}</b> — '
                        f'{len(dets)} pothole(s), {sev_b} severity | '
                        f'GPS: {exif_lat:.4f},{exif_lon:.4f} ({gps_source}) | '
                        f'Road: {r_road or addr[:30]}</div>'
                    )

                except Exception as ex:
                    error_count += 1
                    log_lines.append(
                        f'<div class="bulk-card bulk-err">❌ <b>{name}</b> — Error: {ex}</div>'
                    )

                log_area.markdown("\n".join(log_lines[-20:]), unsafe_allow_html=True)

            progress_bar.progress(100, text="✔️ Done!")

            # ─ Final summary ───────────────────────────────────────────────────
            st.success(
                f"✅ Bulk complete! 🕳️ Saved: {saved_count} — "
                f"🟢 No potholes: {no_pothole} — "
                f"🟡 Skipped (no GPS): {skipped_gps} — "
                f"🔴 Errors: {error_count}"
            )

            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric("Reports saved",   saved_count)
            sm2.metric("No potholes",     no_pothole)
            sm3.metric("GPS skipped",     skipped_gps)
            sm4.metric("Errors",          error_count)

            # ─ Live map of bulk results ────────────────────────────────────────
            if MAP_AVAILABLE and saved_latlons:
                st.markdown("### 🗺️ All Newly Detected Potholes on Map")
                c_lat = np.mean([x[0] for x in saved_latlons])
                c_lon = np.mean([x[1] for x in saved_latlons])
                bm = create_folium_map(location=[c_lat, c_lon], zoom_start=14)
                for lat_b, lon_b, n_ph, sev_b in saved_latlons:
                    mc = {"High": "red", "Medium": "orange",
                          "Low": "green"}.get(sev_b, "blue")
                    folium.CircleMarker(
                        location=[lat_b, lon_b], radius=9,
                        color=mc, fill=True, fill_opacity=0.85,
                        tooltip=f"{n_ph} pothole(s) — {sev_b}",
                    ).add_to(bm)
                st_folium(bm, use_container_width=True, height=420)

            if saved_count:
                st.balloons()
    else:
        st.markdown("""
### ℹ️ How it works

| Step | What happens |
|---|---|
| 1️⃣ Upload photos | Select as many road photos as you want (JPEG from phone works best) |
| 2️⃣ GPS extracted | GPS is read from photo EXIF metadata automatically |
| 3️⃣ YOLO runs | Your trained `best.pt` detects potholes in every image |
| 4️⃣ Road name | Location is reverse-geocoded via OpenStreetMap |
| 5️⃣ Saved to DB | All results saved to `potholes.db` with lat/lon, severity, confidence |
| 6️⃣ Map shown | All new potholes appear on the live map instantly |

> 💡 **Tip:** Photos taken on a **smartphone with Location turned ON** will have GPS embedded automatically.
> No GPS in photo? Set the **Fallback location** above and all photos will use that location.
""")