"""
common.py
==========
Shared utilities, database methods, model loaders, and UI components
for the multi-page Pothole AI & Civic Road Intelligence System.
"""

import os
import json
import sqlite3
import datetime
from io import BytesIO
from functools import lru_cache

import requests
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ── Optional Dependencies ─────────────────────────────────────────────────────
try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut
    GEO_AVAILABLE = True
except ImportError:
    GEO_AVAILABLE = False

try:
    import folium
    from folium.plugins import HeatMap
    from streamlit_folium import st_folium
    MAP_AVAILABLE = True
except ImportError:
    MAP_AVAILABLE = False

try:
    import pandas as pd
    PD_AVAILABLE = True
except ImportError:
    PD_AVAILABLE = False

# ── Project Paths ─────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
_model_env  = os.environ.get("POTHOLE_MODEL", "best.pt")
MODEL_PATH  = _model_env if os.path.isabs(_model_env) else os.path.join(BASE_DIR, _model_env)
# Inference size: use the same size the model was trained at (notebook default = 960).
INFER_IMGSZ = int(os.environ.get("POTHOLE_IMGSZ", "960"))
DB_PATH     = os.path.join(BASE_DIR, "potholes.db")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
SAMPLES_DIR = os.path.join(BASE_DIR, "sample_images")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

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

# ── Custom CSS ────────────────────────────────────────────────────────────────
def inject_custom_css():
    st.markdown("""
<style>
  [data-testid="stMetricValue"] { font-size: 2rem; font-weight: 700; }
  .report-card {
    background: #1e1e2e; border-radius: 12px; padding: 1.2rem;
    margin-bottom: 0.8rem; border-left: 5px solid #7c3aed;
  }
  .high    { border-color: #ef4444 !important; }
  .medium  { border-color: #f59e0b !important; }
  .low     { border-color: #22c55e !important; }
  .badge {
    display: inline-block; border-radius: 6px; padding: 3px 10px;
    font-size: 0.78rem; font-weight: bold; margin-left: 6px;
  }
  .badge-reported   { background:#3b82f6; color:#fff; }
  .badge-critical   { background:#ef4444; color:#fff; }
  .badge-review     { background:#f59e0b; color:#000; }
  .badge-fixed      { background:#22c55e; color:#000; }
  .search-result {
    background: #16213e; border-radius: 12px; padding: 1.2rem;
    margin-bottom: 1rem; border-left: 5px solid #3b82f6;
  }
  .bulk-card {
    background: #0d1b2a; border-radius: 10px; padding: 0.8rem;
    margin-bottom: 0.4rem; border-left: 4px solid #7c3aed;
    font-size: 0.9rem;
  }
  .bulk-ok   { border-color: #22c55e; }
  .bulk-skip { border-color: #6b7280; }
  .bulk-err  { border-color: #ef4444; }
  .hero-card {
    background: linear-gradient(135deg, #1e1b4b 0%, #1e293b 100%);
    border-radius: 16px; padding: 1.8rem; margin-bottom: 1.5rem;
    border: 1px solid rgba(255,255,255,0.08);
  }
</style>
""", unsafe_allow_html=True)

# ── GPS EXIF Helpers ──────────────────────────────────────────────────────────
def _dms_to_decimal(dms, ref):
    """Convert EXIF GPS DMS tuple to decimal degrees."""
    try:
        d, m, s = dms
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
    """Try to read GPS lat/lon from JPEG/PNG EXIF. Returns (lat, lon) or (None, None)."""
    try:
        from PIL.ExifTags import TAGS, GPSTAGS
        img = Image.open(img_file)
        exif_raw = img._getexif()
        if not exif_raw:
            return None, None
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

# ── Database Operations ───────────────────────────────────────────────────────
STATUS_OPTIONS = ["Reported", "In Progress", "Under Review", "Critical", "Repaired", "Fixed"]

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
        sample_p = os.path.join(SAMPLES_DIR, "sample_highway_pothole_annotated.jpg")
    else:
        sample_p = os.path.join(SAMPLES_DIR, "sample_city_pothole_annotated.jpg")
    if os.path.exists(sample_p):
        return sample_p
    return None


def update_status(report_id: int, new_status: str):
    con = sqlite3.connect(DB_PATH)
    con.execute("UPDATE reports SET status=? WHERE id=?", (new_status, report_id))
    con.commit()
    con.close()


def delete_report(report_id: int):
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM reports WHERE id=?", (report_id,))
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


@lru_cache(maxsize=16)
def get_ip_geolocation():
    """
    Fallback geolocation based on client IP network when device browser GPS is unavailable.
    Returns (lat, lon, city, road_name).
    """
    try:
        r = requests.get("http://ip-api.com/json/?fields=status,country,regionName,city,lat,lon", timeout=3.0)
        if r.status_code == 200:
            d = r.json()
            if d.get("status") == "success":
                lat = float(d.get("lat"))
                lon = float(d.get("lon"))
                city = d.get("city") or "Vadodara"
                road, _ = reverse_geocode(lat, lon)
                return lat, lon, city, road
    except Exception:
        pass
    return 22.3008, 73.2043, "Vadodara", "Raopura Road / Nava Bazaar / Vadodara"


@lru_cache(maxsize=512)
def reverse_geocode(lat: float, lon: float):
    """
    Reverse-geocodes (lat, lon) to Street / Area / City format:
        (road_name, full_address)
    e.g. ('Shivaji Road / Kasba Peth / Pune', 'Shivaji Road, Kasba Peth, Pune, Maharashtra, 411001, India')
    """
    if not GEO_AVAILABLE or lat is None or lon is None:
        return f"Road Corridor ({lat:.4f}, {lon:.4f})", f"GPS ({lat:.5f}, {lon:.5f})"
    try:
        geolocator = Nominatim(user_agent="civicroad_ai_pothole_detector_v2", timeout=4)
        loc = geolocator.reverse((lat, lon), timeout=4, language="en")
        if loc and loc.raw:
            raw_addr = loc.raw.get("address", {})
            street = (raw_addr.get("road") or raw_addr.get("pedestrian") or
                      raw_addr.get("street") or raw_addr.get("footway") or
                      raw_addr.get("highway") or raw_addr.get("path") or "").strip()
            area = (raw_addr.get("suburb") or raw_addr.get("neighbourhood") or
                    raw_addr.get("residential") or raw_addr.get("subdistrict") or
                    raw_addr.get("quarter") or raw_addr.get("district") or "").strip()
            city = (raw_addr.get("city") or raw_addr.get("town") or
                    raw_addr.get("village") or raw_addr.get("municipality") or
                    raw_addr.get("county") or raw_addr.get("city_district") or "").strip()

            parts = []
            if street:
                parts.append(street)
            if area and (not parts or area.lower() not in parts[0].lower()):
                parts.append(area)
            if city and (not parts or city.lower() not in parts[-1].lower()):
                parts.append(city)

            if parts:
                road_name = " / ".join(parts)
            else:
                road_name = f"Corridor ({lat:.4f}, {lon:.4f})"

            address = loc.address or f"{road_name} ({lat:.4f}, {lon:.4f})"
            return road_name, address
    except Exception:
        pass
    return f"Road Corridor ({lat:.4f}, {lon:.4f})", f"GPS ({lat:.5f}, {lon:.5f})"




def status_badge(status: str) -> str:
    cls = {
        "Reported": "badge-reported", "Critical": "badge-critical",
        "Under Review": "badge-review", "In Progress": "badge-review",
        "Fixed": "badge-fixed", "Repaired": "badge-fixed"
    }.get(status, "badge-reported")
    icon = {
        "Reported": "📋", "Critical": "🚨", "Under Review": "🔍",
        "In Progress": "🔄", "Fixed": "✅", "Repaired": "✅"
    }.get(status, "📋")
    return f'<span class="badge {cls}">{icon} {status}</span>'


init_db()

# ── Model Loader ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

# ── Computer Vision & Detection Helpers ───────────────────────────────────────
def draw_boxes(img_bgr, results, conf):
    ann = img_bgr.copy()
    h, w = ann.shape[:2]
    scale = max(h, w) / 1000.0
    thickness = max(2, int(scale * 2.2))
    font_scale = max(0.5, scale * 0.55)
    names = results[0].names
    for box in results[0].boxes:
        score = float(box.conf[0].item())
        if score < conf:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0].item())
        label = names.get(cls_id, "pothole")
        
        # Color coding by confidence
        if score >= 0.60:
            color = (30, 30, 235)   # Red
        elif score >= 0.30:
            color = (20, 160, 245)  # Amber
        else:
            color = (40, 200, 60)   # Green
            
        # Semi-transparent overlay inside the cavity
        overlay = ann.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.22, ann, 0.78, 0, ann)
        
        # Solid bounding rectangle
        cv2.rectangle(ann, (x1, y1), (x2, y2), color, thickness)
        
        # Badge label
        text = f"{label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, int(thickness / 2)))
        tag_h = int(th * 1.5)
        tag_w = int(tw + scale * 8)
        tag_y1 = max(0, y1 - tag_h)
        cv2.rectangle(ann, (x1, tag_y1), (x1 + tag_w, y1), color, -1)
        cv2.putText(ann, text, (x1 + int(scale * 4), y1 - int(th * 0.25)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), max(1, int(thickness / 2)))
    return ann


def to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def severity(score):
    if score >= 0.65:
        return "High"
    if score >= 0.35:
        return "Medium"
    return "Low"


def overall_severity(scores):
    if not scores:
        return "None"
    max_s = float(max(scores))
    if max_s >= 0.65 or len(scores) >= 4:
        return "High"
    if max_s >= 0.35 or len(scores) >= 2:
        return "Medium"
    return "Low"


def build_detections(results, conf, class_names, img_h, img_w):
    rows = []
    for b in results[0].boxes:
        s = float(b.conf[0].item())
        if s < conf:
            continue
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        c_name = class_names[int(b.cls[0].item())]
        rows.append({
            "class":      c_name,
            "class_name": c_name,
            "confidence": round(s, 3),
            "severity":   severity(s),
            "bbox":       [round(v, 1) for v in [x1, y1, x2, y2]],
            "area_px":    round((x2 - x1) * (y2 - y1)),
            "area_pct":   round((x2 - x1) * (y2 - y1) / (img_h * img_w) * 100, 3),
        })
    return rows


def analyze(img_bgr, results, conf, class_names):
    """Detailed diagnostic metrics."""
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

    st.subheader("Diagnostic Breakdown")
    if not rows:
        st.info("No detections above the threshold to analyze.")
        return

    st.dataframe(rows, use_container_width=True)
    confs = [r["Confidence"] for r in rows]
    areas = [r["Area_%_of_image"] for r in rows]
    c1, c2, c3 = st.columns(3)
    c1.metric("Highest Conf", f"{max(confs):.3f}")
    c2.metric("Lowest Conf",  f"{min(confs):.3f}")
    c3.metric("Max Area %",   f"{max(areas):.2f}%")
