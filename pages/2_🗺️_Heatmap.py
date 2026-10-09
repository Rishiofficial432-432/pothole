"""
pages/2_🗺️_Heatmap.py
=====================
Dedicated Page 2: National & city-wide spatial GIS heatmap using authenticated CARTO Dark Matter.
Features interactive marker clustering, auto-framing, city quick-filters, and high-visibility incident pins.
"""

import numpy as np
import streamlit as st

from common import (
    load_reports, create_folium_map, inject_custom_css,
    MAP_AVAILABLE, PD_AVAILABLE
)

if MAP_AVAILABLE:
    import folium
    from folium.plugins import HeatMap, MarkerCluster
    from streamlit_folium import st_folium

if PD_AVAILABLE:
    import pandas as pd

st.set_page_config(page_title="Pothole Heatmap | Pothole AI", page_icon="🗺️", layout="wide")
inject_custom_css()

st.title("🗺️ National & Municipal Pothole Heatmap")
st.markdown("Real-time geospatial damage dispersion rendered over authenticated **CARTO Dark Matter** GIS tiles with interactive incident pins.")

reports = load_reports()

if not MAP_AVAILABLE:
    st.error("Map components are not available. Please install `folium` and `streamlit-folium`.")
    st.stop()

if not reports:
    st.info("ℹ️ No pothole reports recorded yet. Upload photos in the Detect or Bulk Process pages to build the heatmap.")
else:
    # ── Top Metrics ───────────────────────────────────────────────────────────
    total_ph = sum(r["num_potholes"] for r in reports)
    high_sev = sum(1 for r in reports if r["severity"] == "High")
    fixed_cnt = sum(1 for r in reports if r.get("status") in ("Fixed", "Repaired"))
    open_cnt  = len(reports) - fixed_cnt

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Incident Sites", len(reports))
    k2.metric("Total Potholes Mapped", total_ph)
    k3.metric("High Severity Hazards", high_sev)
    k4.metric("Civic Resolution Rate", f"{(fixed_cnt / len(reports)):.1%}" if reports else "0%")

    st.markdown("---")

    # ── City & Display Filters ────────────────────────────────────────────────
    # Extract unique city names from addresses
    city_set = set()
    for r in reports:
        addr = r.get("address", "") or ""
        parts = [p.strip() for p in addr.split(",") if p.strip()]
        if len(parts) >= 2:
            city_name = parts[-2]
            if city_name and not city_name.isdigit() and len(city_name) > 2:
                city_set.add(city_name)
    sorted_cities = sorted(list(city_set))

    f_col1, f_col2, f_col3 = st.columns([2, 1.5, 1.5])
    with f_col1:
        city_options = ["All India (All Reports)"] + sorted_cities
        selected_city = st.selectbox("📍 Focus Region / City", city_options, index=0)
    with f_col2:
        sev_filter = st.selectbox("⚠️ Severity Filter", ["All Severities", "High", "Medium", "Low"], index=0)
    with f_col3:
        layer_mode = st.selectbox("🗺️ Map Display Mode", ["Both (Pins + Heatmap)", "Interactive Pins Only", "Heatmap Only"], index=0)

    # Filter reports
    filtered_reports = reports
    if selected_city != "All India (All Reports)":
        filtered_reports = [r for r in filtered_reports if selected_city.lower() in (r.get("address", "") or "").lower()]
    if sev_filter != "All Severities":
        filtered_reports = [r for r in filtered_reports if r.get("severity") == sev_filter]

    if not filtered_reports:
        st.warning(f"No reports match the current filter selection ({selected_city}, {sev_filter}). Showing all reports instead.")
        filtered_reports = reports

    # ── Map Geometry & View Bounds ────────────────────────────────────────────
    lats = [r["lat"] for r in filtered_reports]
    lons = [r["lon"] for r in filtered_reports]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    center_lat = float(np.mean(lats))
    center_lon = float(np.mean(lons))

    # Zoom calculation
    if selected_city != "All India (All Reports)":
        init_zoom = 13
    else:
        init_zoom = 5

    m = create_folium_map(location=[center_lat, center_lon], zoom_start=init_zoom)

    # Auto fit bounds so ALL pins are guaranteed to be in view
    if len(filtered_reports) > 1 and (max_lat - min_lat > 0.001 or max_lon - min_lon > 0.001):
        m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]], padding=[30, 30])

    # 1. Heat Gradient Layer
    if "Heatmap" in layer_mode:
        heat_data = [
            [r["lat"], r["lon"], min(r["num_potholes"] * r["avg_conf"], 10)]
            for r in filtered_reports
        ]
        HeatMap(
            heat_data,
            radius=25,
            blur=20,
            max_zoom=16,
            gradient={0.2: "blue", 0.5: "lime", 0.8: "orange", 1.0: "red"},
            name="Density Heatmap"
        ).add_to(m)

    # 2. Interactive Map Pins Layer (MarkerCluster)
    if "Pins" in layer_mode:
        marker_cluster = MarkerCluster(
            name="Pothole Pins",
            overlay=True,
            control=True,
            options={"spiderfyOnMaxZoom": True, "showCoverageOnHover": False, "zoomToBoundsOnClick": True}
        ).add_to(m)

        for r in filtered_reports:
            sev = r.get("severity", "Low")
            icon_color = {
                "High": "red",
                "Medium": "orange",
                "Low": "green",
                "None": "gray"
            }.get(sev, "blue")

            status_disp = r.get("status", "Reported")
            loc_label = r.get("road_name") or r.get("address") or "Unknown Road"

            popup_html = f"""
            <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 12px; line-height: 1.5; min-width: 200px; color: #1e293b;">
                <div style="font-weight: 700; font-size: 13px; color: #0f172a; margin-bottom: 4px; border-bottom: 2px solid #e2e8f0; padding-bottom: 3px;">
                    Report #{r['id']} — {loc_label}
                </div>
                <div><b>📍 GPS:</b> <code>{r['lat']:.5f}, {r['lon']:.5f}</code></div>
                <div><b>🕳️ Potholes Detected:</b> <span style="font-weight: 700; color: #dc2626;">{r['num_potholes']}</span></div>
                <div><b>🎯 AI Confidence:</b> <b>{r['avg_conf']:.1%}</b></div>
                <div><b>⚠️ Severity:</b> <span style="font-weight: 700;">{sev}</span></div>
                <div><b>📋 Status:</b> <b>{status_disp}</b></div>
                <div style="color: #64748b; font-size: 11px; margin-top: 4px;">📅 {r.get('ts', '')}</div>
            </div>
            """

            folium.Marker(
                location=[r["lat"], r["lon"]],
                icon=folium.Icon(color=icon_color, icon="exclamation-sign", prefix="glyphicon"),
                popup=folium.Popup(popup_html, max_width=280),
                tooltip=f"#{r['id']} {loc_label}: {r['num_potholes']} pothole(s) [{sev}]"
            ).add_to(marker_cluster)

    # Render Folium Map in Streamlit
    st.caption(f"Displaying **{len(filtered_reports)}** incident pins across the map. Click on pins or cluster circles to zoom in.")
    st_folium(m, use_container_width=True, height=620)

    # ── Charts ────────────────────────────────────────────────────────────────
    if PD_AVAILABLE:
        st.divider()
        df_r = pd.DataFrame(filtered_reports)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### ⚠️ Hazard Severity Distribution")
            st.bar_chart(df_r["severity"].value_counts())
        with c2:
            st.markdown("#### 📋 Civic Repair Status")
            st.bar_chart(df_r["status"].value_counts())
