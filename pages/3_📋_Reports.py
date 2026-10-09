"""
pages/3_📋_Reports.py
=====================
Dedicated Page 3: Full audit trail of logged road incidents, photo expanders, and status workflows.
"""

import os
import sqlite3
import streamlit as st

from common import (
    load_reports, update_status, status_badge,
    get_pothole_image, STATUS_OPTIONS, DB_PATH,
    inject_custom_css, PD_AVAILABLE
)

if PD_AVAILABLE:
    import pandas as pd

st.set_page_config(page_title="Civic Reports Ledger | Pothole AI", page_icon="📋", layout="wide")
inject_custom_css()

st.title("📋 Civic Reports Ledger & Status Manager")
st.markdown("Chronological registry of road surface damage reports, photographic evidence, and municipal repair workflows.")

reports = load_reports()

if not reports:
    st.info("ℹ️ No reports in the ledger yet.")
else:
    # ── Top Bar with Filter & Export ──────────────────────────────────────────
    col_search, col_filter, col_export = st.columns([2, 1, 1])
    with col_search:
        filter_text = st.text_input("Filter reports by road/city", placeholder="e.g. Pune, MG Road, NH-48")
    with col_filter:
        status_filter = st.selectbox("Status Filter", ["All"] + STATUS_OPTIONS)
    with col_export:
        st.write("")
        st.write("")
        if PD_AVAILABLE:
            df_export = pd.DataFrame(reports)
            csv_data = df_export.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Export CSV", data=csv_data, file_name="pothole_reports.csv", mime="text/csv", use_container_width=True)

    # Apply filters
    filtered = reports
    if filter_text.strip():
        ft = filter_text.strip().lower()
        filtered = [r for r in filtered if ft in (r.get("road_name") or "").lower() or ft in (r.get("address") or "").lower()]
    if status_filter != "All":
        filtered = [r for r in filtered if r.get("status") == status_filter]

    st.caption(f"Showing **{len(filtered)}** of {len(reports)} total records")
    st.markdown("---")

    # ── Render Report Cards ───────────────────────────────────────────────────
    for r in filtered:
        sev_class = r["severity"].lower()
        loc = r["address"] or f"{r['lat']:.5f}, {r['lon']:.5f}"
        road_disp = f" • Road: <b>{r['road_name']}</b>" if r.get("road_name") else ""
        cur_status = r.get("status", "Reported")

        st.markdown(f"""
<div class="report-card {sev_class}">
  <b>Report #{r['id']} — {loc}</b>{road_disp} &nbsp;|&nbsp; 🗓️ {r['ts']} {status_badge(cur_status)}<br>
  🕳️ Potholes: <b>{r['num_potholes']}</b> &nbsp;•&nbsp;
  🎯 AI Confidence: <b>{r['avg_conf']:.1%}</b> &nbsp;•&nbsp;
  ⚠️ Severity: <b>{r['severity']}</b> &nbsp;•&nbsp;
  🧭 GPS: <code>{r['lat']:.5f}, {r['lon']:.5f}</code>
</div>
""", unsafe_allow_html=True)

        # Photo Viewer Expander
        img_file = get_pothole_image(r.get("image_path"), r.get("road_name", ""), r.get("address", ""))
        if img_file and os.path.exists(img_file):
            with st.expander(f"📸 Inspect Pothole Photograph (Report #{r['id']})", expanded=False):
                st.image(img_file, caption=f"Report #{r['id']} — {r.get('road_name') or loc} ({r['num_potholes']} potholes)", use_container_width=True)

        # Status Modifier
        c_status, c_btn = st.columns([3, 1])
        with c_status:
            try:
                status_idx = STATUS_OPTIONS.index(cur_status)
            except (ValueError, IndexError):
                status_idx = 0
            new_st = st.selectbox(
                f"Update status #{r['id']}",
                STATUS_OPTIONS,
                index=status_idx,
                key=f"rep_status_{r['id']}",
                label_visibility="collapsed"
            )
        with c_btn:
            if st.button("Update Status", key=f"btn_upd_{r['id']}", use_container_width=True):
                update_status(r["id"], new_st)
                st.success(f"Status for #{r['id']} updated to '{new_st}'")
                st.rerun()

    # ── Database Purge ────────────────────────────────────────────────────────
    st.divider()
    with st.expander("⚠️ Administrative Database Management"):
        st.warning("Clearing reports is irreversible.")
        if st.button("🗑️ Clear ALL Reports from Database", type="secondary"):
            con = sqlite3.connect(DB_PATH)
            con.execute("DELETE FROM reports")
            con.commit()
            con.close()
            st.success("All reports cleared.")
            st.rerun()
