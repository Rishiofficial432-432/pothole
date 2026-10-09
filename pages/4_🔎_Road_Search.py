"""
pages/4_🔎_Road_Search.py
=========================
Dedicated Page 4: Road Search & Visual Pothole Evidence Gallery (No Map Plot).
Direct photographic inspection of road damage with AI bounding boxes.
"""

import os
import numpy as np
import streamlit as st

from common import (
    search_reports, update_status, status_badge,
    get_pothole_image, STATUS_OPTIONS, overall_severity,
    inject_custom_css, PD_AVAILABLE
)

if PD_AVAILABLE:
    import pandas as pd

st.set_page_config(page_title="Road Search & Photo Gallery | Pothole AI", page_icon="🔎", layout="wide")
inject_custom_css()

st.title("🔎 Road Search & Photographic Evidence Gallery")
st.markdown("Search any road, street, or national highway to immediately view **actual photographic evidence** of detected potholes with AI annotations.")

# ── Search Input ──────────────────────────────────────────────────────────────
query = st.text_input(
    "Search Road",
    placeholder="Type a road name (e.g.,  MG Road  /  Baner  /  NH-48  /  FC Road  /  Pune)",
    label_visibility="collapsed"
)

if query.strip():
    results = search_reports(query)

    if not results:
        st.warning(f"🔍 No pothole reports found for **‘{query}’**.\n\nTry a broader search term like `Pune`, `MG Road`, `NH-48`, or `Baner`.")
    else:
        # ── Road Executive Summary ────────────────────────────────────────────
        total_ph = sum(r["num_potholes"] for r in results)
        avg_conf = float(np.mean([r["avg_conf"] for r in results]))
        sev_list = [r["severity"] for r in results]
        high_c   = sev_list.count("High")
        fixed_c  = sum(1 for r in results if r.get("status") == "Fixed")
        open_c   = len(results) - fixed_c
        overall  = ("Critical" if high_c >= len(results) // 2
                    else overall_severity([r["avg_conf"] for r in results]))

        st.markdown(f"### 🛣️ Road Summary: **{query}**")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Reports",  len(results))
        c2.metric("Total Potholes", total_ph)
        c3.metric("Avg Confidence", f"{avg_conf:.1%}")
        c4.metric("Open Issues",    open_c)
        c5.metric("Fixed Issues",   fixed_c)

        cond_color = {
            "High": "🔴", "Critical": "🚨",
            "Medium": "🟠", "Low": "🟢", "None": "⚫"
        }.get(overall, "⚫")
        st.markdown(
            f"**Overall Road Health Condition:** {cond_color} `{overall}`  |  "
            f"Last Reported: `{results[0]['ts']}`"
        )

        st.divider()

        # ── Visual Pothole Evidence Gallery (NO MAP) ──────────────────────────
        st.markdown(f"### 📸 Pothole Visual Evidence on **{query}** ({len(results)} Report{'s' if len(results) > 1 else ''})")
        st.caption("AI-annotated camera evidence showing exact damage locations, cavity bounding boxes, and severity ratings.")

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
                        st.info("📷 Photographic evidence is being archived.")

                with c_meta:
                    st.markdown(f"""
- 🛣️ **Road Segment:** `{r.get('road_name') or 'N/A'}`
- 📍 **Full Address:** {loc}
- 🧭 **GPS Coordinates:** `{r['lat']:.5f}, {r['lon']:.5f}`
- 🕳️ **Potholes Detected:** **{r['num_potholes']}**
- 🎯 **AI Confidence:** **{r['avg_conf']:.1%}**
- ⚠️ **Severity Rating:** **{r['severity']}**
- 🗓️ **Recorded On:** `{r['ts']}`
                    """)

                    st.write("**Civic Status Action:**")
                    upd_c1, upd_c2 = st.columns([2, 1])
                    with upd_c1:
                        new_st = st.selectbox(
                            "Change Status",
                            STATUS_OPTIONS,
                            index=STATUS_OPTIONS.index(cur_stat),
                            key=f"search_status_{r['id']}",
                            label_visibility="collapsed",
                        )
                    with upd_c2:
                        if st.button("Update", key=f"upd_{r['id']}", use_container_width=True):
                            update_status(r["id"], new_st)
                            st.success(f"Updated #{r['id']} to '{new_st}'")
                            st.rerun()

                st.markdown("<hr style='margin: 0.8rem 0; border: none; border-top: 1px solid #2d3748;'>", unsafe_allow_html=True)

        # ── Breakdown Charts ──────────────────────────────────────────────────
        if PD_AVAILABLE and len(results) > 1:
            st.divider()
            df_s = pd.DataFrame(results)
            ch1, ch2 = st.columns(2)
            with ch1:
                st.markdown("#### 📋 Civic Status Breakdown")
                st.bar_chart(df_s["status"].value_counts())
            with ch2:
                st.markdown("#### ⚠️ Severity Breakdown")
                st.bar_chart(df_s["severity"].value_counts())
else:
    st.info("💡 Type any road or highway name above to search for its pothole photo records.")
    
    st.markdown("#### ⚡ Popular Indian Roads to Try:")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1:
        st.markdown("- **MG Road** (Pune / Bengaluru)")
        st.markdown("- **Baner Road** (Pune)")
    with col_p2:
        st.markdown("- **FC Road** (Pune)")
        st.markdown("- **Marine Drive** (Mumbai)")
    with col_p3:
        st.markdown("- **NH-48** (Highway)")
        st.markdown("- **NH-44** (National Corridor)")
    with col_p4:
        st.markdown("- **Outer Ring Road** (Bengaluru)")
        st.markdown("- **Connaught Place** (Delhi)")
