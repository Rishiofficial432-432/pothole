"""
Live Ward Data  (pins + live data from kaun.city, no database)

Click the map to drop a pin. The page asks kaun.city's public endpoints which
Bengaluru ward the pin is in and shows live data for it. Nothing is stored:
pins live only in this browser session. No Supabase, no API keys, no changes
to app.py, best.pt or potholes.db.

Endpoints used (all on https://kaun.city):
    POST /api/pin-lookup                     lat/lng -> ward, corporation, constituency
    GET  /api/ward/<corp>/<ward>/live        resident reports / signals (60 s cache)
    GET  /api/ward/<corp>/<ward>             full ward record (optional, on demand)
    GET  /api/map-layers[?layer=potholes]    per-ward values (pothole complaints etc.)
    GET  /api/health                         connection test

Set KAUN_BASE_URL to point at another deployment if you ever self-host.
"""
import os
from urllib.parse import quote

import requests
import streamlit as st

try:
    import folium
    from streamlit_folium import st_folium
    MAP_OK = True
except ImportError:
    MAP_OK = False

try:
    import pandas as pd
except ImportError:
    pd = None

st.set_page_config(page_title="Live Ward Data", page_icon="📍", layout="wide")

BASE = os.environ.get("KAUN_BASE_URL", "https://kaun.city").rstrip("/")
CENTER = (12.9716, 77.5946)           # Bengaluru
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "application/json",
}
TIMEOUT = 20


# ── HTTP helpers ──────────────────────────────────────────────────────────────
def _get(path, params=None):
    r = requests.get(BASE + path, params=params, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def _post(path, payload):
    r = requests.post(BASE + path, json=payload,
                      headers={**HEADERS, "Content-Type": "application/json"}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=3600, show_spinner=False)
def layer_registry():
    return _get("/api/map-layers")


@st.cache_data(ttl=3600, show_spinner=False)
def layer_values(layer_id):
    return _get("/api/map-layers", {"layer": layer_id})


@st.cache_data(ttl=300, show_spinner=False)
def pin_lookup(lat, lng):
    return _post("/api/pin-lookup", {"lat": lat, "lng": lng})


@st.cache_data(ttl=60, show_spinner=False)
def ward_live(corp, ward):
    return _get(f"/api/ward/{quote(str(corp))}/{quote(str(ward))}/live")


@st.cache_data(ttl=300, show_spinner=False)
def ward_record(corp, ward):
    return _get(f"/api/ward/{quote(str(corp))}/{quote(str(ward))}")


def safe(fn, *args):
    """Return (data, error_message)."""
    try:
        return fn(*args), None
    except requests.HTTPError as e:
        code = e.response.status_code if e.response is not None else "?"
        hint = ""
        if code in (401, 403, 429):
            hint = " (kaun.city may be blocking scripted requests or rate-limiting)"
        return None, f"HTTP {code} from {e.request.url}{hint}"
    except requests.RequestException as e:
        return None, f"Could not reach kaun.city: {e.__class__.__name__}"
    except ValueError:
        return None, "kaun.city returned something that is not JSON"


def render_json(obj):
    """Generic renderer: scalars as a small table, nested parts in expanders."""
    if isinstance(obj, dict):
        scalars = {k: v for k, v in obj.items()
                   if v is None or isinstance(v, (str, int, float, bool))}
        nested = {k: v for k, v in obj.items() if k not in scalars}
        if scalars:
            if pd is not None:
                st.dataframe(pd.DataFrame({"field": list(scalars), "value": [str(v) for v in scalars.values()]}),
                             hide_index=True, use_container_width=True)
            else:
                st.json(scalars)
        for k, v in nested.items():
            size = len(v) if hasattr(v, "__len__") else ""
            with st.expander(f"{k}" + (f"  ({size})" if size != "" else "")):
                st.json(v)
    else:
        st.json(obj)


# ── session state (in-memory only) ────────────────────────────────────────────
ss = st.session_state
ss.setdefault("pins", [])            # list of (lat, lng)
ss.setdefault("selected", None)      # (lat, lng)
ss.setdefault("last_click", None)

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Live data source")
    st.caption(BASE)
    if st.button("Test connection"):
        data, err = safe(_get, "/api/health")
        if err:
            st.error(err)
        else:
            st.success("kaun.city reachable")
            st.json(data)
    if st.button("Clear pins"):
        ss.pins, ss.selected, ss.last_click = [], None, None
        st.rerun()
    if st.button("Refresh live data now"):
        ward_live.clear()
        st.rerun()

st.title("📍 Live Ward Data")
st.caption("Click anywhere in Bengaluru to drop a pin. Data comes live from kaun.city; nothing is saved.")

if not MAP_OK:
    st.error("Install the map packages first:  pip install folium streamlit-folium")
    st.stop()

left, right = st.columns([3, 2], gap="large")

# ── map ───────────────────────────────────────────────────────────────────────
with left:
    center = ss.selected or CENTER
    m = folium.Map(location=center, zoom_start=14 if ss.selected else 12, tiles="OpenStreetMap")
    for p in ss.pins:
        is_sel = ss.selected is not None and tuple(p) == tuple(ss.selected)
        folium.Marker(p, icon=folium.Icon(color="red" if is_sel else "blue", icon="map-marker", prefix="fa"),
                      tooltip=f"{p[0]:.5f}, {p[1]:.5f}").add_to(m)
    out = st_folium(m, height=560, use_container_width=True,
                    returned_objects=["last_clicked"], key="kaun_live_map")

    click = (out or {}).get("last_clicked")
    if click:
        pt = (round(click["lat"], 5), round(click["lng"], 5))
        if pt != ss.last_click:
            ss.last_click = pt
            ss.selected = pt
            if pt not in ss.pins:
                ss.pins = (ss.pins + [pt])[-50:]
            st.rerun()

    if ss.pins:
        st.caption("Pins this session (click one location again by dropping a new pin next to it):")
        if pd is not None:
            st.dataframe(pd.DataFrame(ss.pins, columns=["lat", "lng"]), hide_index=True, height=150)

# ── data panel ────────────────────────────────────────────────────────────────
with right:
    if not ss.selected:
        st.info("No pin yet. Click the map.")
        st.stop()

    lat, lng = ss.selected
    st.subheader(f"Pin: {lat:.5f}, {lng:.5f}")

    with st.spinner("Looking up ward..."):
        info, err = safe(pin_lookup, lat, lng)
    if err:
        st.error(err)
        st.markdown(f"You can still open the site directly: [{BASE}]({BASE})")
        st.stop()
    if not info.get("found"):
        st.warning("This spot is outside kaun.city's Bengaluru coverage. Drop a pin inside the city.")
        st.stop()

    corp = info.get("gba_corporation_id")
    ward = info.get("gba_ward_no")
    if corp is None or ward is None:
        st.warning("Found the city, but no current ward for this exact spot.")
        render_json(info)
        st.stop()

    st.markdown(f"### {info.get('gba_ward_name') or 'Ward'}  ·  Ward {ward}")
    c1, c2 = st.columns(2)
    c1.metric("Corporation", str(info.get("gba_corporation") or corp))
    c2.metric("Constituency", str(info.get("gba_ac") or "n/a"))
    pop = info.get("gba_population")
    c3, c4 = st.columns(2)
    c3.metric("Population", f"{int(pop):,}" if isinstance(pop, (int, float)) else "n/a")
    c4.metric("Zone", str(info.get("gba_zone_name") or info.get("gba_zone") or "n/a"))

    # ---- per-ward layer (potholes etc.) ----
    st.markdown("#### City layer")
    reg, err = safe(layer_registry)
    layers = (reg or {}).get("layers", []) if not err else []
    if err:
        st.caption(f"Layers unavailable: {err}")
    elif layers:
        ids = [l.get("id") for l in layers if l.get("id")]
        labels = {l.get("id"): (l.get("label") or l.get("name") or l.get("id")) for l in layers}
        default = ids.index("potholes") if "potholes" in ids else 0
        pick = st.selectbox("Layer", ids, index=default, format_func=lambda i: labels.get(i, i))
        lv, err = safe(layer_values, pick)
        if err:
            st.caption(f"Layer data unavailable: {err}")
        else:
            values = lv.get("values", {}) or {}
            mine = values.get(f"{corp}:{ward}")
            if mine is None:
                st.caption("No value for this ward in this layer yet.")
            else:
                ranked = sorted(values.values(), reverse=True)
                rank = ranked.index(mine) + 1
                a, b = st.columns(2)
                a.metric(labels.get(pick, pick), f"{mine:,.0f}" if float(mine).is_integer() else f"{mine:,.2f}")
                b.metric("Rank (1 = highest)", f"{rank} of {len(ranked)}")

    # ---- live resident data ----
    st.markdown("#### Live from residents")
    live, err = safe(ward_live, corp, ward)
    if err:
        st.error(err)
    else:
        render_json(live)
        st.caption("Updates about once a minute.")

    # ---- full record, on demand ----
    if st.checkbox("Load full ward record"):
        rec, err = safe(ward_record, corp, ward)
        if err:
            st.error(err)
        else:
            render_json(rec)

    st.markdown(f"[Open the full ward page on kaun.city]({BASE})")
