# app_visualize.py — Hyderabad Land Cover (FULLY CORRECTED)
import streamlit as st
import numpy as np
import tifffile
from pathlib import Path
import io
from PIL import Image
import folium
import folium.plugins
from streamlit_folium import st_folium
from core.config import REGIONS, RegionConfig ,REGION_SPECS
st.set_page_config(page_title="Hyderabad Land Cover 2026-2030", layout="wide")

region: RegionConfig | None = st.session_state.get("region")

if region is None:
    st.warning("⬅Please select a valid region first.")
    st.stop()

# Only allow Hyderabad for this multiclass ConvLSTM page
if region.name.lower() != "hyderabad":
    st.error(f" No multiclass ConvLSTM available for **{region.display_name}**. This page is exclusive to Hyderabad.")
    st.stop()

# Optional: sanity check that the Hyderabad model exists
if not region.has_convlstm:
    st.error(f" ConvLSTM model file missing for **{region.display_name}**.")
    st.stop()



# ════════════════════════════════════════════════════════════════════
DATA_DIR = "C:/Kathmandu/data/hyderabad/HydreabadPredicted"
IMAGE_HEIGHT = 4485   # TIFF shape[0] = rows
IMAGE_WIDTH  = 4310   # TIFF shape[1] = columns
MAX_ROI_SIZE = 5000

# GEE Export Bounds
GEE_LON_MIN, GEE_LAT_MIN = 78.30, 17.20
GEE_LON_MAX, GEE_LAT_MAX = 78.70, 17.60

# Class Mapping
CLASS_NAMES  = ["Urban", "Forest", "Water", "Barren"]
CLASS_COLORS = ["#800000", "#228B22", "#00BFFF", "#D2B48C"]
CLASS_VALUES = [0, 1, 2, 3]
YEARS = [2026, 2027, 2028, 2029, 2030]

HYDERABAD_CENTER = [17.40, 78.50]


# ════════════════════════════════════════════════════════════════════
# HELPER: Convert Mask to RGB
# ════════════════════════════════════════════════════════════════════
def mask_to_rgb(mask, colors, values):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for val, hex_color in zip(values, colors):
        c = hex_color.lstrip('#')
        rgb[mask == val] = [int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)]
    return rgb


# ════════════════════════════════════════════════════════════════════
# HELPER: Convert Lat/Lon to Pixel Coordinates
# ════════════════════════════════════════════════════════════════════
def latlon_to_pixels(lat_min, lon_min, lat_max, lon_max):
    """
    Convert geographic bounds to pixel coordinates.
    Returns: (y_start, x_start, height, width)
    """
    lon_norm_start = (lon_min - GEE_LON_MIN) / (GEE_LON_MAX - GEE_LON_MIN)
    lon_norm_end   = (lon_max - GEE_LON_MIN) / (GEE_LON_MAX - GEE_LON_MIN)

    # Invert latitude (top of image = max lat)
    lat_norm_start = (GEE_LAT_MAX - lat_max) / (GEE_LAT_MAX - GEE_LAT_MIN)
    lat_norm_end   = (GEE_LAT_MAX - lat_min) / (GEE_LAT_MAX - GEE_LAT_MIN)

    x_start = int(lon_norm_start * IMAGE_WIDTH)
    x_end   = int(lon_norm_end   * IMAGE_WIDTH)
    y_start = int(lat_norm_start * IMAGE_HEIGHT)
    y_end   = int(lat_norm_end   * IMAGE_HEIGHT)

    # Ensure valid order
    if x_end < x_start: x_start, x_end = x_end, x_start
    if y_end < y_start: y_start, y_end = y_end, y_start

    # Clamp to image bounds
    x_start = max(0, min(x_start, IMAGE_WIDTH))
    x_end   = max(0, min(x_end,   IMAGE_WIDTH))
    y_start = max(0, min(y_start, IMAGE_HEIGHT))
    y_end   = max(0, min(y_end,   IMAGE_HEIGHT))

    return y_start, x_start, y_end - y_start, x_end - x_start


# ════════════════════════════════════════════════════════════════════
# HELPER: Cached TIFF loader
# ════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_tiff_crop(fpath_str, y_start, x_start, h, w):
    """Load and crop a single TIFF, with caching to avoid re-reading disk."""
    arr = tifffile.imread(fpath_str)
    y_end = min(y_start + h, arr.shape[0])
    x_end = min(x_start + w, arr.shape[1])
    cropped = arr[y_start:y_end, x_start:x_end]
    # Pad if needed (edge-fill)
    pad_h = h - cropped.shape[0]
    pad_w = w - cropped.shape[1]
    if pad_h > 0 or pad_w > 0:
        cropped = np.pad(cropped, ((0, pad_h), (0, pad_w)), mode='edge')
    return cropped


# ════════════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════════════
for key, default in [
    ("roi_ready", False),
    ("roi_data",  None),
    ("results",   None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ════════════════════════════════════════════════════════════════════
# PAGE TITLE
# ════════════════════════════════════════════════════════════════════
st.title(" Hyderabad Land Cover Prediction (2026–2030)")
st.caption(f"Image: {IMAGE_WIDTH}×{IMAGE_HEIGHT} pixels | Max ROI: {MAX_ROI_SIZE}×{MAX_ROI_SIZE}")


# ════════════════════════════════════════════════════════════════════
# FOLIUM MAP
# ════════════════════════════════════════════════════════════════════


m = folium.Map(
    location=HYDERABAD_CENTER,
    zoom_start=11,
    tiles="Esri.WorldImagery",
    min_lat=GEE_LAT_MIN, max_lat=GEE_LAT_MAX,
    min_lon=GEE_LON_MIN, max_lon=GEE_LON_MAX,
    max_bounds=True
)

folium.Rectangle(
    bounds=[[GEE_LAT_MIN, GEE_LON_MIN], [GEE_LAT_MAX, GEE_LON_MAX]],
    color="yellow", weight=3, fill=False, popup="Study Area"
).add_to(m)

# FIX: must explicitly import folium.plugins
folium.plugins.Draw(
    draw_options={
        "rectangle":    True,
        "polygon":      False,
        "circle":       False,
        "marker":       False,
        "circlemarker": False,
        "polyline":     False,
    }
).add_to(m)

map_data = st_folium(m, width=1200, height=600, key="hyderabad_map")


# ════════════════════════════════════════════════════════════════════
# PROCESS MAP SELECTION
# ════════════════════════════════════════════════════════════════════
st.session_state.roi_ready = False

# Main ROI detection
if map_data and map_data.get("last_active_drawing"):
    drawing  = map_data["last_active_drawing"]
    geo_type = drawing["geometry"]["type"]

    # FIX: Folium Draw plugin emits rectangles as "Polygon" geometry
    if geo_type == "Polygon":
        coords_raw = drawing["geometry"]["coordinates"][0]
        lons = [p[0] for p in coords_raw]
        lats = [p[1] for p in coords_raw]

        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)

        # Validate drawn area is non-degenerate
        if min_lat >= max_lat or min_lon >= max_lon:
            st.error(" Degenerate selection — please draw a proper rectangle.")
        else:
            y_start, x_start, h, w = latlon_to_pixels(min_lat, min_lon, max_lat, max_lon)

            # Enforce max ROI size
            if h > MAX_ROI_SIZE or w > MAX_ROI_SIZE:
                h = min(h, MAX_ROI_SIZE)
                w = min(w, MAX_ROI_SIZE)
                st.warning(f"Selection auto-cropped to {MAX_ROI_SIZE}×{MAX_ROI_SIZE} pixels.")

            if h == 0 or w == 0:
                st.error(" ROI has zero size — try drawing a larger rectangle.")
            elif y_start + h <= IMAGE_HEIGHT and x_start + w <= IMAGE_WIDTH:
                st.session_state.roi_ready = True
                st.session_state.roi_data  = {
                    "y_start": y_start,
                    "x_start": x_start,
                    "h": h,
                    "w": w
                }
                st.success(
                    f" ROI: {w}×{h}px at (x={x_start}, y={y_start}) — "
                    f"Click Predict below for prediction"
                )
            else:
                st.error(
                    f" ROI out of image bounds: "
                    f"y_end={y_start+h} (max {IMAGE_HEIGHT}), "
                    f"x_end={x_start+w} (max {IMAGE_WIDTH})"
                )
    else:
        st.info(f" Unrecognized geometry type: `{geo_type}`. Please draw a rectangle.")

st.divider()
col_load, col_reset = st.columns([4, 1])

with col_load:
    if not st.session_state.roi_ready:
        st.info(" Draw a rectangle inside the yellow boundary first")

    load_clicked = st.button(
        " Predict 2026-2030",
        type="primary",
        disabled=not st.session_state.roi_ready,
        key="load_btn",
        use_container_width=True  # Makes button fill the column
    )

with col_reset:
    # Add some top margin to align with Load button
    st.write("")
    st.write("")
    reset_clicked = st.button(
        " Reset",
        key="reset_btn",
        use_container_width=True
    )
    if reset_clicked:
        st.session_state.roi_ready = False
        st.session_state.roi_data = None
        st.session_state.results = None
        st.rerun()

if load_clicked and st.session_state.roi_ready and st.session_state.roi_data:
    roi = st.session_state.roi_data
    masks      = {}
    load_errors = []

    with st.spinner(f"Predicting {len(YEARS)} year(s) for {roi['w']}×{roi['h']} ROI…"):
        for year in YEARS:
            fpath = Path(DATA_DIR) / f"mask_{year}.tiff"
            if fpath.exists():
                try:
                    cropped = load_tiff_crop(
                        str(fpath),
                        roi["y_start"], roi["x_start"],
                        roi["h"],       roi["w"]
                    )
                    masks[year] = cropped
                except Exception as e:
                    load_errors.append(f"{year}: {e}")
            else:
                load_errors.append(f"{year}: File not found → {fpath}")

    if load_errors:
        for err in load_errors:
            st.warning(f" {err}")

    if masks:
        st.session_state.results = {
            "masks": masks,
            "rgbs":  {yr: mask_to_rgb(msk, CLASS_COLORS, CLASS_VALUES)
                      for yr, msk in masks.items()},
            "coords": roi,
        }
        if len(masks) == len(YEARS):
            st.success("All years loaded!")
        else:
            st.warning(f" Loaded {len(masks)}/{len(YEARS)} years.")
        st.rerun()
    else:
        st.error(" No TIFF files could be loaded. Check DATA_DIR and file names.")

if st.session_state.results:
    res = st.session_state.results

    legend_cols = st.columns(len(CLASS_NAMES))
    for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        with legend_cols[i]:
            st.markdown(
                f"<div style='background:{color};padding:12px;border-radius:5px;"
                f"text-align:center;color:white;font-weight:bold'>{name}</div>",
                unsafe_allow_html=True
            )
    st.divider()
    st.subheader(" Land Cover Maps")

    # Get all available years
    available_years = [y for y in YEARS if y in res["rgbs"]]

    per_row = 2  # max columns per row

    for i in range(0, len(available_years), per_row):
        row_years = available_years[i:i + per_row]
        cols = st.columns(per_row)  # always create `per_row` columns
        for idx, year in enumerate(row_years):
            with cols[idx]:
                st.image(
                    res["rgbs"][year],
                    caption=f"Predicted {year}",
                    use_container_width=True
                )
        # remaining columns (if any) stay empty, keeping layout consistent
    # ── Legend ──────────────────────────────────────────────────────


    # ════════════════════════════════════════════════════════════════════
    # URBAN GROWTH VISUALIZATION
    # ════════════════════════════════════════════════════════════════════
    st.markdown("### 📈 Urban Growth Analysis")

    # Urban class ID (assuming 0 = Urban based on your CLASS_VALUES)
    URBAN_CLASS = 0

    # Calculate urban area for each year
    urban_areas = {}
    for year, mask in st.session_state.results["masks"].items():
        urban_pixels = np.sum(mask == URBAN_CLASS)
        urban_km2 = urban_pixels * 100 / 1_000_000  # 10m × 10m = 100 m² per pixel
        urban_areas[year] = {"pixels": urban_pixels, "km2": urban_km2}

    # Display urban area per year
    st.markdown("**Urban Area by Year:**")
    area_cols = st.columns(len(YEARS))
    for idx, year in enumerate(YEARS):
        with area_cols[idx]:
            st.metric(f"{year}", f"{urban_areas[year]['km2']:.2f} km²",
                      delta=f"{urban_areas[year]['km2'] - urban_areas[YEARS[0]]['km2']:.2f} km²" if idx > 0 else None)

    # Growth chart
    st.markdown("**Urban Growth Trend:**")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(list(urban_areas.keys()), [v['km2'] for v in urban_areas.values()],
            marker='o', linewidth=2, markersize=8, color='#FF6B6B')
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Urban Area (km²)", fontsize=12)
    ax.set_title("Urban Growth (2026–2030)", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()

    # Change detection map (First year vs Last year)
    st.markdown("**Urban Expansion Map (2026 → 2030):**")

    first_year = YEARS[0]
    last_year = YEARS[-1]
    mask_first = st.session_state.results["masks"][first_year]
    mask_last = st.session_state.results["masks"][last_year]

    # Detect changes: 0=no change, 1=urban gain, 2=urban loss
    change_map = np.zeros_like(mask_first, dtype=np.uint8)
    change_map[(mask_first != URBAN_CLASS) & (mask_last == URBAN_CLASS)] = 1  # New urban
    change_map[(mask_first == URBAN_CLASS) & (mask_last != URBAN_CLASS)] = 2  # Lost urban

    # Color the change map
    change_colors = {0: [128, 128, 128], 1: [255, 0, 0], 2: [0, 128, 0]}  # Gray, Red, Green
    change_rgb = np.zeros((mask_first.shape[0], mask_first.shape[1], 3), dtype=np.uint8)
    for val, color in change_colors.items():
        change_rgb[change_map == val] = color

    # Display change map
    col_c1, col_c2 = st.columns([2, 1])
    with col_c1:
        st.image(change_rgb,
                 caption=f"Urban Change: {first_year} → {last_year}\n🔴 Red = New Urban | 🟢 Green = Lost Urban | ⚫ Gray = No Change",
                 width="stretch")

    with col_c2:
        # Statistics
        new_urban = np.sum(change_map == 1)
        lost_urban = np.sum(change_map == 2)
        no_change = np.sum(change_map == 0)
        total = change_map.size

        st.metric(" New Urban Area", f"{new_urban * 100 / 1_000_000:.2f} km²",
                  f"{new_urban / total * 100:.2f}% of ROI")
        st.metric(" Lost Urban Area", f"{lost_urban * 100 / 1_000_000:.2f} km²",
                  f"{lost_urban / total * 100:.2f}% of ROI")
        st.metric(" No Change", f"{no_change / total * 100:.1f}%")

        # Net growth
        net_growth = (new_urban - lost_urban) * 100 / 1_000_000
        st.metric(" Net Urban Growth", f"{net_growth:+.2f} km²",
                  delta=f"{(urban_areas[last_year]['km2'] - urban_areas[first_year]['km2']):.2f} km²")

    # Download change map
    buf = io.BytesIO()
    Image.fromarray(change_rgb).save(buf, format="PNG")
    buf.seek(0)
    st.download_button(
        label=" Download Change Map (.png)",
        data=buf,
        file_name=f"urban_change_{first_year}_to_{last_year}.png",
        mime="image/png"
    )

    # Year-to-year growth table
    st.markdown("**Year-to-Year Urban Growth:**")
    growth_data = []
    for i in range(1, len(YEARS)):
        prev_year = YEARS[i - 1]
        curr_year = YEARS[i]
        growth = urban_areas[curr_year]['km2'] - urban_areas[prev_year]['km2']
        growth_pct = (growth / urban_areas[prev_year]['km2'] * 100) if urban_areas[prev_year]['km2'] > 0 else 0
        growth_data.append([f"{prev_year}→{curr_year}", growth, growth_pct])

    import pandas as pd

    growth_df = pd.DataFrame(growth_data, columns=["Period", "Growth (km²)", "Growth (%)"])
    st.dataframe(growth_df.style.format({"Growth (km²)": "{:+.2f}", "Growth (%)": "{:+.2f}%"}),
                 use_container_width=True)
    # ── Statistics ──────────────────────────────────────────────────
    st.markdown("###  Area Statistics (km²)")
    available_years = [yr for yr in YEARS if yr in res["masks"]]
    stats_cols = st.columns(len(available_years))

    for idx, year in enumerate(available_years):
        with stats_cols[idx]:
            st.write(f"**{year}**")
            mask  = res["masks"][year]
            total = mask.size
            for name, val in zip(CLASS_NAMES, CLASS_VALUES):
                cnt  = int(np.sum(mask == val))
                # Each pixel = 30m × 30m = 900 m² = 0.0009 km²
                area = cnt * 900 / 1_000_000
                pct  = cnt / total * 100
                st.metric(label=name[:3], value=f"{area:.2f} km²", delta=f"{pct:.1f}%")

    # ── Download ────────────────────────────────────────────────────
    st.markdown("###  Download PNG")
    dl_cols = st.columns(len(available_years))

    for idx, year in enumerate(available_years):
        with dl_cols[idx]:
            rgb_arr = res["rgbs"][year]
            pil_img = Image.fromarray(rgb_arr)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            buf.seek(0)
            st.download_button(
                label=f" {year}",
                data=buf,
                file_name=f"landcover_{year}.png",
                mime="image/png",
                key=f"dl_{year}"
            )

    # ── Reset Button ────────────────────────────────────────────────

# ═══════════════════

