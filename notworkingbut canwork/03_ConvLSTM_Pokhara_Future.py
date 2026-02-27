# pages/03_ConvLSTM_Pokhara_Future.py
# ConvLSTM Future Urban Prediction for Pokhara

import streamlit as st
import folium
from streamlit_folium import st_folium
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
import numpy as np
import torch
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from models_pokhara import UrbanSprawlConvLSTM_Pokhara
from PIL import Image

# ─────────────────────────────────────────────
# PATH CONFIGURATION (POKHARA)
# ─────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "pokhara")
CONVLSTM_PATH = os.path.join(DATA_DIR, "pokharaconvlstm.pth")

PAST_PRED_DIR = os.path.join(DATA_DIR, "past_prediction")
os.makedirs(PAST_PRED_DIR, exist_ok=True)

SEQ_LEN = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIXEL_AREA_M2 = 100  # 10m x 10m

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="ConvLSTM Pokhara Future", page_icon="🏔️", layout="wide")
st.title("🏔️ Pokhara - ConvLSTM Future Urban Prediction")
st.info("📍 Region: **POKHARA** | Model: pokharaconvlstm.pth")


# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_resource
def load_convlstm():
    if not os.path.exists(CONVLSTM_PATH):
        st.error(f"ConvLSTM model not found:\n{CONVLSTM_PATH}")
        st.stop()

    model = UrbanSprawlConvLSTM_Pokhara(input_channels=3, hidden_channels=64)
    checkpoint = torch.load(CONVLSTM_PATH, map_location="cpu")

    try:
        model.load_state_dict(checkpoint, strict=True)
        st.success("✅ Pokhara ConvLSTM model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    model.eval()
    return model.to(DEVICE)


convlstm_model = load_convlstm()

# Debug info in sidebar
with st.sidebar:
    st.markdown("### 🔍 System Info")
    st.write(f"**Region:** Pokhara")
    st.write(f"**Device:** {DEVICE}")
    st.write(f"**Model:** pokharaconvlstm.pth")
    total_params = sum(p.numel() for p in convlstm_model.parameters())
    st.write(f"**Parameters:** {total_params:,}")

    if os.path.exists(DATA_DIR):
        files = [f for f in os.listdir(DATA_DIR) if f.endswith(('.tif', '.TIF', '.pth'))]
        st.write(f"**Files:** {len(files)}")
        with st.expander("Show files"):
            for f in sorted(files):
                st.text(f"• {f}")

# ─────────────────────────────────────────────
# INTERACTIVE MAP
# ─────────────────────────────────────────────
st.markdown("### 📍 Select Area for Prediction")

SW = [28.10, 83.85]
NE = [28.35, 84.20]

m = folium.Map(
    location=[28.21, 83.99],
    zoom_start=12,
    min_zoom=11,
    max_zoom=16,
    max_bounds=True,
    min_lat=SW[0],
    max_lat=NE[0],
    min_lon=SW[1],
    max_lon=NE[1]
)

folium.TileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
    name="Satellite"
).add_to(m)

folium.plugins.Draw(
    export=False,
    draw_options={
        'rectangle': True,
        'polygon': False,
        'circle': False,
        'marker': False,
        'circlemarker': False,
        'polyline': False
    }
).add_to(m)

map_data = st_folium(m, width=1000, height=600, key="pokhara_convlstm_map")

# ─────────────────────────────────────────────
# WHEN USER DRAWS AREA
# ─────────────────────────────────────────────
if map_data and map_data.get("last_active_drawing"):

    coords = map_data["last_active_drawing"]["geometry"]["coordinates"][0]
    lons = [p[0] for p in coords]
    lats = [p[1] for p in coords]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)

    with st.spinner("Loading historical data..."):
        sequence = []
        years_in = []
        reference_shape = None

        # Try to load historical data
        for y in range(2015, 2026):
            mask_path = os.path.join(DATA_DIR, f"mask_{y}.tif")
            img_path = os.path.join(DATA_DIR, f"{y}IMAGE.TIF")

            if os.path.exists(mask_path):
                with rasterio.open(mask_path) as src:
                    wb = transform_bounds("EPSG:4326", src.crs, min_lon, min_lat, max_lon, max_lat)
                    window = from_bounds(*wb, src.transform)
                    mask = src.read(1, window=window).astype(np.float32)

                    if mask.size == 0:
                        continue

                    mask = (mask > 127).astype(np.float32)
                    mask = np.clip(mask, 0.0, 1.0)

                    if reference_shape is None:
                        reference_shape = mask.shape
                    elif mask.shape != reference_shape:
                        mask = cv2.resize(mask, (reference_shape[1], reference_shape[0]), cv2.INTER_NEAREST)

                    # Convert to 3-channel
                    mask = np.stack([mask] * 3, axis=-1)
                    sequence.append(mask)
                    years_in.append(y)

            elif os.path.exists(img_path):
                with rasterio.open(img_path) as src:
                    wb = transform_bounds("EPSG:4326", src.crs, min_lon, min_lat, max_lon, max_lat)
                    window = from_bounds(*wb, src.transform)
                    img = src.read([1, 2, 3], window=window).astype(np.float32) / 255.0
                    img = np.transpose(img, (1, 2, 0))

                    if img.size == 0:
                        continue

                    if reference_shape is None:
                        reference_shape = img.shape[:2]
                    elif img.shape[:2] != reference_shape:
                        img = cv2.resize(img, (reference_shape[1], reference_shape[0]), cv2.INTER_NEAREST)

                    sequence.append(img)
                    years_in.append(y)

        if len(sequence) == 0:
            st.error(f"No data files found in:\n{DATA_DIR}")
            st.stop()

        if len(sequence) < SEQ_LEN:
            st.warning(f"Only {len(sequence)} time steps found. Need at least {SEQ_LEN}.")
            if len(sequence) < 2:
                st.error("Need at least 2 time steps for prediction.")
                st.stop()

        st.success(f"✅ Loaded {len(sequence)} time steps: {years_in}")

    # ─────────────────────────────────────────
    # USER CONTROLS
    # ─────────────────────────────────────────
    st.markdown("### ⚙️ Prediction Settings")
    st.write(f"**Input years:** {years_in[0]} – {years_in[-1]}")

    col1, col2 = st.columns(2)
    with col1:
        max_future_year = st.slider("Predict up to year", 2026, 2040, 2030)
    with col2:
        binary_thresh = st.slider("Urban threshold", 0.1, 0.9, 0.5, 0.05)

    if "predictions_pokhara" not in st.session_state:
        st.session_state.predictions_pokhara = None
        st.session_state.years_list_pokhara = None

    if st.button("🚀 Run Prediction", type="primary"):

        with st.spinner(f"Running ConvLSTM prediction → {max_future_year}..."):

            current_seq = sequence[-SEQ_LEN:]
            predictions = {}
            current_year = years_in[-1]

            while current_year < max_future_year:
                current_year += 1
                seq_np = np.stack(current_seq, axis=0)  # [T,H,W,C]
                seq_np = np.transpose(seq_np, (0, 3, 1, 2))  # [T,C,H,W]
                seq_t = torch.from_numpy(seq_np).float().unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    output = convlstm_model(seq_t)  # [B, 3, H, W]
                    # Take mean across 3 channels to get single probability map
                    prob = output[0].mean(dim=0).cpu().numpy()

                predictions[current_year] = prob
                pred_bin = (prob > binary_thresh).astype(np.float32)
                current_seq = current_seq[1:] + [np.stack([pred_bin] * 3, axis=-1)]

            st.session_state.predictions_pokhara = predictions
            st.session_state.years_list_pokhara = list(predictions.keys())

    # ─────────────────────────────────────────
    # VISUALIZATION
    # ─────────────────────────────────────────
    if st.session_state.predictions_pokhara is not None:
        predictions = st.session_state.predictions_pokhara
        years_list = st.session_state.years_list_pokhara

        st.markdown("---")
        st.markdown("## 📊 Predicted Urban Probability Maps")
        cmap_prob = LinearSegmentedColormap.from_list("urban_risk", ['white', 'yellow', 'orange', 'darkred'])

        vis_thresh = st.slider("Highlight probability ≥", 0.0, 1.0, 0.5, 0.05, key="vis_thresh_pokhara")

        cols = st.columns(min(3, len(years_list)))
        for i, yr in enumerate(years_list):
            prob = predictions[yr]
            with cols[i % len(cols)]:
                fig, ax = plt.subplots(figsize=(7, 7))
                im = ax.imshow(prob, cmap=cmap_prob, vmin=0, vmax=1)
                ax.set_title(f"Year {yr}", fontsize=14, fontweight='bold')
                ax.axis("off")
                fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04)
                st.pyplot(fig)
                plt.close()

                bin_map = (prob > binary_thresh).astype(np.uint8) * 255
                bin_large = cv2.resize(bin_map, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
                st.image(bin_large, caption=f"Binary {yr} (white = urban)", clamp=True)

                area_km2 = bin_map.sum() / 255 * (PIXEL_AREA_M2 / 1_000_000)
                st.metric(f"Urban area {yr}", f"{area_km2:.2f} km²")

        # ─────────────────────────────────────
        # YEAR-TO-YEAR CHANGE
        # ─────────────────────────────────────
        if len(years_list) > 1:
            st.markdown("---")
            st.markdown("## 📈 Year-to-Year Probability Change")
            for j in range(1, len(years_list)):
                prev, curr = years_list[j - 1], years_list[j]
                diff = predictions[curr] - predictions[prev]
                fig, ax = plt.subplots(figsize=(7, 7))
                im = ax.imshow(diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
                ax.set_title(f"Change {prev} → {curr}\n(Red = increase, Blue = decrease)")
                ax.axis("off")
                fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046)
                st.pyplot(fig)
                plt.close()

        # ─────────────────────────────────────
        # EXPANSION MAPS
        # ─────────────────────────────────────
        st.markdown("---")
        st.markdown("## 🆕 New Urban Expansion")
        st.caption("Pixel = 100 m² → small plot size")

        last_known_bin = (sequence[-1][..., 0] > binary_thresh).astype(np.uint8)
        cols_exp = st.columns(min(3, len(years_list)))

        for i, yr in enumerate(years_list):
            prob = predictions[yr]
            pred_bin = (prob > binary_thresh).astype(np.uint8)
            expansion = np.clip(pred_bin - last_known_bin, 0, 1) * 255

            with cols_exp[i % len(cols_exp)]:
                fig, ax = plt.subplots(figsize=(7, 7))
                ax.imshow(expansion, cmap="Reds", vmin=0, vmax=255)
                ax.set_title(f"New Urban {yr}", fontsize=14, fontweight='bold')
                ax.axis("off")
                st.pyplot(fig)
                plt.close()

                exp_large = cv2.resize(expansion, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
                st.image(exp_large, caption=f"New Built-up {yr} (red)", clamp=True)

                exp_km2 = expansion.sum() / 255 * (PIXEL_AREA_M2 / 1_000_000)
                st.metric(f"New built-up {yr}", f"{exp_km2:.2f} km²")

            last_known_bin = pred_bin

        # ─────────────────────────────────────
        # GROWTH CHARTS
        # ─────────────────────────────────────
        st.markdown("---")
        st.markdown("## 📈 Growth Trend Analysis")

        urban_areas = [(prob > binary_thresh).sum() * PIXEL_AREA_M2 / 1_000_000 for prob in predictions.values()]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(years_list, urban_areas, marker='o', linewidth=2, markersize=8, color='#e74c3c')
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Urban Area (km²)", fontsize=12)
        ax.set_title("Total Urban Area Growth Projection - Pokhara", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.fill_between(years_list, urban_areas, alpha=0.3, color='#e74c3c')
        st.pyplot(fig)
        plt.close()

        # New expansion per year
        new_areas = []
        prev_bin = (sequence[-1][..., 0] > binary_thresh).astype(np.float32)
        for yr in years_list:
            curr_bin = (predictions[yr] > binary_thresh).astype(np.float32)
            new_km2 = ((curr_bin - prev_bin) > 0).sum() * (PIXEL_AREA_M2 / 1_000_000)
            new_areas.append(new_km2)
            prev_bin = curr_bin

        fig_bar, ax = plt.subplots(figsize=(10, 5))
        ax.bar(years_list, new_areas, color='#ff7f0e', alpha=0.8)
        ax.set_title("New Expansion per Year - Pokhara", fontsize=14, fontweight='bold')
        ax.set_ylabel("km²", fontsize=12)
        ax.set_xlabel("Year", fontsize=12)
        ax.grid(True, axis='y', alpha=0.4)
        st.pyplot(fig_bar)
        plt.close()

        if len(urban_areas) > 1:
            total_growth = urban_areas[-1] - urban_areas[0]
            years_span = years_list[-1] - years_list[0]
            annual_growth = total_growth / years_span if years_span > 0 else 0

            col1, col2, col3 = st.columns(3)
            col1.metric("Initial Area", f"{urban_areas[0]:.2f} km²")
            col2.metric("Projected Area", f"{urban_areas[-1]:.2f} km²")
            col3.metric("Annual Growth", f"{annual_growth:.2f} km²/yr")

        # ─────────────────────────────────────
        # EXPORT BUTTON
        # ─────────────────────────────────────
        st.markdown("---")
        if st.button("💾 Export Predictions as PNG"):
            run_num = len(os.listdir(PAST_PRED_DIR)) + 1 if os.path.exists(PAST_PRED_DIR) else 1
            folder = os.path.join(PAST_PRED_DIR, f"export_{run_num}")
            os.makedirs(folder, exist_ok=True)
            saved = 0

            for yr, prob in predictions.items():
                bin_map = (prob > binary_thresh).astype(np.uint8) * 255
                bin_large = cv2.resize(bin_map, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
                Image.fromarray(bin_large).save(os.path.join(folder, f"binary_{yr}.png"))
                saved += 1

                # Expansion
                if yr == years_list[0]:
                    prev_bin = (sequence[-1][..., 0] > binary_thresh).astype(np.uint8)
                else:
                    prev_bin = (predictions[years_list[years_list.index(yr) - 1]] > binary_thresh).astype(np.uint8)
                exp = np.clip((prob > binary_thresh).astype(np.uint8) - prev_bin, 0, 1) * 255
                exp_large = cv2.resize(exp, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
                Image.fromarray(exp_large).save(os.path.join(folder, f"expansion_{yr}.png"))
                saved += 1

            st.success(f"✅ Saved {saved} PNG files to:\n`{folder}`")

    else:
        st.info("Click '🚀 Run Prediction' to generate results.")

else:
    st.info("👆 Draw a rectangle on the Pokhara map above to start prediction.")