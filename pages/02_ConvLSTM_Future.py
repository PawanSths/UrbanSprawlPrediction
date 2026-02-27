# pages/03_ConvLSTM_Future.py
# Multi-region ConvLSTM future prediction with comprehensive analysis
# Supports: Kathmandu (1-channel binary), Pokhara (3-channel multi-class)

import streamlit as st
import folium
from streamlit_folium import st_folium
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
import numpy as np

import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from PIL import Image
# Top of the file — first lines

import os
import ctypes

torch_dll_path = r"D:\Kathmandu\Kathmandu\.venv\Lib\site-packages\torch\lib\c10.dll"

if os.path.exists(torch_dll_path):
    try:
        ctypes.CDLL(torch_dll_path)
        print("Successfully pre-loaded c10.dll")
    except Exception as e:
        print(f"Pre-load failed: {e}")
else:
    print("c10.dll not found at expected path!")

# ────────────────────────────────────────────────
# Now safe to do all other imports
# ────────────────────────────────────────────────
import sys
import torch
# ─────────────────────────────────────────────
# GET SELECTED REGION
# ─────────────────────────────────────────────
if "region_key" not in st.session_state:
    st.error(" No region selected. Please go to Home and select a region first.")
    st.stop()

region_key = st.session_state["region_key"]

# ─────────────────────────────────────────────
# REGION CONFIGURATIONS
# ─────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

REGION_CONFIGS = {
    "kathmandu": {
        "display_name": "Kathmandu",
        "data_dir": os.path.join(PROJECT_ROOT, "data", "kathmandu"),
        "model_path": "convlstm.pth",
        "map_center": [27.70, 85.32],
        "bounds": {"SW": [27.55, 85.15], "NE": [27.85, 85.55]},
        "icon": "🏔️",
        "model_type": "kathmandu",
        "input_channels": 1,
        "hidden_channels": 32,
        "num_layers": 2,
        "output_channels": 1,
        "year_range": (2019, 2026),
        "mode": "binary",
    },
    "pokhara": {
        "display_name": "Pokhara",
        "data_dir": os.path.join(PROJECT_ROOT, "data", "pokhara"),
        "model_path": "pokharaconvlstm.pth",
        "map_center": [28.21, 83.99],
        "bounds": {"SW": [28.10, 83.85], "NE": [28.35, 84.20]},
        "icon": "⛰️",
        "model_type": "pokhara",
        "input_channels": 3,
        "hidden_channels": 64,
        "output_channels": 3,
        "year_range": (2019, 2026),
        "mode": "multiclass",
        "class_mapping": {0: "background", 1: "water", 2: "vegetation", 3: "urban"},
    },
}

if region_key not in REGION_CONFIGS:
    st.error(f"❌ Region '{region_key}' not configured.")
    st.stop()

config = REGION_CONFIGS[region_key]
DATA_DIR = config["data_dir"]
CONVLSTM_PATH = os.path.join(DATA_DIR, config["model_path"])

# Export directory
PAST_PRED_DIR = os.path.join(DATA_DIR, "past_prediction")
os.makedirs(PAST_PRED_DIR, exist_ok=True)

SEQ_LEN = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIXEL_AREA_M2 = 100

# ─��───────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title=f"ConvLSTM Future - {config['display_name']}",
    page_icon=config['icon'],
    layout="wide"
)
st.title(f"{config['icon']} {config['display_name']} - ConvLSTM Future Urban Prediction")
st.info(f"Region: **{config['display_name'].upper()}** | Model: {config['model_path']}")


# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_resource
def load_convlstm(region_key):
    config = REGION_CONFIGS[region_key]
    model_path = os.path.join(config["data_dir"], config["model_path"])

    if not os.path.exists(model_path):
        st.error(f"Model not found: {model_path}")
        st.stop()

    if config["model_type"] == "pokhara":
        from models_pokhara import UrbanSprawlConvLSTM_Pokhara
        model = UrbanSprawlConvLSTM_Pokhara(
            input_channels=config["input_channels"],
            hidden_channels=config["hidden_channels"]
        )
    else:
        from models import UrbanSprawlConvLSTM
        model = UrbanSprawlConvLSTM(
            config["input_channels"],
            config["hidden_channels"],
            config["num_layers"],
            config["output_channels"]
        )

    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    return model.to(DEVICE)


convlstm_model = load_convlstm(region_key)

# Sidebar
with st.sidebar:
    st.markdown("###  System Info")
    st.write(f"**Region:** {config['display_name']}")
    st.write(f"**Mode:** {config['mode'].title()}")
    st.write(f"**Device:** {DEVICE}")
    total_params = sum(p.numel() for p in convlstm_model.parameters())
    st.write(f"**Parameters:** {total_params:,}")

    if config["mode"] == "multiclass":
        st.markdown("### 🎨 Classes")
        st.write("🔵 Water")
        st.write("🟢 Vegetation")
        st.write("🔴 Urban")

# ─────────────────────────────────────────────
# MAP
# ──────────────────────────────���──────────────
st.markdown("###  Select Area for Prediction")

SW = config["bounds"]["SW"]
NE = config["bounds"]["NE"]

m = folium.Map(
    location=config["map_center"],
    zoom_start=12,
    min_zoom=11, max_zoom=16,
    max_bounds=True,
    min_lat=SW[0], max_lat=NE[0],
    min_lon=SW[1], max_lon=NE[1]
)

folium.TileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri"
).add_to(m)

folium.plugins.Draw(draw_options={"rectangle": True}).add_to(m)

map_data = st_folium(m, width=1000, height=600, key=f"convlstm_map_{region_key}")

# ─────────────────────────────────────────────
# DATA LOADING
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

        start_year, end_year = config["year_range"]

        for y in range(start_year, end_year):
            mask_path = os.path.join(DATA_DIR, f"mask_{y}.tif")

            if not os.path.exists(mask_path):
                continue

            try:
                with rasterio.open(mask_path) as src:
                    wb = transform_bounds("EPSG:4326", src.crs, min_lon, min_lat, max_lon, max_lat)
                    window = from_bounds(*wb, src.transform)
                    mask = src.read(1, window=window).astype(np.float32)

                    if mask.size == 0 or mask.shape[0] < 1 or mask.shape[1] < 1:
                        continue

                    if reference_shape is None:
                        reference_shape = mask.shape
                    elif mask.shape != reference_shape:
                        mask = cv2.resize(mask, (reference_shape[1], reference_shape[0]), cv2.INTER_NEAREST)

                    if config["mode"] == "binary":
                        if mask.max() > 1.5:
                            mask = (mask > 127).astype(np.float32)
                        mask = np.clip(mask, 0.0, 1.0)
                        sequence.append(mask)
                    else:  # multiclass
                        water = (mask == 1).astype(np.float32)
                        vegetation = (mask == 2).astype(np.float32)
                        urban = (mask == 3).astype(np.float32)
                        mask_3ch = np.stack([water, vegetation, urban], axis=-1)
                        sequence.append(mask_3ch)

                    years_in.append(y)

            except Exception as e:
                st.error(f"Error loading mask_{y}.tif: {e}")

        if len(sequence) < SEQ_LEN:
            st.warning(f"Only {len(sequence)} years found. Using available sequence.")
        if len(sequence) < 2:
            st.error("Need at least 2 years of data.")
            st.stop()

        st.success(f"✅ Loaded {len(sequence)} years: {years_in}")

    # ─────────────────────────────────────────
    # USER CONTROLS
    # ─────────────────────────────────────────
    st.markdown("###  Prediction Settings")
    st.write(f"**Input years:** {years_in[0]} – {years_in[-1]}")

    col1, col2 = st.columns(2)
    with col1:
        max_future_year = st.slider("Predict up to year", 2026, 2030, 2030)
    with col2:
        binary_thresh = st.slider("Binary threshold", 0.1, 0.9, 0.5, 0.05)

    session_key = f"predictions_{region_key}"
    years_key = f"years_{region_key}"

    if session_key not in st.session_state:
        st.session_state[session_key] = None
        st.session_state[years_key] = None

    if st.button(" Run Future Prediction", type="primary"):
        with st.spinner(f"Predicting → {max_future_year}..."):
            current_seq = sequence[-SEQ_LEN:] if len(sequence) >= SEQ_LEN else sequence
            predictions = {}
            current_year = years_in[-1]

            while current_year < max_future_year:
                current_year += 1
                seq_np = np.stack(current_seq, axis=0)

                if config["mode"] == "binary":
                    seq_t = torch.from_numpy(seq_np).float().unsqueeze(0).unsqueeze(2).to(DEVICE)
                else:
                    seq_np = np.transpose(seq_np, (0, 3, 1, 2))
                    seq_t = torch.from_numpy(seq_np).float().unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    output = convlstm_model(seq_t)

                    if config["mode"] == "binary":
                        prob = output[0, 0].cpu().numpy()
                        pred_bin = (prob > binary_thresh).astype(np.float32)
                        current_seq = current_seq[1:] + [pred_bin]
                    else:
                        prob_3ch = output[0].cpu().numpy()
                        pred_3ch = np.zeros((prob_3ch.shape[1], prob_3ch.shape[2], 3), dtype=np.float32)
                        pred_3ch[:, :, 0] = (prob_3ch[0] > binary_thresh).astype(np.float32)
                        pred_3ch[:, :, 1] = (prob_3ch[1] > binary_thresh).astype(np.float32)
                        pred_3ch[:, :, 2] = (prob_3ch[2] > binary_thresh).astype(np.float32)
                        current_seq = current_seq[1:] + [pred_3ch]
                        prob = prob_3ch

                predictions[current_year] = prob

            st.session_state[session_key] = predictions
            st.session_state[years_key] = list(predictions.keys())
            st.success(f"✅ Predicted {len(predictions)} years!")

    # ═══════════════════════════════════════════════════════
    # VISUALIZATION
    # ═══════════════════════════════════════════════════════
    if st.session_state[session_key] is not None:
        predictions = st.session_state[session_key]
        years_list = st.session_state[years_key]

        if config["mode"] == "binary":
            # ═══════════════════════════════════════════════════════
            # KATHMANDU VISUALIZATION
            # ═══════════════════════════════════════════════════════
            st.markdown("---")
            st.markdown("##  Predicted Urban Probability Maps")

            cmap_prob = LinearSegmentedColormap.from_list("urban", ['white', 'yellow', 'orange', 'darkred'])
            vis_thresh = st.slider("Highlight probability ≥", 0.0, 1.0, 0.5, 0.05, key="vis_thresh")

            cols = st.columns(min(3, len(years_list)))
            for i, yr in enumerate(years_list):
                prob = predictions[yr]
                with cols[i % len(cols)]:
                    fig, ax = plt.subplots(figsize=(7, 7))
                    im = ax.imshow(prob, cmap=cmap_prob, vmin=0, vmax=1)
                    ax.set_title(f"Year {yr} – Probability", fontsize=14, fontweight='bold')
                    ax.axis("off")
                    fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046)
                    st.pyplot(fig)
                    plt.close()

                    bin_map = (prob > binary_thresh).astype(np.uint8) * 255
                    bin_large = cv2.resize(bin_map, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
                    st.image(bin_large, caption=f"Binary {yr} (white = urban)", clamp=True)

                    area_km2 = bin_map.sum() / 255 * (PIXEL_AREA_M2 / 1_000_000)
                    st.metric(f"Urban area {yr}", f"{area_km2:.2f} km²")

            # Year-to-Year Change
            if len(years_list) > 1:
                st.markdown("---")
                st.markdown("### Year-to-Year Probability Change")
                for j in range(1, len(years_list)):
                    prev, curr = years_list[j - 1], years_list[j]
                    diff = predictions[curr] - predictions[prev]
                    fig, ax = plt.subplots(figsize=(7, 7))
                    im = ax.imshow(diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
                    ax.set_title(f"Change {prev} → {curr}\n(Red = increase)", fontsize=12, fontweight='bold')
                    ax.axis("off")
                    fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046)
                    st.pyplot(fig)
                    plt.close()

            # New Built-up Expansion
            st.markdown("---")
            st.markdown("###  New Built-up Expansion")
            st.caption("Pixel = 100 m² → small house plot size")

            last_known_bin = (sequence[-1] > binary_thresh).astype(np.uint8)
            cols_exp = st.columns(min(3, len(years_list)))
            for i, yr in enumerate(years_list):
                prob = predictions[yr]
                pred_bin = (prob > binary_thresh).astype(np.uint8)
                expansion = np.clip(pred_bin - last_known_bin, 0, 1) * 255

                with cols_exp[i % len(cols_exp)]:
                    fig, ax = plt.subplots(figsize=(7, 7))
                    ax.imshow(expansion, cmap="Reds", vmin=0, vmax=255)
                    ax.set_title(f"New in {yr}", fontsize=12, fontweight='bold')
                    ax.axis("off")
                    st.pyplot(fig)
                    plt.close()

                    exp_large = cv2.resize(expansion, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
                    st.image(exp_large, caption=f"New Built-up {yr} (red)", clamp=True)

                    exp_area = expansion.sum() / 255 * (PIXEL_AREA_M2 / 1_000_000)
                    st.metric(f"New built-up {yr}", f"{exp_area:.2f} km²")

                last_known_bin = pred_bin

            # Growth Charts
            st.markdown("---")
            st.markdown("###  Growth Trends")

            urban_areas = [(prob > binary_thresh).sum() * (PIXEL_AREA_M2 / 1_000_000) for prob in predictions.values()]

            fig_growth, ax = plt.subplots(figsize=(10, 5))
            ax.plot(years_list, urban_areas, marker='o', linewidth=2, markersize=8, color='#1f77b4')
            ax.set_title("Total Urban Area Growth", fontsize=14, fontweight='bold')
            ax.set_ylabel("km²", fontsize=12)
            ax.set_xlabel("Year", fontsize=12)
            ax.grid(True, alpha=0.4)
            st.pyplot(fig_growth)
            plt.close()

            # New expansion per year
            new_areas = []
            prev_bin = (sequence[-1] > binary_thresh).astype(np.float32)
            for yr in years_list:
                curr_bin = (predictions[yr] > binary_thresh).astype(np.float32)
                new_km2 = ((curr_bin - prev_bin) > 0).sum() * (PIXEL_AREA_M2 / 1_000_000)
                new_areas.append(new_km2)
                prev_bin = curr_bin

            fig_bar, ax = plt.subplots(figsize=(10, 5))
            ax.bar(years_list, new_areas, color='#ff7f0e', alpha=0.8)
            ax.set_title("New Expansion per Year", fontsize=14, fontweight='bold')
            ax.set_ylabel("km²", fontsize=12)
            ax.set_xlabel("Year", fontsize=12)
            ax.grid(True, axis='y', alpha=0.4)
            st.pyplot(fig_bar)
            plt.close()

            # Export button
            st.markdown("---")
            if st.button(" Export Predictions as PNG"):
                run_num = len(os.listdir(PAST_PRED_DIR)) + 1
                folder = os.path.join(PAST_PRED_DIR, f"export_{run_num}")
                os.makedirs(folder, exist_ok=True)
                saved = 0

                for yr, prob in predictions.items():
                    bin_map = (prob > binary_thresh).astype(np.uint8) * 255
                    bin_large = cv2.resize(bin_map, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
                    Image.fromarray(bin_large).save(os.path.join(folder, f"binary_{yr}.png"))
                    saved += 1

                st.success(f"✅ Saved {saved} PNG files to:\n`{folder}`")

        else:
            # ═══════════════════════════════════════════════════════
            # POKHARA VISUALIZATION (Multi-class)
            # ═══════════════════════════════════════════════════════
            st.markdown("---")
            st.markdown("##  Predicted Land Cover Maps")

            colors = np.array([
                [139, 90, 43],  # Brown
                [50, 100, 255],  # Blue
                [50, 200, 50],  # Green
                [255, 50, 50]  # Red
            ]) / 255.0
            cmap_landcover = ListedColormap(colors)

            cols = st.columns(min(3, len(years_list)))
            for i, yr in enumerate(years_list):
                prob_3ch = predictions[yr]

                with cols[i % len(cols)]:
                    multiclass = np.zeros(prob_3ch.shape[1:], dtype=np.int32)
                    multiclass[prob_3ch[0] > binary_thresh] = 1
                    multiclass[prob_3ch[1] > binary_thresh] = 2
                    multiclass[prob_3ch[2] > binary_thresh] = 3

                    fig, ax = plt.subplots(figsize=(8, 8))
                    im = ax.imshow(multiclass, cmap=cmap_landcover, vmin=0, vmax=3, interpolation='nearest')
                    ax.set_title(f"Year {yr}", fontsize=16, fontweight='bold')
                    ax.axis("off")
                    st.pyplot(fig)
                    plt.close()

                    water_area = (multiclass == 1).sum() * PIXEL_AREA_M2 / 1_000_000
                    veg_area = (multiclass == 2).sum() * PIXEL_AREA_M2 / 1_000_000
                    urban_area = (multiclass == 3).sum() * PIXEL_AREA_M2 / 1_000_000

                    st.metric("🔵 Water", f"{water_area:.3f} km²")
                    st.metric("🟢 Vegetation", f"{veg_area:.3f} km²")
                    st.metric("🔴 Urban", f"{urban_area:.3f} km²")

            # Probability Heatmaps
            st.markdown("---")
            st.markdown("## 🔥 Probability Heatmaps by Class")

            first_year = years_list[0]
            prob_3ch_example = predictions[first_year]

            col_w, col_v, col_u = st.columns(3)

            with col_w:
                st.markdown(f"### 🔵 Water Probability {first_year}")
                fig, ax = plt.subplots(figsize=(6, 6))
                im = ax.imshow(prob_3ch_example[0], cmap='Blues', vmin=0, vmax=1)
                ax.set_title("Water", fontsize=14, fontweight='bold')
                ax.axis("off")
                fig.colorbar(im, ax=ax, orientation='horizontal')
                st.pyplot(fig)
                plt.close()

            with col_v:
                st.markdown(f"### 🟢 Vegetation Probability {first_year}")
                fig, ax = plt.subplots(figsize=(6, 6))
                im = ax.imshow(prob_3ch_example[1], cmap='Greens', vmin=0, vmax=1)
                ax.set_title("Vegetation", fontsize=14, fontweight='bold')
                ax.axis("off")
                fig.colorbar(im, ax=ax, orientation='horizontal')
                st.pyplot(fig)
                plt.close()

            with col_u:
                st.markdown(f"### 🔴 Urban Probability {first_year}")
                fig, ax = plt.subplots(figsize=(6, 6))
                im = ax.imshow(prob_3ch_example[2], cmap='Reds', vmin=0, vmax=1)
                ax.set_title("Urban", fontsize=14, fontweight='bold')
                ax.axis("off")
                fig.colorbar(im, ax=ax, orientation='horizontal')
                st.pyplot(fig)
                plt.close()

            # Growth charts
            st.markdown("---")
            st.markdown("## 📈 Land Cover Change Trends")

            water_areas, veg_areas, urban_areas = [], [], []
            for prob_3ch in predictions.values():
                water_areas.append((prob_3ch[0] > binary_thresh).sum() * PIXEL_AREA_M2 / 1_000_000)
                veg_areas.append((prob_3ch[1] > binary_thresh).sum() * PIXEL_AREA_M2 / 1_000_000)
                urban_areas.append((prob_3ch[2] > binary_thresh).sum() * PIXEL_AREA_M2 / 1_000_000)

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(years_list, water_areas, marker='^', linewidth=2, markersize=8, label='🔵 Water', color='#3264ff')
            ax.plot(years_list, veg_areas, marker='s', linewidth=2, markersize=8, label='🟢 Vegetation', color='#32c832')
            ax.plot(years_list, urban_areas, marker='o', linewidth=2, markersize=8, label='🔴 Urban', color='#ff3232')
            ax.set_xlabel("Year", fontsize=12)
            ax.set_ylabel("Area (km²)", fontsize=12)
            ax.set_title(f"Land Cover Change - {config['display_name']}", fontsize=14, fontweight='bold')
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()

            if len(urban_areas) > 1:
                water_change = water_areas[-1] - water_areas[0]
                veg_change = veg_areas[-1] - veg_areas[0]
                urban_change = urban_areas[-1] - urban_areas[0]

                col1, col2, col3 = st.columns(3)
                col1.metric("🔵 Water Change", f"{water_change:+.2f} km²",
                            delta=f"{water_change / water_areas[0] * 100:+.1f}%" if water_areas[0] > 0 else "N/A")
                col2.metric("🟢 Vegetation Change", f"{veg_change:+.2f} km²",
                            delta=f"{veg_change / veg_areas[0] * 100:+.1f}%" if veg_areas[0] > 0 else "N/A")
                col3.metric("🔴 Urban Change", f"{urban_change:+.2f} km²",
                            delta=f"{urban_change / urban_areas[0] * 100:+.1f}%" if urban_areas[0] > 0 else "N/A")

    else:
        st.info("Click '🚀 Run Future Prediction' to generate results.")

else:
    st.info(f"👆 Draw a rectangle on the {config['display_name']} map to start prediction.")