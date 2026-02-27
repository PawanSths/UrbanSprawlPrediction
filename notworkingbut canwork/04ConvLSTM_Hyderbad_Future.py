# pages/04_ConvLSTM_Hyderabad_Future.py
# ConvLSTM future urban prediction for Hyderabad

import streamlit as st
import folium
from streamlit_folium import st_folium
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
import numpy as np
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from PIL import Image

# ─── PATH CONFIGURATION ──────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "hyderabad")
MODEL_PATH = os.path.join(DATA_DIR, "hyderabad_convlstm.h5")

PAST_PRED_DIR = os.path.join(DATA_DIR, "past_prediction")
os.makedirs(PAST_PRED_DIR, exist_ok=True)

SEQ_LEN_MODEL = 8    # ConvLSTM time steps
CHANNELS_MODEL = 4   # ConvLSTM input channels
PIXEL_AREA_M2 = 100  # 10m x 10m pixels
DEVICE = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"

# ─── LOAD CONVLSTM MODEL ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Hyderabad ConvLSTM model not found:\n{MODEL_PATH}")
        st.stop()
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

convlstm_model = load_model()

# ─── INTERACTIVE MAP ─────────────────────────────────────────────────────────
SW = [17.35, 78.30]  # Hyderabad
NE = [17.55, 78.60]

m = folium.Map(
    location=[17.45, 78.45],
    zoom_start=11,
    min_zoom=10, max_zoom=16,
    max_bounds=True,
    min_lat=SW[0], max_lat=NE[0],
    min_lon=SW[1], max_lon=NE[1],
)

folium.TileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri", name="Satellite"
).add_to(m)

draw = folium.plugins.Draw(draw_options={"rectangle": True})
draw.add_to(m)

map_data = st_folium(m, width=1000, height=600, key="convlstm_map")

# ─── LOAD HISTORICAL MASK SEQUENCE ───────────────────────────────────────────
if map_data and map_data.get("last_active_drawing"):
    coords = map_data["last_active_drawing"]["geometry"]["coordinates"][0]
    lons = [p[0] for p in coords]
    lats = [p[1] for p in coords]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)

    with st.spinner("Loading historical mask sequence..."):
        sequence = []
        years_in = []
        reference_shape = (256, 256)  # model input

        for y in range(2017, 2026):
            path = os.path.join(DATA_DIR, f"mask_{y}.tif")
            if not os.path.exists(path):
                continue
            with rasterio.open(path) as src:
                wb = transform_bounds("EPSG:4326", src.crs, min_lon, min_lat, max_lon, max_lat)
                window = from_bounds(*wb, src.transform)
                mask = src.read(1, window=window).astype(np.float32)

                if mask.max() > 1.5:
                    mask = (mask > 127).astype(np.float32)
                mask = np.clip(mask, 0.0, 1.0)

                # resize to model input
                mask = cv2.resize(mask, (reference_shape[1], reference_shape[0]), cv2.INTER_NEAREST)
                sequence.append(mask)
                years_in.append(y)

        if len(sequence) < SEQ_LEN_MODEL:
            st.error(f"Need at least {SEQ_LEN_MODEL} past masks. Found: {len(sequence)} ({years_in})")
            st.stop()

    # ─── USER CONTROLS ───────────────────────────────────────────────────────
    st.markdown("### ConvLSTM Future Urban Prediction (Hyderabad)")
    st.write(f"**Input years:** {years_in[0]} – {years_in[-1]}")

    max_future_year = st.slider("Predict up to year", 2026, 2035, 2030, step=1)
    binary_thresh = st.slider("Binary urban threshold", 0.1, 0.9, 0.5, 0.05)

    if "predictions" not in st.session_state:
        st.session_state.predictions = None
        st.session_state.years_list = None

    if st.button("Run Future Prediction"):
        with st.spinner(f"Predicting → {max_future_year}..."):
            current_seq = sequence[-SEQ_LEN_MODEL:]
            predictions = {}
            current_year = years_in[-1]

            while current_year < max_future_year:
                current_year += 1

                # prepare input: (batch=1, time_steps=8, height=256, width=256, channels=4)
                seq_for_model = current_seq[-SEQ_LEN_MODEL:]
                if len(seq_for_model) < SEQ_LEN_MODEL:
                    seq_for_model = [seq_for_model[-1]] * (SEQ_LEN_MODEL - len(seq_for_model)) + seq_for_model

                seq_np = np.stack([np.stack([m]*CHANNELS_MODEL, axis=-1) for m in seq_for_model], axis=0)
                seq_np = np.expand_dims(seq_np, axis=0)

                with tf.device(DEVICE):
                    pred = convlstm_model.predict(seq_np, verbose=0)

                # take first channel for probability
                prob = pred[0, ..., 0]
                predictions[current_year] = prob

                pred_bin = (prob > binary_thresh).astype(np.float32)
                current_seq = current_seq[1:] + [pred_bin]

            st.session_state.predictions = predictions
            st.session_state.years_list = list(predictions.keys())

# ─── VISUALIZATION ─────────────────────────────────────────────────────────
if st.session_state.get("predictions") is not None:
    predictions = st.session_state.predictions
    years_list = st.session_state.years_list

    st.markdown("### Predicted Urban Probability Maps")
    colors = ['white', 'yellow', 'orange', 'darkred']
    cmap_prob = LinearSegmentedColormap.from_list("urban_risk", colors)
    vis_thresh = st.slider("Highlight probability ≥", 0.0, 1.0, 0.5, 0.05, key="vis_thresh")

    cols = st.columns(min(3, len(years_list)))
    for i, yr in enumerate(years_list):
        prob = predictions[yr]
        with cols[i % len(cols)]:
            fig, ax = plt.subplots(figsize=(7,7))
            im = ax.imshow(prob, cmap=cmap_prob, vmin=0, vmax=1)
            ax.set_title(f"Year {yr} – Probability")
            ax.axis("off")
            fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046)
            st.pyplot(fig)

            bin_map = (prob > binary_thresh).astype(np.uint8) * 255
            bin_large = cv2.resize(bin_map, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
            st.image(bin_large, caption=f"Binary {yr} (white = urban)", clamp=True)

    # ─── Growth Charts ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Urban Growth Trends")
    urban_areas = [(pred > binary_thresh).sum() * (PIXEL_AREA_M2 / 1_000_000) for pred in predictions.values()]
    fig_growth, ax = plt.subplots(figsize=(8,4))
    ax.plot(years_list, urban_areas, marker='o', color='#1f77b4')
    ax.set_title("Total Urban Area Growth")
    ax.set_ylabel("km²")
    ax.grid(True, alpha=0.4)
    st.pyplot(fig_growth)

    # ─── New Built-up Expansion ─────────────────────────────────────────────
    st.markdown("### New Built-up Expansion")
    last_known_bin = (sequence[-1] > binary_thresh).astype(np.uint8)
    cols_exp = st.columns(min(3, len(years_list)))
    for i, yr in enumerate(years_list):
        prob = predictions[yr]
        pred_bin = (prob > binary_thresh).astype(np.uint8)
        expansion = np.clip(pred_bin - last_known_bin, 0, 1) * 255
        with cols_exp[i % len(cols_exp)]:
            fig, ax = plt.subplots(figsize=(7,7))
            ax.imshow(expansion, cmap="Reds", vmin=0, vmax=255)
            ax.set_title(f"New in {yr}")
            ax.axis("off")
            st.pyplot(fig)

            exp_large = cv2.resize(expansion, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
            st.image(exp_large, caption=f"New Built-up {yr} (red)", clamp=True)

            exp_area = expansion.sum() / 255 * (PIXEL_AREA_M2 / 1_000_000)
            st.metric(f"New built-up {yr}", f"{exp_area:.2f} km²")

        last_known_bin = pred_bin
