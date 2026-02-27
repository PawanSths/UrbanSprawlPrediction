"""
Multiclass Segmentation Page — Hyderabad Land Cover Classification
Memory-efficient: Only processes user-selected cropped regions (max 1000×1000 px)
Uses sliding window with proper Colab normalization
"""

import json
import os
import gc
from typing import Tuple
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from rasterio.crs import CRS
from rasterio.windows import from_bounds
from rasterio.warp import transform
from sklearn.metrics import confusion_matrix
import folium
from folium import plugins
from streamlit_folium import st_folium

from core.config import REGIONS, RegionConfig
from core.loaders import load_unet

st.set_page_config(page_title="Multiclass Segmentation", layout="wide")

# ════════════════════════════════════════════════════════════════════
# NORMALIZATION CONSTANTS (From your Colab training)
# ════════════════════════════════════════════════════════════════════

NORMALIZATION_STATS = {
    "hyderabad": {
        "mean": np.array([1.0546437e+03, 1.4515833e+03, 1.5928341e+03, 2.8228894e+03,
                         3.4636748e+03, 2.5146472e+03, 2.6376092e-01, -3.3397126e-01,
                         -2.7405408e-01, 4.9883842e-02]),
        "std": np.array([3.8909647e+02, 4.0717056e+02, 5.1408856e+02, 5.7885687e+02,
                        8.1401770e+02, 7.0319159e+02, 1.6960555e-01, 1.5608695e-01,
                        1.5603507e-01, 1.6167323e-01]),
    },
}


# ════════════════════════════════════════════════════════════════════
# PREDICTION FUNCTION (Your sliding window logic - KEPT INTACT)
# ════════════════════════════════════════════════════════════════════

def predict_cropped_image(
    model,
    img: np.ndarray,
    region_name: str,
    patch_size: int = 256,
    batch_size: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Memory-efficient prediction on CROPPED images using sliding window.
    Matches your Colab code exactly.
    """
    H, W, C = img.shape
    full_mask = np.zeros((H, W), dtype=np.uint8)
    full_confidence = np.zeros((H, W), dtype=np.float32)

    if region_name not in NORMALIZATION_STATS:
        st.error(f" No normalization stats for '{region_name}'")
        return None, None

    stats = NORMALIZATION_STATS[region_name]
    global_mean = stats["mean"]
    global_std = stats["std"]

    patches = []
    coords = []

    # Calculate total patches for progress
    total_patches = ((H + patch_size - 1) // patch_size) * ((W + patch_size - 1) // patch_size)
    processed = 0
    progress_bar = st.progress(0, text="Starting...")
    status_text = st.empty()

    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            orig_end_i = min(i + patch_size, H)
            orig_end_j = min(j + patch_size, W)
            orig_h = orig_end_i - i
            orig_w = orig_end_j - j

            patch = img[i:orig_end_i, j:orig_end_j, :]

            pad_i = patch_size - orig_h
            pad_j = patch_size - orig_w
            if pad_i > 0 or pad_j > 0:
                patch = np.pad(patch, ((0, pad_i), (0, pad_j), (0, 0)), mode='reflect')

            patch = np.nan_to_num(patch, nan=0.0)
            patch = (patch - global_mean[None, None, :]) / global_std[None, None, :]

            patches.append(patch)
            coords.append((i, j, orig_h, orig_w))

            if len(patches) == batch_size:
                batch = np.stack(patches, axis=0)
                preds = model.predict(batch, verbose=0)

                for k, (x, y, oh, ow) in enumerate(coords):
                    mask_patch = np.argmax(preds[k], axis=-1)
                    conf_patch = np.max(preds[k], axis=-1)
                    full_mask[x:x+oh, y:y+ow] = mask_patch[:oh, :ow]
                    full_confidence[x:x+oh, y:y+ow] = conf_patch[:oh, :ow]

                # 🔥 Cleanup batch memory
                del batch, preds
                patches, coords = [], []
                processed += batch_size

            # Update progress per patch (smoother UX)
            current = processed + len(patches)
            progress = min(current / total_patches, 1.0)
            progress_bar.progress(progress, text=f"Processed {current}/{total_patches} patches...")
            status_text.text(f"Working... {int(progress*100)}%")

    # Remaining patches
    if len(patches) > 0:
        batch = np.stack(patches, axis=0)
        preds = model.predict(batch, verbose=0)

        for k, (x, y, oh, ow) in enumerate(coords):
            mask_patch = np.argmax(preds[k], axis=-1)
            conf_patch = np.max(preds[k], axis=-1)
            full_mask[x:x+oh, y:y+ow] = mask_patch[:oh, :ow]
            full_confidence[x:x+oh, y:y+ow] = conf_patch[:oh, :ow]

        del batch, preds

    #  Final cleanup
    progress_bar.empty()
    status_text.empty()
    gc.collect()

    return full_mask, full_confidence


# ════════════════════════════════════════════════════════════════════
# REGION CHECK & CONFIG
# ════════════════════════════════════════════════════════════════════

region: RegionConfig | None = st.session_state.get("region")
if region is None:
    st.warning("Please select a region from the main page first.")
    st.stop()


if not region.has_multiclass_unet:
    st.error(f" No multiclass U-Net model available for **{region.display_name}**.")
    st.stop()

MODEL_PATH = str(region.unet_multiclass_path)
NUM_BANDS = region.num_bands
CLASS_NAMES = region.class_names or ["Urban", "Forest", "Water", "Barren"]
CLASS_COLORS = region.class_colors or ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]

if not os.path.exists(MODEL_PATH):
    st.error(f" Model not found: {MODEL_PATH}")
    st.stop()

# 🔧 FIX: Pass model_path as argument to cache_resource
@st.cache_resource
def _load_model_cached(model_path: str):
    return load_unet(model_path)

model = _load_model_cached(MODEL_PATH)

# ════════════════════════════════════════════════════════════════════
# PAGE TITLE
# ════════════════════════════════════════════════════════════════════

st.title(f" Multiclass Land Cover — {region.display_name}")


# ════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header(" Settings")
    years = region.image_years
    base_year = st.selectbox("Base Year", years, index=0)
    compare_year = st.selectbox("Comparison Year", years, index=len(years) - 1)

    if compare_year <= base_year:
        st.error("Comparison year must be after base year")
        st.stop()

    st.markdown("### Processing")
    patch_size = st.slider("Patch Size", 128, 512, 256, 64)
    batch_size = st.slider("Batch Size", 1, 8, 4, 1)

    st.markdown("### Display")
    show_metrics = st.checkbox("Show Metrics", value=True)
    show_transitions = st.checkbox("Show Transitions", value=True)
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.0, 0.1,
                               help="Hide predictions below this confidence (gray = low confidence)")

# ════════════════════════════════════════════════════════════════════
# MAP
# ════════════════════════════════════════════════════════════════════




center = region.map_center
zoom = region.map_zoom
bounds = region.map_bounds

if bounds:
    SW, NE = bounds
else:
    SW = [center[0] - 0.15, center[1] - 0.2]
    NE = [center[0] + 0.15, center[1] + 0.2]

m = folium.Map(location=list(center), zoom_start=zoom, min_zoom=zoom-1, max_zoom=16, max_bounds=True)
folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", attr="Esri").add_to(m)

draw = plugins.Draw(
    draw_options={"rectangle": True, "circle": False, "polygon": False, "marker": False, "polyline": False},
    edit_options={"edit": False, "remove": True}
)
draw.add_to(m)

map_data = st_folium(m, width=1200, height=600, key=f"map_{region.name}")

# ════════════════════════════════════════════════════════════════════
# PROCESS DRAWN ROI
# ════════════════════════════════════════════════════════════════════

if map_data and map_data.get("last_active_drawing"):
    coordinates = map_data["last_active_drawing"]["geometry"]["coordinates"][0]
    longitudes = [p[0] for p in coordinates]
    latitudes = [p[1] for p in coordinates]
    minx, maxx = min(longitudes), max(longitudes)
    miny, maxy = min(latitudes), max(latitudes)

    def load_crop(year_param):
        import rasterio
        img_path = str(region.images[year_param])
        with rasterio.open(img_path) as src:
            xs, ys = [minx, maxx], [miny, maxy]
            x_utm, y_utm = transform(src_crs=CRS.from_epsg(4326), dst_crs=src.crs, xs=xs, ys=ys)
            window = from_bounds(x_utm[0], y_utm[0], x_utm[1], y_utm[1], src.transform)
            data = src.read(window=window)
            if data.size == 0:
                data = src.read()
            img = np.transpose(data, (1, 2, 0)).astype("float32")
            return img[..., :NUM_BANDS]

    try:
        with st.spinner(f"Loading {base_year}..."):
            img_base = load_crop(base_year)
        with st.spinner(f"Loading {compare_year}..."):
            img_compare = load_crop(compare_year)
    except Exception as e:
        st.error(f" Error loading images: {e}")
        st.stop()

    img_h, img_w = img_base.shape[:2]
    MAX_PIXELS = 1200 * 1200
    if img_h * img_w > MAX_PIXELS:
        st.error(f" Area too large ({img_h}×{img_w} px, max 1000×1000)")
        st.stop()

    # Info metrics
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Crop Size", f"{img_h}×{img_w}")
    col2.metric("Bands", NUM_BANDS)
    col3.metric("Memory", f"{(img_h*img_w*NUM_BANDS*4)/1024**2:.2f} MB")
    col4.metric("Patches", ((img_h+patch_size-1)//patch_size) * ((img_w+patch_size-1)//patch_size))

    # 🔧 Run predictions WITH error checking
    st.markdown("---")
    st.markdown("### Running Predictions")

    pred_base_classes, pred_base_conf = predict_cropped_image(model, img_base, region.name, patch_size, batch_size)
    pred_compare_classes, pred_compare_conf = predict_cropped_image(model, img_compare, region.name, patch_size, batch_size)

    # 🔧 FIX: Check if predictions failed
    if pred_base_classes is None or pred_compare_classes is None:
        st.error(" Prediction returned no results. Check normalization stats or model path.")
        st.stop()

    st.success(" Predictions complete!")

    num_classes = model.output_shape[-1]
    if len(CLASS_NAMES) != num_classes:
        st.error(f"Model outputs {num_classes} classes but config has {len(CLASS_NAMES)}")
        st.stop()

    # ════════════════════════════════════════════════════════════════════
    # 🔧 VISUALIZATION (Fixed patterns that work)
    # ════════════════════════════════════════════════════════════════════

    st.markdown("###  Results")

    # ─── Row 1: RGB Images ───
    col_rgb_base, col_rgb_compare = st.columns(2)
    col_width = 500  # Adjust this as needed

    if NUM_BANDS >= 4:
        rgb_base = np.clip(img_base[..., [2, 1, 0]] / 3000, 0, 1)
        rgb_compare = np.clip(img_compare[..., [2, 1, 0]] / 3000, 0, 1)
    else:
        rgb_base = np.clip(img_base[..., :3] / 3000, 0, 1)
        rgb_compare = np.clip(img_compare[..., :3] / 3000, 0, 1)

    with col_rgb_base:
        st.markdown(f"**{base_year} Sentinel-2 Image**")
        st.image(rgb_base, width=col_width)

    with col_rgb_compare:
        st.markdown(f"**{compare_year}  Sentinel-2 Image**")
        st.image(rgb_compare, width=col_width)

    # ─── Row 2: Predicted Classes ───
    col_pred_base, col_pred_compare = st.columns(2)

    pred_base_display = np.where(pred_base_conf >= conf_threshold, pred_base_classes, 255)
    pred_compare_display = np.where(pred_compare_conf >= conf_threshold, pred_compare_classes, 255)

    with col_pred_base:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        cmap = plt.cm.colors.ListedColormap(CLASS_COLORS + ["#888888"])
        ax.imshow(pred_base_display, cmap=cmap, vmin=0, vmax=num_classes)
        ax.set_title(f"{base_year} Classification", fontsize=12, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig, width=col_width)
        plt.close(fig)
        gc.collect()

    with col_pred_compare:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        cmap = plt.cm.colors.ListedColormap(CLASS_COLORS + ["#888888"])
        ax.imshow(pred_compare_display, cmap=cmap, vmin=0, vmax=num_classes)
        ax.set_title(f"{compare_year} Classification", fontsize=12, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig, width=col_width)
        plt.close(fig)
        gc.collect()

    # Change detection
    st.markdown("---")
    change_mask = (pred_base_classes != pred_compare_classes).astype("uint8")

    col_change1, col_change2 = st.columns(2)

    with col_change1:
        fig, ax = plt.subplots(figsize=(7, 7), dpi=100)
        ax.imshow(rgb_compare)
        ax.imshow(change_mask, cmap="RdYlGn_r", alpha=0.6, vmin=0, vmax=1)
        ax.set_title(f"Changes ({base_year}→{compare_year})", fontsize=12, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        gc.collect()

    with col_change2:
        cm = confusion_matrix(pred_base_classes.flatten(), pred_compare_classes.flatten(), labels=list(range(num_classes)))
        fig, ax = plt.subplots(figsize=(8, 7), dpi=100)
        im = ax.imshow(cm, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(CLASS_NAMES, fontsize=9)
        ax.set_xlabel(f"{compare_year}", fontsize=10)
        ax.set_ylabel(f"{base_year}", fontsize=10)
        ax.set_title("Transition Matrix", fontsize=12, fontweight="bold")

        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                       color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=9)
        plt.colorbar(im, ax=ax, label="Pixels")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        gc.collect()

    # Metrics
    if show_metrics:
        st.markdown("---")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        changed = change_mask.sum()
        total = change_mask.size
        col_m1.metric("Changed Pixels", f"{changed:,}", f"{(changed/total)*100:.1f}%")
        col_m2.metric(f"{base_year} Confidence", f"{pred_base_conf.mean():.3f}", f"min: {pred_base_conf.min():.3f}")
        col_m3.metric(f"{compare_year} Confidence", f"{pred_compare_conf.mean():.3f}", f"min: {pred_compare_conf.min():.3f}")
        col_m4.metric("Unchanged", f"{total-changed:,}", f"{(100-(changed/total)*100):.1f}%")

    # Distribution charts
    st.markdown("---")
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        fig, ax = plt.subplots(figsize=(9, 6), dpi=100)
        unique, counts = np.unique(pred_base_classes, return_counts=True)
        colors = [CLASS_COLORS[i] if i < len(CLASS_COLORS) else "#888" for i in unique]
        labels = [CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"C{i}" for i in unique]
        ax.bar(labels, counts, color=colors, edgecolor="black", linewidth=1.2)
        ax.set_title(f"Distribution - {base_year}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Pixels")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        gc.collect()

    with col_d2:
        fig, ax = plt.subplots(figsize=(9, 6), dpi=100)
        unique, counts = np.unique(pred_compare_classes, return_counts=True)
        colors = [CLASS_COLORS[i] if i < len(CLASS_COLORS) else "#888" for i in unique]
        labels = [CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"C{i}" for i in unique]
        ax.bar(labels, counts, color=colors, edgecolor="black", linewidth=1.2)
        ax.set_title(f"Distribution - {compare_year}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Pixels")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        gc.collect()

    # Transitions table
    if show_transitions:
        st.markdown("---")
        st.markdown("### Top Transitions")
        transition_data = []
        for i in range(num_classes):
            for j in range(num_classes):
                if i != j:
                    count = np.sum((pred_base_classes == i) & (pred_compare_classes == j))
                    if count > 0:
                        transition_data.append({
                            "From": CLASS_NAMES[i], "To": CLASS_NAMES[j],
                            "Pixels": int(count), "%": f"{(count / max(change_mask.sum(), 1)) * 100:.2f}%"
                        })
        transition_data.sort(key=lambda x: x["Pixels"], reverse=True)
        if transition_data:
            st.dataframe(transition_data[:15], use_container_width=True, hide_index=True)
        else:
            st.info("No significant transitions detected.")

    # Export
    st.markdown("---")
    export_stats = {
        "region": region.display_name, "base_year": int(base_year), "compare_year": int(compare_year),
        "crop_size": f"{img_h}×{img_w}", "changed_pixels": int(change_mask.sum()),
        "change_percentage": float((change_mask.sum() / change_mask.size) * 100),
    }
    st.download_button(" Download Report (JSON)", json.dumps(export_stats, indent=2),
                      f"report_{region.name}_{base_year}_{compare_year}.json", "application/json")

    # 🔧 Final cleanup
    gc.collect()

else:
    st.info(" **Draw a rectangle** on the map above to select your analysis area")