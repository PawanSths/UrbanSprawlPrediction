
import json
import os
from typing import TYPE_CHECKING

import cv2
import folium
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from folium import plugins
from rasterio.crs import CRS
from rasterio.windows import from_bounds
from rasterio.warp import transform
from sklearn.metrics import jaccard_score
from streamlit_folium import st_folium

from core.config import RegionConfig
from core.loaders import load_unet



if TYPE_CHECKING:
    import rasterio



region: RegionConfig | None = st.session_state.get("region")
#egion: RegionConfig | None = st.session_state.get("region")

if region is None:
    st.warning("Please select a region from the main page first.")
    st.stop()

if not region.has_binary_unet:
    st.error(f" No binary U-Net model available for **{region.display_name}**.")
    st.info(f"This region uses {region.model_type} segmentation instead.")
    st.stop()

DATA_DIR = str(region.data_dir)
MODEL_PATH = str(region.unet_path)

# Validate image files exist
missing = []
for year_check in region.image_years:
    path_check = str(region.images[year_check])
    if not os.path.exists(path_check):
        missing.append(year_check)
if missing:
    st.error(f"Missing images for years: {missing}")
    st.stop()

# ────────────────────────────────────────────────
# MODEL (cached via core.loaders)
# ────────────────────────────────────────────────

model = load_unet(MODEL_PATH)

# ────────────────────────────────────────────────
# SIDEBAR
# ────────────────────────────────────────────────

with st.sidebar:
    st.header(f"U-Net Settings — {region.display_name}")
    years = region.image_years
    base_year = st.selectbox("Base Year", years, index=0)
    truth_year = st.selectbox("Truth Year", years, index=len(years) - 1)
    if truth_year <= base_year:
        st.warning("Truth year must be after base year")
        st.stop()

# ────────────────────────────────────────────────
# MAP (region-aware bounds)
# ────────────────────────────────────────────────

bounds = region.map_bounds
center = region.map_center
zoom = region.map_zoom

if bounds:
    SW, NE = bounds
else:
    SW = [center[0] - 0.15, center[1] - 0.2]
    NE = [center[0] + 0.15, center[1] + 0.2]

m = folium.Map(
    location=list(center),
    zoom_start=zoom,
    min_zoom=zoom - 1,
    max_zoom=16,
    max_bounds=True,
    min_lat=SW[0],
    max_lat=NE[0],
    min_lon=SW[1],
    max_lon=NE[1],
)

folium.TileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
    name="Satellite",
).add_to(m)

draw = plugins.Draw(
    draw_options={
        "rectangle": True,
        "circle": False,
        "polygon": False,
        "marker": False,
        "polyline": False,
    }
)
draw.add_to(m)

map_data = st_folium(m, width=1000, height=600, key=f"unet_map_{region.name}")

# ────────────────────────────────────────────────
# PREDICTION
# ────────────────────────────────────────────────

if map_data.get("last_active_drawing"):
    coordinates = map_data["last_active_drawing"]["geometry"]["coordinates"][0]
    longitudes = [p[0] for p in coordinates]
    latitudes = [p[1] for p in coordinates]
    minx, maxx = min(longitudes), max(longitudes)
    miny, maxy = min(latitudes), max(latitudes)

    def load_crop(year_param):
        """Load and crop a satellite image for the given year."""
        import rasterio  # Local import to avoid TYPE_CHECKING issues

        img_path = str(region.images[year_param])
        # noinspection PyTypeChecker
        with rasterio.open(img_path) as src:
            xs = [minx, maxx]
            ys = [miny, maxy]
            x_utm, y_utm = transform(
                src_crs=CRS.from_epsg(4326),  # Type-safe CRS
                dst_crs=src.crs,
                xs=xs,
                ys=ys,
            )
            window = from_bounds(x_utm[0], y_utm[0], x_utm[1], y_utm[1], src.transform)
            data = src.read(window=window)
            if data.size == 0:
                data = src.read()
            return np.transpose(data, (1, 2, 0)).astype("float32"), src

    img_base, _ = load_crop(base_year)
    img_truth, _ = load_crop(truth_year)

    patch_base = cv2.resize(img_base, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    patch_truth_img = cv2.resize(img_truth, (256, 256), interpolation=cv2.INTER_LANCZOS4)

    with st.spinner("Predicting urban change..."):
        prediction_base = model.predict(patch_base[np.newaxis, ...], verbose=0)[0, :, :, 0]
        prediction_truth = model.predict(patch_truth_img[np.newaxis, ...], verbose=0)[0, :, :, 0]

        thresh = st.slider(
            "Urban detection threshold",
            0.3,
            0.7,
            0.5,
            0.05,
            help="Lower = more urban detected | Higher = stricter",
        )

        # noinspection SpellCheckingInspection
        pred_base_bin = (prediction_base >= thresh).astype("uint8")  # "pred" is intentional
        # noinspection SpellCheckingInspection
        pred_truth_bin = (prediction_truth >= thresh).astype("uint8")
        expansion = pred_truth_bin - pred_base_bin
        expansion = np.clip(expansion, 0, 1)

    # ════════════════════════════════════════════════
    # MAIN VISUALS
    # ════════════════════════════════════════════════

    st.markdown(f"### Urban Expansion Analysis — {region.display_name} (U-Net)")
    col1, col2, col3, col4 = st.columns(4)

    rgb_base = np.clip(patch_base[..., [2, 1, 0]] / 3000, 0, 1)
    rgb_truth = np.clip(patch_truth_img[..., [2, 1, 0]] / 3000, 0, 1)

    with col1:
        st.image(rgb_base, caption=f"{base_year} Image")

    with col2:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.imshow(rgb_base)
        ax.imshow(pred_base_bin, cmap="Reds", alpha=0.75)
        ax.set_title(f"Predicted {base_year}")
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)

    with col3:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.imshow(rgb_truth)
        ax.imshow(expansion, cmap="Blues", alpha=0.75)
        ax.set_title(f"New Built-Up\n({base_year}→{truth_year})")
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)

    with col4:
        st.image(rgb_truth, caption=f"{truth_year} Image")

    # ════════════════════════════════════════════════
    # METRICS
    # ════════════════════════════════════════════════

    iou = jaccard_score(pred_truth_bin.flatten(), pred_base_bin.flatten())
    new_km2 = expansion.sum() * (10 * 10 / 1e6)
    growth = (new_km2 / (pred_base_bin.sum() * 0.0001) * 100) if pred_base_bin.sum() > 0 else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("IoU (Model)", f"{iou:.3f}")
    c2.metric("New Built-Up", f"{new_km2:.3f} km²")
    c3.metric("Growth Rate", f"{growth:+.1f}%")

    # ════════════════════════════════════════════════
    # PIE CHART + INSIGHTS
    # ════════════════════════════════════════════════

    st.markdown("---")
    st.markdown("### Quick Insights & Change Summary")

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        fig_pie, ax_pie = plt.subplots(figsize=(7, 6))
        urban_base = np.sum(pred_base_bin)
        urban_truth = np.sum(pred_truth_bin)
        new_built = np.sum(expansion)
        lost_urban = np.sum(pred_base_bin * (1 - pred_truth_bin))

        change_types = ["Urban in Base Year", "Urban in Truth Year", "New Built-up", "Lost Urban"]
        values = [urban_base, urban_truth, new_built, lost_urban]
        colors = ["#e74c3c", "#3498db", "#f39c12", "#95a5a6"]

        # noinspection SpellCheckingInspection
        wedges, _ = ax_pie.pie(
            values,
            colors=colors,
            startangle=90,
            explode=[0.04, 0.04, 0.08, 0.04],
            shadow=True,
            wedgeprops={"edgecolor": "white", "linewidth": 1.2},  # edgecolor is valid
        )

        legend_labels = []
        total = sum(values)
        for label, val in zip(change_types, values, strict=False):
            percent = (val / total * 100) if total > 0 else 0
            legend_labels.append(f"{label}: {percent:.1f}% ({int(val)} pixels)")

        # noinspection SpellCheckingInspection
        ax_pie.legend(
            wedges,
            legend_labels,
            title="Legend",
            loc="center left",
            bbox_to_anchor=(1.0, 0.5, 0.5, 1),
            fontsize=10,
            title_fontsize=12,
            frameon=True,
            edgecolor="gray",  # edgecolor is valid
        )
        ax_pie.set_title(
            f"Urban Area Comparison\n({base_year} vs {truth_year})",
            fontsize=14,
            pad=20,
        )
        st.pyplot(fig_pie)
        plt.close(fig_pie)

    with col_g2:
        fig_trend, ax_trend = plt.subplots(figsize=(6, 6))
        years_plot = [base_year, truth_year]
        urban_pixels = [np.sum(pred_base_bin), np.sum(pred_truth_bin)]
        ax_trend.plot(years_plot, urban_pixels, marker="o", color="#e74c3c", linewidth=2)
        ax_trend.fill_between(years_plot, urban_pixels, color="#e74c3c", alpha=0.15)
        ax_trend.set_title("Urban Growth Trend", fontsize=13)
        ax_trend.set_xlabel("Year", fontsize=11)
        ax_trend.set_ylabel("Urban Pixels", fontsize=11)
        ax_trend.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig_trend)
        plt.close(fig_trend)

    # Download
    report = {
        "region": region.display_name,
        "base_year": base_year,
        "truth_year": truth_year,
        "iou": round(iou, 3),
        "new_built_up_km2": round(new_km2, 3),
        "growth_percent": round(growth, 1),
    }
    st.download_button(
        label=" Download Report ",
        data=json.dumps(report, indent=2),
        file_name=f"urban_expansion_{region.name}_{base_year}_{truth_year}.json",
        mime="application/json",
    )

else:
    st.info(f"**Draw a rectangle in {region.display_name}** to begin analysis.")