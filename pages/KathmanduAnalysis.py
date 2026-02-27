#!/usr/bin/env python3

import streamlit as st
import numpy as np
import rasterio
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import gc

# ========================================
# PAGE CONFIG
# ========================================
st.set_page_config(
    page_title="Kathmandu Land Cover Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        color: #667eea;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        padding: 12px 24px;
    }
</style>
""", unsafe_allow_html=True)

# ========================================
# CONFIG
# ========================================
class Config:
    CLASS_NAMES = [ 'Forest','Urban']
    CLASS_COLORS = {
        0: [34, 204, 34],  # Urban - Red
        1: [255, 68, 68]  # Forest - Green
    }
    CLASS_COLORS_HEX = {
        0: "#22CC22",
        1: "#FF4444",
    }
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data" / "kathmandu"

# ========================================
# DATA LOADER
# ========================================
class DataLoader:
    @staticmethod
    def load_sentinel_image(year):
        path = Config.DATA_DIR / f"sentinel_kathmandu_{year}.tif"
        if not path.exists(): return None
        with rasterio.open(path) as src:
            img = src.read()
            img = np.moveaxis(img, 0, -1)
            return img.astype(np.float32)

    @staticmethod
    def load_mask(year):
        path = Config.DATA_DIR / f"mask_{year}.tif"
        if not path.exists(): return None
        with rasterio.open(path) as src:
            mask = src.read(1)

        # Map values: Urban=0, Forest=1
        MASK_MAPPING = {0: 0, 64: 1, 128: 1, 255: 0}
        vectorized_map = np.vectorize(lambda x: MASK_MAPPING.get(x, -1))
        mask_mapped = vectorized_map(mask).astype(np.int8)
        mask_mapped[mask_mapped == -1] = 1  # unknown -> Forest
        return mask_mapped

    @staticmethod
    def get_available_years():
        years = []
        for year in range(2019, 2026):
            mask_path = Config.DATA_DIR / f"mask_{year}.tif"
            if mask_path.exists(): years.append(year)
        return sorted(years)

# ========================================
# VISUALIZATION
# ========================================
class Visualizer:
    @staticmethod
    def colorize_mask(mask):
        h, w = mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for cls, color in Config.CLASS_COLORS.items():
            rgb[mask == cls] = color
        return rgb

    @staticmethod
    def get_rgb_from_sentinel(img):
        if img is None: return None
        if img.shape[-1] >= 4:
            rgb = img[..., [3, 2, 1]]
            rgb = np.clip(rgb / 3000.0 * 255, 0, 255).astype(np.uint8)
            return rgb
        return None

    @staticmethod
    def create_overlay(rgb, mask, alpha=0.6):
        if rgb is None: return Visualizer.colorize_mask(mask)
        colored_mask = Visualizer.colorize_mask(mask)
        return cv2.addWeighted(rgb, 1 - alpha, colored_mask, alpha, 0)

# ========================================
# ANALYSIS
# ========================================
class Analysis:
    @staticmethod
    def get_class_distribution(mask):
        unique, counts = np.unique(mask, return_counts=True)
        total = mask.size
        results = {}
        for cls_idx, count in zip(unique, counts):
            if cls_idx < len(Config.CLASS_NAMES):
                results[Config.CLASS_NAMES[cls_idx]] = {
                    'pixels': int(count),
                    'percentage': (count / total) * 100
                }
        for name in Config.CLASS_NAMES:
            if name not in results:
                results[name] = {'pixels': 0, 'percentage': 0.0}
        return results

    @staticmethod
    def compute_urban_change(mask1, mask2):
        # Automatic detection of Urban index per year
        def get_urban_index(mask):
            counts = {0: np.sum(mask==0), 1: np.sum(mask==1)}
            return 0 if counts[0] >= counts[1] else 1

        urban_idx1 = get_urban_index(mask1)
        urban_idx2 = get_urban_index(mask2)

        urban1 = (mask1 == urban_idx1).astype(np.uint8)
        urban2 = (mask2 == urban_idx2).astype(np.uint8)

        change_map = urban2.astype(np.int8) - urban1.astype(np.int8)
        total = mask1.size

        # Swap expansion and loss
        return {
            'change_map': change_map,
            'expansion_pixels': int(np.sum(change_map == -1)),  # previously loss
            'loss_pixels': int(np.sum(change_map == 1)),         # previously expansion
            'expansion_pct': (np.sum(change_map == -1) / total) * 100,
            'loss_pct': (np.sum(change_map == 1) / total) * 100,
            'urban1_pct': (np.sum(urban1) / total) * 100,
            'urban2_pct': (np.sum(urban2) / total) * 100,
            'net_change_pct': ((np.sum(urban2) - np.sum(urban1)) / total) * 100
        }

# ========================================
# INITIALIZATION
# ========================================
available_years = DataLoader.get_available_years()
if not available_years:
    st.error(" No mask files found in data/kathmandu. Expected format: mask_YYYY.tif")
    st.stop()

# ========================================
# SIDEBAR
# ========================================
with st.sidebar:
    st.markdown("<h2 style='color:#667eea;'>⚙ Configuration</h2>", unsafe_allow_html=True)

    analysis_mode = st.radio(
        " Select Analysis",
        ["Single Year", "Urban Change (2 Years)"],
    )

    st.markdown("---")

    if analysis_mode == "Single Year":
        selected_year = st.selectbox(
            " Select Year",
            available_years,
            index=len(available_years) - 1
        )
    else:
        col1, col2 = st.columns(2)
        with col1:
            year1 = st.selectbox(" Year 1", available_years, index=0, key="year1")
        with col2:
            year2 = st.selectbox(" Year 2", available_years, index=len(available_years) - 1, key="year2")

    st.markdown("---")

    overlay_alpha = st.slider(" Overlay Opacity", 0.0, 1.0, 0.6, 0.05)
    show_satellite = st.checkbox("Show Satellite Imagery", value=True)

# ========================================
# MAIN PAGE
# ========================================
st.markdown("<h1 class='main-header'> Kathmandu Land Cover Analyzer</h1>", unsafe_allow_html=True)
st.markdown("---")

# ===================== SINGLE YEAR =====================
if analysis_mode == "Single Year":
    st.subheader(f" Visualization for {selected_year}")

    if st.button("Load Data", type="primary", use_container_width=True):
        mask = DataLoader.load_mask(selected_year)
        if mask is None:
            st.error(f" No mask found for year {selected_year}")
            st.stop()

        rgb = None
        if show_satellite:
            img = DataLoader.load_sentinel_image(selected_year)
            if img is not None:
                rgb = Visualizer.get_rgb_from_sentinel(img)
            del img
            gc.collect()

        mask_h, mask_w = mask.shape
        st.info(f" Mask dimensions: {mask_w} × {mask_h} px")

        colored_mask = Visualizer.colorize_mask(mask)
        overlay = Visualizer.create_overlay(rgb, mask, overlay_alpha) if rgb is not None else colored_mask

        # Display
        st.markdown("### Visualization")
        if rgb is not None:
            col_rgb, col_mask, col_overlay = st.columns(3)
            with col_rgb: st.image(rgb, caption=f"Sentinel-2 RGB ({selected_year})", use_container_width=True)
            with col_mask: st.image(colored_mask, caption="Land Cover Classification", use_container_width=True)
            with col_overlay: st.image(overlay, caption="Overlay", use_container_width=True)
        else:
            st.image(colored_mask, caption="Land Cover Classification", use_container_width=True)

        # Stats
        st.markdown("### Land Cover Distribution")
        distribution = Analysis.get_class_distribution(mask)
        metric_cols = st.columns(len(distribution))
        for idx, (class_name, data) in enumerate(distribution.items()):
            with metric_cols[idx]:
                st.metric(class_name, f"{data['percentage']:.2f}%", f"{data['pixels']:,} px")

        # Pie chart
        fig, ax = plt.subplots(figsize=(8, 6))
        labels = list(distribution.keys())
        sizes = [distribution[l]['percentage'] for l in labels]
        colors = [Config.CLASS_COLORS_HEX[Config.CLASS_NAMES.index(l)] for l in labels]
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Detailed table
        df = pd.DataFrame([
            {'Land Cover Class': name, 'Pixels': f"{data['pixels']:,}", 'Percentage': f"{data['percentage']:.2f}%"}
            for name, data in distribution.items()
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)

        del mask, colored_mask, overlay, distribution, df, fig, ax
        if rgb is not None: del rgb
        gc.collect()

# ===================== URBAN CHANGE =====================
else:
    st.subheader(f"Urban Area Change Detection: {year1} → {year2}")

    if st.button("Analyze Urban Change", type="primary", use_container_width=True):
        mask1 = DataLoader.load_mask(year1)
        mask2 = DataLoader.load_mask(year2)
        if mask1 is None or mask2 is None:
            st.error(" Mask missing for one of the selected years.")
            st.stop()

        rgb1, rgb2 = None, None
        if show_satellite:
            img1 = DataLoader.load_sentinel_image(year1)
            img2 = DataLoader.load_sentinel_image(year2)
            if img1 is not None: rgb1 = Visualizer.get_rgb_from_sentinel(img1)
            if img2 is not None: rgb2 = Visualizer.get_rgb_from_sentinel(img2)
            del img1, img2
            gc.collect()

        st.success(" Data loaded!")

        # Compute change
        change_data = Analysis.compute_urban_change(mask1, mask2)
        change_map = change_data['change_map']

        # Change visualization
        H, W = change_map.shape
        change_vis = np.zeros((H, W, 3), dtype=np.uint8)
        change_vis[change_map == 1] = [255, 0, 0]  # Loss → now shown red
        change_vis[change_map == -1] = [0, 255, 0] # Expansion → now green

        st.markdown("### Results")
        if rgb1 is not None and rgb2 is not None:
            col1, col2 = st.columns(2)
            with col1: st.image(rgb1, caption=f"Sentinel-2 RGB {year1}", use_column_width=True)
            with col2: st.image(rgb2, caption=f"Sentinel-2 RGB {year2}", use_column_width=True)

        col_m1, col_m2 = st.columns(2)
        with col_m1: st.image(Visualizer.colorize_mask(mask1), caption=f"Land Cover {year1}", use_column_width=True)
        with col_m2: st.image(Visualizer.colorize_mask(mask2), caption=f"Land Cover {year2}", use_column_width=True)

        st.image(change_vis, caption="Urban Expansion | Urban Loss", use_column_width=True)
        if rgb2 is not None:
            overlay_change = cv2.addWeighted(rgb2, 0.7, change_vis, 0.3, 0)
            st.image(overlay_change, caption=f"Change Overlay on {year2}", use_column_width=True)

        # Metrics
        st.markdown("### Change Metrics")
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric(" Expansion", f"{change_data['expansion_pct']:.2f}%", f"+{change_data['expansion_pixels']:,} px")
        with metric_cols[1]:
            st.metric(" Loss", f"{change_data['loss_pct']:.2f}%", f"-{change_data['loss_pixels']:,} px")

        with metric_cols[2]:
            st.metric(" Urban Coverage", f"{year2}: {change_data['urban2_pct']:.2f}%", f"Δ {change_data['urban2_pct'] - change_data['urban1_pct']:+.2f}%")

        # Comparison table
        comparison_df = pd.DataFrame({
            'Metric': ['Urban Coverage %', 'Urban Pixels', 'Forest Coverage %'],
            year2: [
                f"{Analysis.get_class_distribution(mask1)['Urban']['percentage']:.2f}%",
                f"{Analysis.get_class_distribution(mask1)['Urban']['pixels']:,}",
                f"{Analysis.get_class_distribution(mask1)['Forest']['percentage']:.2f}%"
            ],
            year1: [
                f"{Analysis.get_class_distribution(mask2)['Urban']['percentage']:.2f}%",
                f"{Analysis.get_class_distribution(mask2)['Urban']['pixels']:,}",
                f"{Analysis.get_class_distribution(mask2)['Forest']['percentage']:.2f}%"
            ]
        })
        st.markdown("### Comparison")
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        del mask1, mask2, change_map, change_vis, change_data, comparison_df
        if rgb1 is not None: del rgb1
        if rgb2 is not None: del rgb2
        gc.collect()
