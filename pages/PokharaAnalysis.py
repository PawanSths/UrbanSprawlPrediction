#!/usr/bin/env python3
import streamlit as st
import numpy as np
import rasterio
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import gc

st.set_page_config(
    page_title="Pokhara Land Cover Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { 
        font-size: 2.4rem; 
        font-weight: 700; 
        color: #2ecc71; 
        text-align: center; 
        margin-bottom: 1rem;
    }
    .stButton>button { 
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%); 
        color: white; 
        font-weight: bold; 
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

# ========================================
# FIXED DATA PATH
# ========================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "pokhara" / "PokharaPredicted"

if not DATA_DIR.exists():
    st.error(f"Data folder not found: {DATA_DIR}")
    st.stop()

# ========================================
# CONFIG
# ========================================
class Config:
    CLASS_NAMES = ['Background', 'Water', 'Vegetation', 'Urban']
    CLASS_COLORS = {
        0: [139, 90, 43],   # Background - Brown
        1: [51, 136, 255],  # Water - Blue
        2: [34, 204, 34],   # Vegetation - Green
        3: [255, 68, 68]    # Urban - Red
    }
    CLASS_COLORS_HEX = {
        0: "#8B5A2B",
        1: "#3388FF",
        2: "#22CC22",
        3: "#FF4444"
    }
    URBAN_CLASS = 3
    PIXEL_AREA_M2 = 10 * 10  # 10m x 10m pixels

# ========================================
# DATA LOADER
# ========================================
class DataLoader:
    @staticmethod
    def load_mask(year):
        path = DATA_DIR / f"mask_{year}.tif"
        if not path.exists():
            return None
        with rasterio.open(path) as src:
            mask = src.read(1).astype(np.uint8)
        mask = np.clip(mask, 0, 3)
        return mask

    @staticmethod
    def load_sentinel_image(year):
        # try several naming variants
        for filename in [f"{year}IMAGE.TIF", f"{year}image.tif", f"{year}Image.tif", f"{year}.tif"]:
            path = DATA_DIR / filename
            if path.exists():
                with rasterio.open(path) as src:
                    img = src.read()
                    img = np.moveaxis(img, 0, -1)
                    return img.astype(np.float32)
        return None

    @staticmethod
    def get_available_years():
        years = []
        for y in range(2015, 2036):
            if (DATA_DIR / f"mask_{y}.tif").exists():
                years.append(y)
        if not years:
            st.warning("No mask files found. Checked years 2015-2035.")
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
        if img is None or img.shape[-1] < 3:
            return None
        rgb = img[..., :3].astype(np.float32)
        # contrast stretch per channel
        p2, p98 = np.percentile(rgb, (2, 98))
        # Avoid divide by zero
        denom = (p98 - p2) if (p98 - p2) != 0 else 1.0
        rgb = np.clip((rgb - p2) / denom, 0, 1)
        rgb = (rgb * 255).astype(np.uint8)
        return rgb

    @staticmethod
    def create_overlay(rgb, mask, alpha=0.6):
        colored_mask = Visualizer.colorize_mask(mask)
        if rgb is None:
            return colored_mask
        if rgb.shape[:2] != mask.shape:
            colored_mask = cv2.resize(colored_mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        # ensure same dtype
        rgb_u8 = (rgb).astype(np.uint8)
        return cv2.addWeighted(rgb_u8, 1 - alpha, colored_mask, alpha, 0)

    @staticmethod
    def create_urban_change_map(mask1, mask2):
        urban1 = (mask1 == Config.URBAN_CLASS).astype(np.int8)
        urban2 = (mask2 == Config.URBAN_CLASS).astype(np.int8)
        change = urban2 - urban1
        h, w = change.shape
        change_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        change_rgb[change == 1] = [0, 255, 0]      # Green: new urban
        change_rgb[change == -1] = [255, 0, 0]     # Red: lost urban
        change_rgb[(change == 0) & (urban2 == 1)] = [255, 200, 200]  # Light red: stable urban
        change_rgb[(change == 0) & (urban2 == 0)] = [50, 50, 50]     # Dark gray: non-urban
        return change_rgb

# ========================================
# ANALYSIS
# ========================================
class Analysis:
    @staticmethod
    def get_class_distribution(mask):
        total = mask.size
        results = {}
        for cls_id, name in enumerate(Config.CLASS_NAMES):
            count = int(np.sum(mask == cls_id))
            results[name] = {
                'pixels': count,
                'percentage': (count / total) * 100 if total > 0 else 0.0,
                'area_km2': count * (Config.PIXEL_AREA_M2 / 1_000_000)
            }
        return results

    @staticmethod
    def compute_urban_change(mask1, mask2):
        urban1 = (mask1 == Config.URBAN_CLASS).astype(np.uint8)
        urban2 = (mask2 == Config.URBAN_CLASS).astype(np.uint8)
        change_map = urban2.astype(np.int8) - urban1.astype(np.int8)
        total = mask1.size
        expansion_pixels = int(np.sum(change_map == 1))
        loss_pixels = int(np.sum(change_map == -1))
        expansion_km2 = expansion_pixels * (Config.PIXEL_AREA_M2 / 1_000_000)
        loss_km2 = loss_pixels * (Config.PIXEL_AREA_M2 / 1_000_000)
        urban1_total = int(np.sum(urban1))
        urban2_total = int(np.sum(urban2))
        return {
            'change_map': change_map,
            'expansion_pixels': expansion_pixels,
            'loss_pixels': loss_pixels,
            'expansion_km2': expansion_km2,
            'loss_km2': loss_km2,
            'net_change_km2': expansion_km2 - loss_km2,
            'urban1_percentage': (urban1_total / total) * 100 if total > 0 else 0.0,
            'urban2_percentage': (urban2_total / total) * 100 if total > 0 else 0.0,
            'urban1_km2': urban1_total * (Config.PIXEL_AREA_M2 / 1_000_000),
            'urban2_km2': urban2_total * (Config.PIXEL_AREA_M2 / 1_000_000),
            'urban1_pixels': urban1_total,
            'urban2_pixels': urban2_total
        }

# ========================================
# INITIALIZATION
# ========================================
available_years = DataLoader.get_available_years()
if not available_years:
    st.error("No mask files found in data/pokhara")
    st.stop()

# ========================================
# SIDEBAR
# ========================================
with st.sidebar:


    analysis_mode = st.radio(
        " Select Analysis",
        ["Single Year", "Urban Change (2 Years)"],
        help="Choose analysis type"
    )

    st.markdown("---")

    if analysis_mode == "Single Year":
        selected_year = st.selectbox(" Select Year", available_years, index=len(available_years) - 1)

    elif analysis_mode == "Urban Change (2 Years)":
        col1, col2 = st.columns(2)
        with col1:
            year1 = st.selectbox(" Year 1", available_years, index=0)
        with col2:
            year2 = st.selectbox(" Year 2", available_years, index=len(available_years) - 1)

    st.markdown("---")
    overlay_alpha = st.slider(" Overlay Opacity", 0.0, 1.0, 0.6, 0.05)
    show_satellite = st.checkbox(" Show Satellite Imagery", value=True)

# ========================================
# MAIN PAGE
# ========================================
st.markdown("<h1 class='main-header'> Pokhara Land Cover Analyzer</h1>", unsafe_allow_html=True)

st.markdown("---")

# SINGLE YEAR MODE
if analysis_mode == "Single Year":
    st.subheader(f" Visualization for {selected_year}")

    if st.button(" Load Data"):
        with st.spinner(f"Loading mask for {selected_year}..."):
            mask = DataLoader.load_mask(selected_year)
            if mask is None:
                st.error(f"No mask for {selected_year}")
                st.stop()

        rgb = None
        if show_satellite:
            with st.spinner(f"Loading Sentinel-2 for {selected_year}..."):
                img = DataLoader.load_sentinel_image(selected_year)
                if img is not None:
                    rgb = Visualizer.get_rgb_from_sentinel(img)
                else:
                    st.warning(f"No satellite image found for {selected_year}")

        st.info(f"Mask dimensions: {mask.shape[1]} × {mask.shape[0]} px")

        colored_mask = Visualizer.colorize_mask(mask)
        overlay = Visualizer.create_overlay(rgb, mask, overlay_alpha) if rgb is not None else colored_mask

        st.markdown("### Visualization")
        if rgb is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(rgb, caption=f"Sentinel-2 RGB ({selected_year})")
            with col2:
                st.image(colored_mask, caption="Land Cover Classification")
            with col3:
                st.image(overlay, caption=f"Overlay (α={overlay_alpha})")
        else:
            st.image(colored_mask, caption="Land Cover Classification")

        # Stats (include background in computations but exclude it from pie/table)
        distribution = Analysis.get_class_distribution(mask)

        metric_cols = st.columns(4)
        # show summary metrics: Water, Vegetation, Urban only for clarity
        metric_cols[0].metric("Water", f"{distribution['Water']['percentage']:.2f}%", f"{distribution['Water']['area_km2']:.2f} km²")
        metric_cols[1].metric("Vegetation", f"{distribution['Vegetation']['percentage']:.2f}%", f"{distribution['Vegetation']['area_km2']:.2f} km²")
        metric_cols[2].metric("Urban", f"{distribution['Urban']['percentage']:.2f}%", f"{distribution['Urban']['area_km2']:.2f} km²")
        metric_cols[3].metric("Background", f"{distribution['Background']['percentage']:.2f}%", f"{distribution['Background']['area_km2']:.2f} km²")

        # Pie chart (background removed)
        st.markdown("### Land Cover Distribution (background excluded)")
        labels = ['Water', 'Vegetation', 'Urban']
        sizes = [distribution[l]['percentage'] for l in labels]
        # Normalize to sum of active classes so pie shows share among active classes only
        total_active = sum(sizes) if sum(sizes) > 0 else 1.0
        sizes_norm = [s * 100.0 / total_active for s in sizes]
        colors = [Config.CLASS_COLORS_HEX[1], Config.CLASS_COLORS_HEX[2], Config.CLASS_COLORS_HEX[3]]

        fig, ax = plt.subplots(figsize=(8, 6))
        wedges, texts, autotexts = ax.pie(sizes_norm, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title(f"Land Cover Distribution - {selected_year}", fontsize=14, fontweight='bold')
        ax.axis('equal')
        st.pyplot(fig)
        plt.close(fig)

        # Table (background row removed)
        st.markdown("### Detailed Statistics (background excluded)")
        df = pd.DataFrame([
            {
                'Class': name,
                'Pixels': f"{data['pixels']:,}",
                'Percentage': f"{data['percentage']:.2f}%",
                'Area (km²)': f"{data['area_km2']:.2f}"
            }
            for name, data in distribution.items()
            if name != "Background"
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)

        # cleanup
        del mask, colored_mask, overlay, distribution, df, wedges, texts, autotexts
        if rgb is not None:
            del rgb
        gc.collect()

# URBAN CHANGE MODE
elif analysis_mode == "Urban Change (2 Years)":
    st.subheader(f" Urban Change Detection: {year1} → {year2}")

    if st.button(" Analyze Change"):
        with st.spinner(f"Loading {year1}..."):
            mask1 = DataLoader.load_mask(year1)
            if mask1 is None:
                st.error(f"No mask for {year1}")
                st.stop()

        rgb1 = None
        if show_satellite:
            img1 = DataLoader.load_sentinel_image(year1)
            if img1 is not None:
                rgb1 = Visualizer.get_rgb_from_sentinel(img1)

        with st.spinner(f"Loading {year2}..."):
            mask2 = DataLoader.load_mask(year2)
            if mask2 is None:
                st.error(f"No mask for {year2}")
                st.stop()

        rgb2 = None
        if show_satellite:
            img2 = DataLoader.load_sentinel_image(year2)
            if img2 is not None:
                rgb2 = Visualizer.get_rgb_from_sentinel(img2)

        change_data = Analysis.compute_urban_change(mask1, mask2)
        change_vis = Visualizer.create_urban_change_map(mask1, mask2)

        st.markdown("### Results")
        col1, col2 = st.columns(2)
        with col1:
            st.image(Visualizer.colorize_mask(mask1), caption=f"Land Cover {year1}")
        with col2:
            st.image(Visualizer.colorize_mask(mask2), caption=f"Land Cover {year2}")

        st.image(change_vis, caption="🟢 New Urban | 🔴 Lost Urban")

        if rgb2 is not None:
            overlay_change = cv2.addWeighted(rgb2, 0.7, change_vis, 0.3, 0)
            st.image(overlay_change, caption=f"Change Overlay on {year2}")

        # Change metrics
        st.markdown("### Change Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🟢 Expansion", f"{change_data['expansion_km2']:.2f} km²")
        c2.metric("🔴 Loss", f"{change_data['loss_km2']:.2f} km²")
        c3.metric("Net Change", f"{change_data['net_change_km2']:+.2f} km²")
        c4.metric("Urban Coverage", f"{year2}: {change_data['urban2_percentage']:.2f}%")

        # Comparison table (urban stats)
        st.markdown("### Comparison")
        df = pd.DataFrame({
            'Metric': ['Urban %', 'Urban km²', 'Urban Pixels'],
            year1: [
                f"{change_data['urban1_percentage']:.2f}%",
                f"{change_data['urban1_km2']:.2f}",
                f"{change_data['urban1_pixels']:,}"
            ],
            year2: [
                f"{change_data['urban2_percentage']:.2f}%",
                f"{change_data['urban2_km2']:.2f}",
                f"{change_data['urban2_pixels']:,}"
            ]
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

        del mask1, mask2, change_data, change_vis, df
        if rgb1 is not None:
            del rgb1
        if rgb2 is not None:
            del rgb2
        gc.collect()