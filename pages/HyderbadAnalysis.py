#!/usr/bin/env python3


import streamlit as st
import numpy as np
import rasterio
import cv2
import os
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import gc

# ========================================
# PAGE CONFIG
# ========================================

st.set_page_config(
    page_title="Hyderabad Land Cover Analyzer",
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
    CLASS_NAMES = ['Urban', 'Forest', 'Water', 'Barren']

    CLASS_COLORS = {
    0:[128, 0, 0],
    1:[34, 139, 34],
    2:[0, 191, 255],
    3:[210, 180, 140]
    }

    CLASS_COLORS_HEX = {
    0:"#800000", 1:"#228B22", 2:"#00BFFF", 3:"#D2B48C"
    }

    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data" / "hyderabad"


# ========================================
# DATA LOADER
# ========================================

class DataLoader:
    """Load satellite data and masks efficiently"""

    @staticmethod
    def load_sentinel_image(year):
        """Load single Sentinel-2 image"""
        path = Path(Config.DATA_DIR) / f"{year}image.tif"
        if not path.exists():
            return None

        with rasterio.open(path) as src:
            img = src.read()  # (bands, H, W)
            img = np.moveaxis(img, 0, -1)  # → (H, W, bands)
            return img.astype(np.float32)

    @staticmethod
    def load_mask(year):
        """Load pre-computed mask"""
        path = Path(Config.DATA_DIR) / f"mask_{year}.tif"
        if not path.exists():
            return None

        with rasterio.open(path) as src:
            mask = src.read(1)  # Read first band

            # Ensure mask has correct class indices (0-3)
            if mask.max() > 3:
                # If mask values are 255-based, normalize to 0-3
                mask = mask // 64  # 0-255 → 0-3

            return mask.astype(np.uint8)

    @staticmethod
    def get_available_years():
        """Get list of available years (based on masks)"""
        data_dir = Path(Config.DATA_DIR)
        years = []

        for year in range(2017, 2027):
            mask_path = data_dir / f"mask_{year}.tif"
            if mask_path.exists():
                years.append(year)

        return sorted(years)


# ========================================
# VISUALIZATION
# ========================================

class Visualizer:
    """Image processing and display"""

    @staticmethod
    def colorize_mask(mask):
        """Convert class indices to RGB colors"""
        h, w = mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for cls, color in Config.CLASS_COLORS.items():
            rgb[mask == cls] = color
        return rgb

    @staticmethod
    def get_rgb_from_sentinel(img):
        """Extract RGB from Sentinel bands 4,3,2"""
        if img is None:
            return None

        if img.shape[-1] >= 4:
            rgb = img[..., [2, 1, 0]]  # B04, B03, B02 → RGB
            rgb = np.nan_to_num(rgb, nan=0.0)  # convert NaN to 0
            rgb = np.clip(rgb / 3000.0 * 255, 0, 255).astype(np.uint8)
            return rgb
        return None

    @staticmethod
    def create_overlay(rgb, mask, alpha=0.6):
        """Create transparent mask overlay"""
        if rgb is None:
            # If no RGB, just return colorized mask
            return Visualizer.colorize_mask(mask)

        colored_mask = Visualizer.colorize_mask(mask)
        return cv2.addWeighted(rgb, 1 - alpha, colored_mask, alpha, 0)


# ========================================
# ANALYSIS
# ========================================

class Analysis:
    """Compute statistics"""

    @staticmethod
    def get_class_distribution(mask):
        """Get per-class pixel counts and percentages"""
        unique, counts = np.unique(mask, return_counts=True)
        total = mask.size

        results = {}
        for cls_idx, count in zip(unique, counts):
            if cls_idx < len(Config.CLASS_NAMES):
                results[Config.CLASS_NAMES[cls_idx]] = {
                    'pixels': int(count),
                    'percentage': (count / total) * 100
                }

        # Include zero-count classes
        for name in Config.CLASS_NAMES:
            if name not in results:
                results[name] = {'pixels': 0, 'percentage': 0.0}

        return results

    @staticmethod
    def compute_urban_change(mask1, mask2):
        """Compute urban area changes"""
        urban1 = (mask1 == 0).astype(np.uint8)
        urban2 = (mask2 == 0).astype(np.uint8)
        change_map = urban2.astype(np.int8) - urban1.astype(np.int8)

        total = mask1.size
        expansion = np.sum(change_map == 1)
        loss = np.sum(change_map == -1)

        return {
            'change_map': change_map,
            'expansion_pixels': int(expansion),
            'loss_pixels': int(loss),
            'expansion_pct': (expansion / total) * 100,
            'loss_pct': (loss / total) * 100,
            'urban1_pct': (np.sum(urban1) / total) * 100,
            'urban2_pct': (np.sum(urban2) / total) * 100,
            'net_change_pct': ((np.sum(urban2) - np.sum(urban1)) / total) * 100
        }


# ========================================
# INITIALIZATION
# ========================================

available_years = DataLoader.get_available_years()

if not available_years:
    st.error(" No mask files found in data directory. Expected format: mask_YYYY.tif")
    st.stop()

# ========================================
# SIDEBAR
# ========================================

with st.sidebar:
    st.markdown("<h2 style='color:#667eea;'>⚙ Configuration</h2>", unsafe_allow_html=True)

    analysis_mode = st.radio(
        " Select Analysis",
        ["Single Year", "Urban Change (2 Years)"],
        help="Choose single year visualization or compare two years"
    )

    st.markdown("---")

    if analysis_mode == "Single Year":
        selected_year = st.selectbox(
            "Select Year",
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
    overlay_alpha = st.slider(
        " Overlay Opacity",
        0.0, 1.0, 0.6, 0.05,
        help="Higher = more visible mask"
    )

    show_satellite = st.checkbox("Show Satellite Imagery", value=True, help="Display Sentinel-2 RGB if available")

# ========================================
# MAIN PAGE
# ========================================

st.markdown("<h1 class='main-header'> Hyderabad Land Cover Analyzer</h1>", unsafe_allow_html=True)

st.markdown("---")

# ========================================
# SINGLE YEAR MODE
# ========================================

if analysis_mode == "Single Year":

    st.subheader(f"Visualization for {selected_year}")

    if st.button(" Load Data", type="primary", width="stretch"):

        # Load mask
        with st.spinner(f"Loading mask for {selected_year}..."):
            mask = DataLoader.load_mask(selected_year)
            if mask is None:
                st.error(f"No mask found for year {selected_year}")
                st.stop()

        # Load satellite image if requested
        rgb = None
        if show_satellite:
            with st.spinner(f"Loading Sentinel-2 image for {selected_year}..."):
                img = DataLoader.load_sentinel_image(selected_year)
                if img is not None:
                    rgb = Visualizer.get_rgb_from_sentinel(img)
                    del img
                    gc.collect()

        mask_h, mask_w = mask.shape
        st.info(f"Mask dimensions: {mask_w} × {mask_h} px")

        # Get visualizations
        colored_mask = Visualizer.colorize_mask(mask)

        overlay = Visualizer.create_overlay(rgb, mask, overlay_alpha) if rgb is not None else colored_mask

        # Display results
        st.markdown("### Visualization")

        if rgb is not None:
            col_rgb, col_mask, col_overlay = st.columns(3)

            with col_rgb:
                st.image(rgb, caption=f"Sentinel-2 RGB ({selected_year})", width='stretch')

            with col_mask:
                st.image(colored_mask, caption="Land Cover Classification", width='stretch')

            with col_overlay:
                st.image(overlay, caption="Overlay", width="stretch")
        else:
            st.image(colored_mask, caption="Land Cover Classification", width='stretch')

        # Statistics
        st.markdown("### Land Cover Distribution")

        distribution = Analysis.get_class_distribution(mask)

        # Metrics
        metric_cols = st.columns(4)
        for idx, (class_name, data) in enumerate(distribution.items()):
            with metric_cols[idx]:
                st.metric(
                    class_name,
                    f"{data['percentage']:.2f}%",
                    f"{data['pixels']:,} px"
                )

        # Pie chart
        fig, ax = plt.subplots(figsize=(8, 6))
        labels = list(distribution.keys())
        sizes = [distribution[l]['percentage'] for l in labels]
        colors = [Config.CLASS_COLORS_HEX[Config.CLASS_NAMES.index(l)] for l in labels]

        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct='%1.1f%%',
            colors=colors, startangle=90
        )
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        st.pyplot(fig, width='stretch')
        plt.close(fig)

        # Data table
        st.markdown("### Detailed Statistics")

        df = pd.DataFrame([
            {
                'Land Cover Class': name,
                'Pixels': f"{data['pixels']:,}",
                'Percentage': f"{data['percentage']:.2f}%"
            }
            for name, data in distribution.items()
        ])

        st.dataframe(df, width='stretch', hide_index=True)

        # Export
        st.markdown("###  Export Results")

        csv_data = df.to_csv(index=False)
        st.download_button(
            "Download Statistics (CSV)",
            csv_data,
            file_name=f"land_cover_stats_{selected_year}.csv",
            mime="text/csv"
        )

        # Clean up memory
        del mask, colored_mask, overlay, distribution, df, fig, ax
        if rgb is not None:
            del rgb
        gc.collect()

# ========================================
# URBAN CHANGE MODE
# ========================================

else:  # Urban Change (2 Years)

    st.subheader(f"Urban Area Change Detection: {year1} → {year2}")

    if st.button("Analyze Urban Change", type="primary", width='stretch'):

        # Load first mask
        with st.spinner(f"Loading mask for {year1}..."):
            mask1 = DataLoader.load_mask(year1)
            if mask1 is None:
                st.error(f" No mask found for year {year1}")
                st.stop()

        # Load satellite image 1 if requested
        rgb1 = None
        if show_satellite:
            img1 = DataLoader.load_sentinel_image(year1)
            if img1 is not None:
                rgb1 = Visualizer.get_rgb_from_sentinel(img1)
                del img1
                gc.collect()

        # Load second mask
        with st.spinner(f" Loading mask for {year2}..."):
            mask2 = DataLoader.load_mask(year2)
            if mask2 is None:
                st.error(f"No mask found for year {year2}")
                st.stop()

        # Load satellite image 2 if requested
        rgb2 = None
        if show_satellite:
            img2 = DataLoader.load_sentinel_image(year2)
            if img2 is not None:
                rgb2 = Visualizer.get_rgb_from_sentinel(img2)
                del img2
                gc.collect()

        st.success(" Data loaded!")

        # Compute change
        change_data = Analysis.compute_urban_change(mask1, mask2)
        change_map = change_data['change_map']

        # Create change visualization
        H, W = change_map.shape
        change_vis = np.zeros((H, W, 3), dtype=np.uint8)
        change_vis[change_map == 1] = [0, 255, 0]  # Green: expansion
        change_vis[change_map == -1] = [255, 0, 0]  # Red: loss

        # Display visualizations
        st.markdown("###  Results")

        if rgb1 is not None and rgb2 is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.image(rgb1, caption=f"Sentinel-2 RGB {year1}", width='stretch')
            with col2:
                st.image(rgb2, caption=f"Sentinel-2 RGB {year2}",width='stretch')

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.image(Visualizer.colorize_mask(mask1), caption=f"Land Cover {year1}", width='stretch')
        with col_m2:
            st.image(Visualizer.colorize_mask(mask2), caption=f"Land Cover {year2}",width='stretch')

        #st.image(change_vis, caption=" Urban Expansion |  Urban Loss", width='stretch')

        if rgb2 is not None:
            overlay_change = cv2.addWeighted(rgb2, 0.7, change_vis, 0.3, 0)
            st.image(overlay_change, caption=f"Change Overlay on {year2}", width='stretch')

        # Metrics
        st.markdown("###  Change Metrics")

        metric_cols = st.columns(4)

        with metric_cols[0]:
            st.metric(
                " Expansion",
                f"{change_data['expansion_pct']:.2f}%",
                f"+{change_data['expansion_pixels']:,} px"
            )

        with metric_cols[1]:
            st.metric(
                " Loss",
                f"{change_data['loss_pct']:.2f}%",
                f"-{change_data['loss_pixels']:,} px"
            )

        with metric_cols[2]:
            st.metric(
                " Net Change",
                f"{change_data['net_change_pct']:+.2f}%",
                f"{change_data['expansion_pixels'] - change_data['loss_pixels']:+,} px"
            )

        with metric_cols[3]:
            st.metric(
                " Urban Coverage",
                f"{year2}: {change_data['urban2_pct']:.2f}%",
                f"Δ {change_data['urban2_pct'] - change_data['urban1_pct']:+.2f}%"
            )

        # Comparison table
        st.markdown("###  Comparison")

        comparison_df = pd.DataFrame({
            'Metric': [
                'Urban Coverage %',
                'Urban Pixels',
                'Forest Coverage %',
                'Water Coverage %',
                'Barren Coverage %'
            ],
            year1: [
                f"{Analysis.get_class_distribution(mask1)['Urban']['percentage']:.2f}%",
                f"{Analysis.get_class_distribution(mask1)['Urban']['pixels']:,}",
                f"{Analysis.get_class_distribution(mask1)['Forest']['percentage']:.2f}%",
                f"{Analysis.get_class_distribution(mask1)['Water']['percentage']:.2f}%",
                f"{Analysis.get_class_distribution(mask1)['Barren']['percentage']:.2f}%"
            ],
            year2: [
                f"{Analysis.get_class_distribution(mask2)['Urban']['percentage']:.2f}%",
                f"{Analysis.get_class_distribution(mask2)['Urban']['pixels']:,}",
                f"{Analysis.get_class_distribution(mask2)['Forest']['percentage']:.2f}%",
                f"{Analysis.get_class_distribution(mask2)['Water']['percentage']:.2f}%",
                f"{Analysis.get_class_distribution(mask2)['Barren']['percentage']:.2f}%"
            ]
        })

        st.dataframe(comparison_df, width='stretch'
                     , hide_index=True)

        # Export comparison
        csv_data = comparison_df.to_csv(index=False)
        st.download_button(
            " Download Comparison (CSV)",
            csv_data,
            file_name=f"urban_change_{year1}_to_{year2}.csv",
            mime="text/csv"
        )

        # Clean up memory
        del mask1, mask2, change_map, change_vis, change_data, comparison_df
        if rgb1 is not None:
            del rgb1
        if rgb2 is not None:
            del rgb2
        gc.collect()


