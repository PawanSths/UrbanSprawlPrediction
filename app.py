"""
app.py — UrbanScope Pro: Multi-Region Urban Expansion Prediction
Entry point with region/model selection.
"""

import streamlit as st

st.set_page_config(
    page_title="UrbanScope Pro – Urban Sprawl Prediction",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

from core.config import REGIONS

# ══════════════════════════════════════════════════
# SIDEBAR — Region & Capability Detection
# ══════════════════════════════════════════════════

st.sidebar.title("🌍 UrbanScope Pro")
st.sidebar.markdown("---")

if not REGIONS:
    st.error(
        "❌ No regions found!\n\n"
        "Make sure your `data/` folder has subfolders like:\n"
        "```\ndata/kathmandu/\ndata/hyderabad/\ndata/pokhara/\n```"
    )
    st.stop()

# Build display → key mapping
region_options = {r.display_name: key for key, r in REGIONS.items()}
selected_display = st.sidebar.selectbox(
    "📍 Select Region",
    list(region_options.keys()),
    help="Regions are auto-detected from data/ subfolders",
)
region_key = region_options[selected_display]
region = REGIONS[region_key]

# Store in session_state so pages can access it
st.session_state["region_key"] = region_key
st.session_state["region"] = region

# Show capabilities
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Region Capabilities")
st.sidebar.markdown(f"**📸 Images:** {len(region.images)} years")
if region.image_years:
    st.sidebar.markdown(f"  ↳ {min(region.image_years)} – {max(region.image_years)}")
st.sidebar.markdown(f"**🎯 Masks:** {len(region.masks)} years")
st.sidebar.markdown(f"**🧠 U-Net:** {'✅ Available' if region.has_unet else '❌ Not available'}")
st.sidebar.markdown(f"**🔮 ConvLSTM:** {'✅ Available' if region.has_convlstm else '❌ Not available'}")
st.sidebar.markdown(f"**🔗 Pipeline:** {'✅ Available' if region.has_pipeline else '❌ Not available'}")

st.sidebar.markdown("---")
st.sidebar.markdown("### Available Modes")
for mode in region.available_modes:
    st.sidebar.markdown(f"  • {mode}")

if not region.available_modes:
    st.sidebar.warning("No trained models found for this region.")

# ══════════════════════════════════════════════════
# MAIN — Home Dashboard
# ══════════════════════════════════════════════════

st.title(f"🏙️ UrbanScope Pro — {region.display_name}")
st.caption("Multi-region urban expansion prediction using U-Net & ConvLSTM")

# Overview cards for ALL regions
st.markdown("### 🗺️ All Discovered Regions")

cols = st.columns(len(REGIONS))
for i, (key, r) in enumerate(REGIONS.items()):
    with cols[i]:
        # Highlight active region
        if key == region_key:
            st.markdown(f"#### 📍 **{r.display_name}** ← Active")
        else:
            st.markdown(f"#### {r.display_name}")

        st.metric("Images", len(r.images))
        st.metric("Masks", len(r.masks))
        st.write(f"U-Net: {'✅' if r.has_unet else '❌'} | ConvLSTM: {'✅' if r.has_convlstm else '❌'}")

st.markdown("---")

# Instructions
st.markdown(
    f"""
    ### 🚀 Getting Started with {region.display_name}

    Use the **sidebar pages** to navigate:

    | Page | What it does | Available? |
    |------|-------------|------------|
    | **🔬 U-Net Expansion** | Single-year urban segmentation with interactive map | {'✅' if region.has_unet else '❌ No U-Net model'} |
    | **🔮 ConvLSTM Future** | Multi-year temporal prediction | {'✅' if region.has_convlstm else '❌ No ConvLSTM model'} |
    | **📊 Evaluation** | Batch metrics & confusion matrix | {'✅' if region.has_unet or region.has_convlstm else '❌'} |

    **How it works:**
    1. Select your region above
    2. Navigate to a page
    3. Draw a rectangle on the map
    4. Get predictions!
    """
)