import streamlit as st
import os

st.set_page_config(
    page_title="UrbanScope Pro – Urban Sprawl Prediction",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

from core.config import REGIONS


st.sidebar.markdown("---")

if not REGIONS:
    st.sidebar.error(" No regions found! Check your data/ folder.")
    st.stop()

# Build display → key mapping
region_options = {r.display_name: key for key, r in REGIONS.items()}

# Ensure Kathmandu appears first
region_display_names = list(region_options.keys())
if "Kathmandu" in region_display_names:
    region_display_names.remove("Kathmandu")
region_display_names = ["Kathmandu"] + region_display_names

selected_display = st.sidebar.selectbox(
    " Select Region",
    region_display_names,
    help="Regions are auto-detected from data/ subfolders",
)

region_key = region_options[selected_display]
region = REGIONS[region_key]

# Store in session_state
st.session_state["region_key"] = region_key
st.session_state["region"] = region

# Show region info
st.sidebar.markdown("---")
st.sidebar.markdown("###  Region Info")
st.sidebar.write(f"**Region:** {region.display_name}")
st.sidebar.write(f"**Images:** {len(region.images)}")
st.sidebar.write(f"**Masks:** {len(region.masks)}")

# ══════════════════════════════════════════════════
# MAIN — Home Dashboard
# ══════════════════════════════════════════════════


st.title("Urban Sprawl Prediction using U-Net and ConvLSTM")
st.caption("Multi-region urban expansion prediction")

st.markdown("---")

# Bigger Currently Active Region
st.markdown("###  Currently Active Region")
st.markdown(
    f"""
    <h1 style='color:#2E86C1; font-weight:700; margin-top:0;'>
        {region.display_name}
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# Presented By Section (Clean Layout)
st.markdown("###  Presented By")

col1, col2, col3 = st.columns(3)

col1.markdown("""
**Pawan Shrestha**  
021-353
""")

col2.markdown("""
**Prashows Amatya**  
021-359
""")

col3.markdown("""
**Sananda Satyal**  
021-376
""")
