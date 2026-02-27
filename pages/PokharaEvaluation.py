# pages/04_PokharaEvaluation.py
# Pokhara — ConvLSTM Multi-Class Evaluation (with Background visible as Green)

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio
import cv2
import torch
from core.config import RegionConfig

st.set_page_config(page_title="Pokhara Evaluation", layout="wide")
st.title("Pokhara — ConvLSTM Multi-Class Evaluation")

# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 5  # model expects 5-year input sequence
ACTIVE_CLASSES = [1, 2, 3]
CLASSES = {0: "Background", 1: "Water", 2: "Forest", 3: "Built-up"}

# Define colors: Background is now Green (#2d6a4f)
CLASS_COLORS = {
    0: "#2d6a4f",  # Green for Background
    1: "#1d6fa4",  # Blue for Water
    2: "#2d6a4f",  # Dark Green for Forest
    3: "#e9c46a"  # Yellow for Built-up
}


# ─────────────────────────────────────────────
# MODEL LOADER (strict)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model(path):
    from models_pokhara import UrbanSprawlConvLSTM_Pokhara
    state = torch.load(path, map_location="cpu")
    model = UrbanSprawlConvLSTM_Pokhara(input_channels=3, hidden_channels=64)
    model.load_state_dict(state, strict=True)  # fail fast if mismatch
    model.eval()
    return model.to(DEVICE)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def load_mask(path, size_wh):
    """Return resized 2D label array (H, W). size_wh = (W, H)"""
    w, h = size_wh
    with rasterio.open(path) as src:
        raw = src.read(1).astype(np.uint8)
    return cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)


def mask_to_onehot(mask_2d, size_wh):
    """Return (C, H, W) one-hot for classes 1..3. size_wh = (W,H)."""
    w, h = size_wh
    r = cv2.resize(mask_2d, (w, h), interpolation=cv2.INTER_NEAREST)
    out = np.zeros((3, h, w), dtype=np.float32)
    for i, c in enumerate(ACTIVE_CLASSES):
        out[i] = (r == c).astype(np.float32)
    return out


def compute_metrics(gt, pred):
    from sklearn.metrics import confusion_matrix
    labels = [0, 1, 2, 3]
    cm = confusion_matrix(gt.ravel(), pred.ravel(), labels=labels)
    per_class = {}
    for c in [1, 2, 3]:
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        per_class[c] = {
            "iou": tp / (tp + fp + fn + 1e-8),
            "precision": tp / (tp + fp + 1e-8),
            "recall": tp / (tp + fn + 1e-8),
            "f1": 2 * tp / (2 * tp + fp + fn + 1e-8),
            "tp": int(tp), "fp": int(fp), "fn": int(fn)
        }
    accuracy = float(np.trace(cm) / (cm.sum() + 1e-8))
    macro_iou = np.mean([per_class[c]["iou"] for c in per_class])
    macro_f1 = np.mean([per_class[c]["f1"] for c in per_class])
    macro_recall = np.mean([per_class[c]["recall"] for c in per_class])
    return per_class, accuracy, macro_iou, macro_f1, macro_recall, cm


# ─────────────────────────────────────────────
# PLOTTING (UPDATED TO SHOW BACKGROUND)
# ─────────────────────────────────────────────

def plot_map_with_background(mask_2d, title):
    """
    Plots the map including Class 0 (Background) as Green.
    """
    # Define colormap: 0=Background(Green), 1=Water, 2=Forest, 3=Built-up
    # We use a list of colors corresponding to indices 0, 1, 2, 3
    colors = [CLASS_COLORS[0], CLASS_COLORS[1], CLASS_COLORS[2], CLASS_COLORS[3]]
    cmap = mcolors.ListedColormap(colors)

    # Boundaries for the 4 classes
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(mask_2d, cmap=cmap, norm=norm, interpolation="nearest")

    # Colorbar with labels for all 4 classes
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], fraction=0.046)
    cbar.ax.set_yticklabels(["Background", CLASSES[1], CLASSES[2], CLASSES[3]])

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# PAGE / REGION
# ─────────────────────────────────────────────
region: RegionConfig = st.session_state.get("region")
if region is None:
    st.warning("⬅️ Please select a region from the main page first.")
    st.stop()
if "pokhara" not in region.display_name.lower():
    st.warning(f"This page is for Pokhara. Current region: **{region.display_name}**.")
    st.stop()

# UI settings
resolution = st.selectbox("Evaluation Resolution", ["256x256 (fast)", "512x512 (detailed)"], index=0)
EVAL_SIZE = (256, 256) if "256" in resolution else (512, 512)  # (W,H)
available_years = sorted(set(region.image_years) & set(region.mask_years))
selected_years = st.multiselect("Select years (used as candidate input years)", available_years,
                                default=available_years)
if len(selected_years) == 0:
    st.warning("No years selected.")
    st.stop()

target_year = st.selectbox("Target prediction year (must have GT mask)",
                           [y for y in available_years if y > min(selected_years)])
if target_year is None:
    st.warning("Pick a target year that is later than selected input years.")
    st.stop()

# RUN evaluation
if st.button("▶ Run Pokhara Evaluation", type="primary"):
    # build input_years = those selected < target_year (chronological)
    input_years = sorted([y for y in selected_years if y < target_year])
    if len(input_years) == 0:
        st.error("No input years exist before the target year. Choose earlier input years or a later target.")
        st.stop()

    # load model
    try:
        model = load_model(str(region.convlstm_path))
    except Exception as e:
        st.error(f"Failed to load model strictly: {e}")
        st.stop()

    # Build sequence list of one-hot arrays (C,H,W)
    seq = []
    missing_files = []
    for y in input_years:
        try:
            raw = load_mask(str(region.masks[y]), EVAL_SIZE)  # (H,W)
            onehot = mask_to_onehot(raw, EVAL_SIZE)  # (3,H,W)
            seq.append(onehot)
        except Exception as e:
            missing_files.append((y, str(e)))
            # continue - we still might have some years

    if len(seq) == 0:
        st.error("No masks could be loaded for the chosen input years (files missing or unreadable). See errors:")
        for y, err in missing_files:
            st.write(f"- year {y}: {err}")
        st.stop()

    # If we have fewer than SEQ_LEN frames, pad by repeating the most recent mask (forward-fill)
    if len(seq) < SEQ_LEN:
        pad_count = SEQ_LEN - len(seq)
        st.warning(
            f"Only {len(seq)} valid input years available before {target_year}. Padding by repeating the MOST RECENT mask {pad_count} time(s) to reach {SEQ_LEN} frames. This affects predictions.")
        last = seq[-1]
        seq = seq + [last] * pad_count

    # Ensure sequence length == SEQ_LEN (or more); then keep last SEQ_LEN (most recent)
    if len(seq) > SEQ_LEN:
        seq = seq[-SEQ_LEN:]

    # Convert to tensor (T, C, H, W) -> (1, T, C, H, W)
    seq_np = np.stack(seq, axis=0)
    seq_t = torch.from_numpy(seq_np).float().unsqueeze(0).to(DEVICE)

    # Forward
    with torch.no_grad():
        logits = model(seq_t)  # expect (1, 3, H, W)
        soft = torch.softmax(logits, dim=1)[0].cpu().numpy()
    # soft shape (3,H,W)
    pred_ch = np.argmax(soft, axis=0)  # 0,1,2
    pred_lbl = (pred_ch + 1).astype(
        np.uint8)  # 1,2,3 (Background remains 0 implicitly in pred_lbl if we initialized with zeros, but here we only have 1,2,3)

    # Load GT
    try:
        gt_eval = load_mask(str(region.masks[target_year]), EVAL_SIZE)
    except Exception as e:
        st.error(f"Could not load GT mask for target {target_year}: {e}")
        st.stop()

    # Metrics (ignore background class 0 in per-class report)
    per_class, acc, miou, mf1, mrec, cm = compute_metrics(gt_eval, pred_lbl)

    # Dashboard metrics
    st.markdown("---")
    st.subheader(f"📊 Results — Predicting {target_year}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Accuracy", f"{acc:.4f}")
    c2.metric("Macro IoU (classes 1-3)", f"{miou:.4f}")
    c3.metric("Macro F1  (classes 1-3)", f"{mf1:.4f}")
    c4.metric("Macro Recall (classes 1-3)", f"{mrec:.4f}")

    rows = []
    for c in [1, 2, 3]:
        m = per_class[c]
        gt_px = m["tp"] + m["fn"]
        rows.append({
            "Class": f"{c} — {CLASSES[c]}",
            "IoU": m["iou"], "Precision": m["precision"],
            "Recall": m["recall"], "F1": m["f1"],
            "GT pixels": gt_px,
            "GT %": f"{100 * gt_px / (gt_eval.size + 1e-8):.1f}%"
        })
    st.dataframe(pd.DataFrame(rows).style.format(
        {"IoU": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}", "F1": "{:.4f}", "GT pixels": "{:,}"}),
                 use_container_width=True)

    # Spatial maps (WITH BACKGROUND) - Error Map Removed
    st.markdown("---")
    st.subheader("🗺️ Spatial Comparison (Background Visible)")
    m1, m2 = st.columns(2)  # Changed to 2 columns
    with m1:
        st.pyplot(plot_map_with_background(gt_eval, f"Ground Truth — {target_year}"))
    with m2:
        st.pyplot(plot_map_with_background(pred_lbl, f"Predicted — {target_year}"))



