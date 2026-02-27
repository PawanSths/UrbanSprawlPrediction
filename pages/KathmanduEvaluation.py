# pages/03_Evaluation.py
# Evaluation dashboard — U-Net (.h5 Keras) + ConvLSTM (.pth PyTorch)
# Works for Kathmandu, Hyderabad, and other regions
# Fixed: multi-class U-Net handling + region-aware threshold hint

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
import cv2



from core.config import RegionConfig
from core.loaders import load_unet
from core.metrics import compute_metrics
from models import UrbanSprawlConvLSTM
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
st.title(" Model Evaluation Dashboard")

# ─── Region & Model Selection ───────────────────────────────────────────────
region: RegionConfig = st.session_state.get("region")
if region is None:
    st.warning(" Please select a region from the main page first.")
    st.stop()

st.write(f"**Current region:** {region.display_name}")

model_type = st.radio("Model to evaluate", ["U-Net", "ConvLSTM"], horizontal=True)

# Resolution selection
resolution_choice = st.selectbox(
    "Evaluation Resolution",
    [
        "256×256 (fast, lower detail, low memory)",
        "512×512 (accurate, higher detail, more memory)"
    ],
    index=1
)

if "256" in resolution_choice:
    EVAL_SIZE = (256, 256)
    res_name = "256×256"
else:
    EVAL_SIZE = (512, 512)
    res_name = "512×512"

st.info(f"Evaluating at **{res_name}** ({EVAL_SIZE[0]}×{EVAL_SIZE[1]} = {EVAL_SIZE[0]*EVAL_SIZE[1]:,} pixels)")

# Basic existence checks
if model_type == "U-Net" and not hasattr(region, "unet_path"):
    st.error("No U-Net model path found for this region.")
    st.stop()

if model_type == "ConvLSTM" and not hasattr(region, "convlstm_path"):
    st.error("No ConvLSTM model path found for this region.")
    st.stop()

# ─── Find overlapping years ─────────────────────────────────────────────────
overlapping = sorted(set(region.image_years) & set(region.mask_years))
if not overlapping:
    st.warning("No years have both image and mask files available for this region.")
    st.stop()

st.markdown(f"**Region:** {region.display_name}")
st.markdown(f"Available years: **{overlapping}**")

# ─── User Settings ───────────────────────────────────────────────────────────
selected_years = st.multiselect(
    "Years to evaluate (U-Net) / input sequence years (ConvLSTM)",
    overlapping,
    default=overlapping[-min(5, len(overlapping)):]
)

# Region-aware default threshold suggestion
default_thresh = 0.5
if "Hyderabad" in region.display_name or "hyderbad" in region.display_name.lower():
    default_thresh = 0.3  # lower for regions where model under-predicts urban
    st.info("Using lower default threshold (0.3) for Hyderabad-like regions to avoid zero predictions.")

thresh = st.slider("Binary threshold", 0.05, 0.95, default_thresh, 0.05)

if model_type == "ConvLSTM":
    st.info("ConvLSTM uses last 5 years to predict next year.")
    if len(selected_years) < 5:
        st.warning("ConvLSTM needs at least 5 years.")
        st.stop()

    if EVAL_SIZE == (512, 512):
        st.warning("  512×512 + ConvLSTM may use a lot of memory. If OOM, switch to 256×256.")

# ─── Evaluation Button ───────────────────────────────────────────────────────
if st.button(f"Run Evaluation ({res_name})", type="primary") and selected_years:

    progress = st.progress(0)
    status = st.empty()

    rows = []
    conf_matrices = {}
    conf_labels = []

    # ────────────────────────────────────────────────────────────────────────────
    # U-Net branch (Keras .h5 model)
    # ────────────────────────────────────────────────────────────────────────────
    if model_type == "U-Net":
        model = load_unet(str(region.unet_path))

        for idx, year in enumerate(selected_years):
            status.text(f"U-Net → {year} @ {res_name}")

            # Load image
            with rasterio.open(str(region.images[year])) as src:
                img = np.moveaxis(src.read(), 0, -1).astype(np.float32)
                img_resized = cv2.resize(img, EVAL_SIZE, cv2.INTER_LANCZOS4)

            # Load and prepare ground truth
            with rasterio.open(str(region.masks[year])) as src:
                gt_raw = src.read(1).astype(np.float32)
                if gt_raw.max() > 1.5:
                    gt_raw = (gt_raw > 127).astype(np.float32)
                gt_raw = np.clip(gt_raw, 0, 1)
                gt_resized = cv2.resize(gt_raw, EVAL_SIZE, cv2.INTER_NEAREST)
                gt_bin = (gt_resized >= 0.5).astype(np.uint8)

            # Keras prediction + channel selection fix
            pred_prob_full = model.predict(np.expand_dims(img_resized, 0), verbose=0)[0]  # (H, W, C)

            # Debug for problematic regions/years
            debug = False
            if "Hyderabad" in region.display_name and year == "2021":
                debug = True
                st.subheader(f"Debug: {region.display_name} {year}")
                st.write("Model output shape:", pred_prob_full.shape)
                for ch in range(pred_prob_full.shape[-1]):
                    mean_prob = pred_prob_full[..., ch].mean()
                    st.write(f"Channel {ch} mean probability: {mean_prob:.4f}")

            # Choose the channel with highest mean probability (auto-detect urban channel)
            channel_probs = [pred_prob_full[..., ch].mean() for ch in range(pred_prob_full.shape[-1])]
            best_channel = np.argmax(channel_probs)
            pred_prob = pred_prob_full[..., best_channel]

            if debug:
                st.write(f"→ Selected channel {best_channel} (highest mean: {channel_probs[best_channel]:.4f})")

            # Resize if needed
            if pred_prob.shape[:2] != EVAL_SIZE:
                pred_prob = cv2.resize(pred_prob, EVAL_SIZE, cv2.INTER_LINEAR)

            pred_bin = (pred_prob >= thresh).astype(np.uint8)

            # Safety check
            if gt_bin.shape != pred_bin.shape:
                st.error(f"Shape mismatch in U-Net for {year}!\nGT: {gt_bin.shape}, Pred: {pred_bin.shape}")
                st.stop()

            # Extra debug: if zero urban predicted
            if pred_bin.sum() == 0:
                st.warning(f"Zero urban pixels predicted for {year} — all probabilities below threshold {thresh}")

            m = compute_metrics(gt_bin, pred_bin)

            rows.append({
                "Year": year,
                "IoU": float(m.iou) if hasattr(m, 'iou') else np.nan,
                "Accuracy": float(m.accuracy) if hasattr(m, 'accuracy') else np.nan,
                "Precision": float(m.precision) if hasattr(m, 'precision') else np.nan,
                "Recall": float(m.recall) if hasattr(m, 'recall') else np.nan,
                "F1": float(m.f1) if hasattr(m, 'f1') else np.nan,
            })
            conf_matrices[year] = m.confusion_matrix
            conf_labels.append(year)

            progress.progress((idx + 1) / len(selected_years))

    # ────────────────────────────────────────────────────────────────────────────
    # ConvLSTM branch (PyTorch .pth model)
    # ────────────────────────────────────────────────────────────────────────────
    else:
        model = UrbanSprawlConvLSTM(1, 32, 2, 1)
        state = torch.load(str(region.convlstm_path), map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        for idx in range(len(selected_years) - 4):
            input_years = selected_years[idx:idx + 5]
            target_year = input_years[-1] + 1

            if target_year not in region.mask_years:
                continue

            status.text(f"ConvLSTM → {input_years} → {target_year} @ {res_name}")

            seq = []
            for y in input_years:
                with rasterio.open(str(region.masks[y])) as src:
                    msk = src.read(1).astype(np.float32)
                    if msk.max() > 1.5:
                        msk = (msk > 127).astype(np.float32)
                    msk = np.clip(msk, 0, 1)
                    msk_resized = cv2.resize(msk, EVAL_SIZE, cv2.INTER_NEAREST)
                    seq.append(msk_resized)

            seq_np = np.stack(seq)[:, np.newaxis, :, :]
            seq_t = torch.from_numpy(seq_np).float().to(device).unsqueeze(0)

            with torch.no_grad():
                out = model(seq_t)
                pred_prob = out[0, 0].cpu().numpy()

            if pred_prob.shape[:2] != EVAL_SIZE:
                pred_prob = cv2.resize(pred_prob, EVAL_SIZE, cv2.INTER_LINEAR)

            pred_bin = (pred_prob >= thresh).astype(np.uint8)

            with rasterio.open(str(region.masks[target_year])) as src:
                gt_raw = src.read(1).astype(np.float32)
                if gt_raw.max() > 1.5:
                    gt_raw = (gt_raw > 127).astype(np.float32)
                gt_raw = np.clip(gt_raw, 0, 1)
                gt_resized = cv2.resize(gt_raw, EVAL_SIZE, cv2.INTER_NEAREST)
                gt_bin = (gt_resized >= 0.5).astype(np.uint8)

            if gt_bin.shape != pred_bin.shape:
                st.error(f"Shape mismatch in ConvLSTM for pred {target_year}!")
                st.stop()

            m = compute_metrics(gt_bin, pred_bin)

            label = f"Pred {target_year} (from {input_years[0]}–{input_years[-1]})"
            rows.append({
                "Year": label,
                "IoU": float(m.iou) if hasattr(m, 'iou') else np.nan,
                "Accuracy": float(m.accuracy) if hasattr(m, 'accuracy') else np.nan,
                "Precision": float(m.precision) if hasattr(m, 'precision') else np.nan,
                "Recall": float(m.recall) if hasattr(m, 'recall') else np.nan,
                "F1": float(m.f1) if hasattr(m, 'f1') else np.nan,
            })
            conf_matrices[label] = m.confusion_matrix
            conf_labels.append(label)

            progress.progress((idx + 1) / max(1, len(selected_years) - 4))

    progress.empty()
    status.empty()

    if not rows:
        st.error("No valid evaluation results were generated.")
        st.stop()

    # ─── Results ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    st.subheader(f"Metrics per Year / Prediction ({res_name})")
    st.dataframe(
        df.style.format("{:.4f}", subset=df.columns[1:]).highlight_max(
            subset=["IoU", "Accuracy", "Precision", "Recall", "F1"], axis=0
        ),
        use_container_width=True
    )

    st.subheader(f"Performance Trends ({res_name})")
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in ["IoU", "Accuracy", "Precision", "Recall", "F1"]:
        if col in df.columns:
            ax.plot(df["Year"], df[col], marker="o", label=col, linewidth=2)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader(f"Confusion Matrices ({res_name})")
    if conf_matrices:
        cols = st.columns(3)
        for idx, label in enumerate(conf_labels):
            cm = conf_matrices[label]
            with cols[idx % 3]:
                fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
                im = ax_cm.imshow(cm, cmap="Blues")
                for (i, j), val in np.ndenumerate(cm):
                    ax_cm.text(j, i, int(val), ha="center", va="center",
                               color="white" if val > cm.max()/2 else "black", fontsize=10)
                ax_cm.set_xticks([0, 1])
                ax_cm.set_yticks([0, 1])
                ax_cm.set_xticklabels(["Non-Urban", "Urban"])
                ax_cm.set_yticklabels(["Non-Urban", "Urban"])
                ax_cm.set_title(label, fontsize=11)
                plt.colorbar(im, ax=ax_cm, fraction=0.046)
                st.pyplot(fig_cm)

    csv = df.to_csv(index=False)
    st.download_button(
        "Download Metrics as CSV",
        csv,
        f"eval_{region.display_name}_{model_type.lower()}_{EVAL_SIZE[0]}x{EVAL_SIZE[1]}.csv",
        "text/csv"
    )

else:
    st.info("Select years and click the button to start evaluation.")