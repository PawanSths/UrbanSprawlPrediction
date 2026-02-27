# pages/05_HyderabadEvaluation.py
# Hyderabad — U-Net + ConvLSTM multi-class evaluation (4 classes)
# U-Net:    Keras, input=(256,256,10 bands), output=(256,256,4) softmax
# ConvLSTM: Keras, input=(8 timesteps, 256,256,4 one-hot), output=(256,256,4) softmax

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio
import cv2
import tensorflow as tf
from core.config import RegionConfig
GLOBAL_MEAN = np.array([1.0546437e+03, 1.4515833e+03, 1.5928341e+03, 2.8228894e+03,
                        3.4636748e+03, 2.5146472e+03, 2.6376092e-01, -3.3397126e-01,
                        -2.7405408e-01, 4.9883842e-02], dtype=np.float32)

GLOBAL_STD = np.array([3.8909647e+02, 4.0717056e+02, 5.1408856e+02, 5.7885687e+02,
                       8.1401770e+02, 7.0319159e+02, 1.6960555e-01, 1.5608695e-01,
                       1.5603507e-01, 1.6167323e-01], dtype=np.float32)

st.set_page_config(page_title="Hyderabad Evaluation", layout="wide")
st.title("Hyderabad — Multi-Class Model Evaluation")

# ─── Class definitions ───────────────────────────────────────────────────────
# Hyderabad has 4 classes [0,1,2,3].
#  Rename these labels to match your actual ground truth legend:
CLASSES      = {0: "Urban", 1: "Forest", 2: "Water", 3: "Barren"}
CLASS_COLORS = {0: "#800000", 1: "#228B22", 2: "#00BFFF", 3: "#D2B48C"}
N_CLASSES    = 4

MODEL_INPUT_SIZE  = (256, 256)   # both models were trained at this size
CONVLSTM_TIMESTEPS = 8           # Keras ConvLSTM input: (8, 256, 256, 4)
UNET_BANDS         = 10          # U-Net image bands

# ─── Region check ────────────────────────────────────────────────────────────
region: RegionConfig = st.session_state.get("region")
if region is None:
    st.warning("⬅️ Please select a region from the main page first.")
    st.stop()

if "hyderabad" not in region.display_name.lower():
    st.warning(f"This page is for Hyderabad. Current region: **{region.display_name}**.")
    st.stop()

st.write(f"**Region:** {region.display_name}")

# ─── Model selection ─────────────────────────────────────────────────────────
model_type = st.radio("Model to evaluate", ["U-Net", "ConvLSTM"], horizontal=True)

col1, col2 = st.columns(2)
with col1:
    resolution_choice = st.selectbox(
        "Display Resolution (metrics computed at 256×256 always)",
        ["256×256", "512×512 (upsampled for display)"],
        index=0
    )
    DISPLAY_SIZE = (256, 256) if "256" in resolution_choice else (512, 512)

# Overlapping years
all_image_years = sorted(region.image_years)
all_mask_years  = sorted(region.mask_years)

st.markdown(
    f"**Image years:** {all_image_years}  |  **Mask years:** {all_mask_years}"
)

if model_type == "U-Net":
    overlapping = sorted(set(all_image_years) & set(all_mask_years))
    if not overlapping:
        st.warning("No overlapping image + mask years for U-Net evaluation.")
        st.stop()
    selected_years = st.multiselect(
        "Years to evaluate (U-Net — each year independently)",
        overlapping,
        default=overlapping
    )
    if not selected_years:
        st.stop()

else:  # ConvLSTM
    st.info(f"ConvLSTM uses **{CONVLSTM_TIMESTEPS} consecutive mask years** as input to predict the next year.")
    seq_candidates = [y for y in all_mask_years]
    if len(seq_candidates) < CONVLSTM_TIMESTEPS + 1:
        st.warning(f"Need at least {CONVLSTM_TIMESTEPS + 1} mask years. Found {len(seq_candidates)}.")
        st.stop()

    # Let user pick target year; input = last 8 years before it
    possible_targets = [y for y in all_mask_years if len([x for x in all_mask_years if x < y]) >= CONVLSTM_TIMESTEPS]
    if not possible_targets:
        st.warning("Not enough mask years to form an input sequence.")
        st.stop()
    target_year = st.selectbox("Target prediction year (GT mask required)", possible_targets)
    input_years = sorted([y for y in all_mask_years if y < target_year])[-CONVLSTM_TIMESTEPS:]
    st.info(f"Input years: **{input_years}** → predicting **{target_year}**")

# ─── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_resource
def load_unet_hyderabad(model_path: str):
    return tf.keras.models.load_model(model_path, compile=False)

@st.cache_resource
def load_convlstm_hyderabad(model_path: str):
    return tf.keras.models.load_model(model_path, compile=False)


def mask_to_onehot(mask_2d: np.ndarray, n_classes: int) -> np.ndarray:
    """Convert (H, W) class map to (H, W, n_classes) one-hot float32."""
    h, w = mask_2d.shape
    onehot = np.zeros((h, w, n_classes), dtype=np.float32)
    for c in range(n_classes):
        onehot[:, :, c] = (mask_2d == c).astype(np.float32)
    return onehot


def resize_mask(mask_2d: np.ndarray, size: tuple) -> np.ndarray:
    """Resize a class label mask (INTER_NEAREST)."""
    return cv2.resize(mask_2d.astype(np.float32), size,
                      interpolation=cv2.INTER_NEAREST).astype(np.uint8)


def load_and_resize_image(path: str, size: tuple, n_bands: int,
                          mean=None, std=None) -> np.ndarray:  # 👈 ADD THESE PARAMS
    """Load GeoTIFF, force to n_bands, resize to (H, W, n_bands)."""
    with rasterio.open(path) as src:
        img = np.moveaxis(src.read(), 0, -1).astype(np.float32)  # (H, W, C)
    if img.shape[-1] > n_bands:
        img = img[..., :n_bands]
    elif img.shape[-1] < n_bands:
        pad = np.zeros((*img.shape[:2], n_bands - img.shape[-1]), dtype=np.float32)
        img = np.concatenate([img, pad], axis=-1)
    img = cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)

    # 👇 ADD THIS NORMALIZATION BLOCK 👇
    if mean is not None and std is not None:
        img = np.nan_to_num(img, nan=0.0)  # Handle NaNs
        img = (img - mean) / std  # Apply GLOBAL_MEAN/STD
    # 👆 👆 👆

    return img.astype(np.float32)

def load_and_resize_mask(path: str, size: tuple) -> np.ndarray:
    """Load GeoTIFF mask, normalise to class indices, resize."""
    with rasterio.open(path) as src:
        raw = src.read(1).astype(np.uint8)
    return resize_mask(raw, size)


def compute_multiclass_metrics(gt: np.ndarray, pred: np.ndarray, n_classes: int) -> dict:
    """Compute confusion matrix, per-class and macro metrics."""
    from sklearn.metrics import confusion_matrix as sk_cm
    labels = list(range(n_classes))
    cm = sk_cm(gt.ravel(), pred.ravel(), labels=labels)

    per_class = {}
    for c in labels:
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        iou       = tp / (tp + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        per_class[c] = dict(iou=iou, precision=precision, recall=recall, f1=f1,
                            tp=int(tp), fp=int(fp), fn=int(fn))

    accuracy = float(np.diag(cm).sum() / cm.sum())
    macro = {k: float(np.mean([per_class[c][k] for c in labels]))
             for k in ["iou", "precision", "recall", "f1"]}
    return dict(per_class=per_class, macro=macro, accuracy=accuracy, confusion_matrix=cm)


def plot_confusion_matrix(cm, class_names, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f"{int(val):,}", ha="center", va="center", fontsize=8,
                color="white" if val > cm.max() * 0.6 else "black")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Ground Truth", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    return fig


def plot_class_map(mask_2d, title, display_size=None):
    if display_size:
        mask_2d = resize_mask(mask_2d, display_size)
    cmap   = mcolors.ListedColormap([CLASS_COLORS[c] for c in range(N_CLASSES)])
    bounds = [-0.5 + c for c in range(N_CLASSES + 1)]
    norm   = mcolors.BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(mask_2d, cmap=cmap, norm=norm, interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax, ticks=range(N_CLASSES), fraction=0.046)
    cbar.ax.set_yticklabels([CLASSES[c] for c in range(N_CLASSES)], fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    return fig


def show_results(gt_eval: np.ndarray, pred_label: np.ndarray,
                 pred_prob: np.ndarray, tag: str):
    """Render metrics, maps, confusion matrix for one prediction."""
    metrics = compute_multiclass_metrics(gt_eval, pred_label, N_CLASSES)

    # Summary cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Accuracy", f"{metrics['accuracy']:.4f}")
    c2.metric("Macro IoU",        f"{metrics['macro']['iou']:.4f}")
    c3.metric("Macro F1",         f"{metrics['macro']['f1']:.4f}")
    c4.metric("Macro Recall",     f"{metrics['macro']['recall']:.4f}")

    # Per-class table
    rows = []
    for c in range(N_CLASSES):
        m = metrics["per_class"][c]
        rows.append({
            "Class": f"{c} — {CLASSES[c]}",
            "IoU": m["iou"], "Precision": m["precision"],
            "Recall": m["recall"], "F1": m["f1"],
            "GT pixels": m["tp"] + m["fn"],
        })
    df = pd.DataFrame(rows)
    st.dataframe(
        df.style.format({
            "IoU": "{:.4f}", "Precision": "{:.4f}",
            "Recall": "{:.4f}", "F1": "{:.4f}", "GT pixels": "{:,}"
        }).highlight_max(subset=["IoU", "Precision", "Recall", "F1"], axis=0),
        use_container_width=True
    )

    # Maps
    st.subheader("🗺️ Spatial Maps")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.pyplot(plot_class_map(gt_eval, f"Ground Truth — {tag}", DISPLAY_SIZE))
    with m2:
        st.pyplot(plot_class_map(pred_label, f"Predicted — {tag}", DISPLAY_SIZE))
    with m3:
        error_map = (gt_eval != pred_label).astype(np.uint8)
        if DISPLAY_SIZE != (256, 256):
            error_map = cv2.resize(error_map.astype(np.float32), DISPLAY_SIZE, cv2.INTER_NEAREST).astype(np.uint8)
        fig_e, ax_e = plt.subplots(figsize=(6, 4))
        ax_e.imshow(error_map, cmap="RdYlGn_r", vmin=0, vmax=1, interpolation="nearest")
        ax_e.set_title(f"Error Map — {tag}\n(red=wrong, green=correct)", fontsize=11, fontweight="bold")
        ax_e.axis("off")
        plt.tight_layout()
        st.pyplot(fig_e)

    # Probability maps
    if pred_prob is not None and pred_prob.shape[-1] == N_CLASSES:
        st.subheader("📉 Class Probability Maps")
        prob_cols = st.columns(N_CLASSES)
        for c in range(N_CLASSES):
            pm = pred_prob[..., c]
            if DISPLAY_SIZE != MODEL_INPUT_SIZE:
                pm = cv2.resize(pm, DISPLAY_SIZE, cv2.INTER_LINEAR)
            with prob_cols[c]:
                fig_p, ax_p = plt.subplots(figsize=(4, 3))
                im_p = ax_p.imshow(pm, cmap="hot", vmin=0, vmax=1)
                plt.colorbar(im_p, ax=ax_p, fraction=0.046)
                ax_p.set_title(f"P({CLASSES[c]})", fontsize=9, fontweight="bold")
                ax_p.axis("off")
                plt.tight_layout()
                st.pyplot(fig_p)

    # Confusion matrix
    st.subheader("🧮 Confusion Matrix")
    class_names = [f"{c}\n{CLASSES[c]}" for c in range(N_CLASSES)]
    st.pyplot(plot_confusion_matrix(metrics["confusion_matrix"], class_names, f"Confusion Matrix — {tag}"))

    # Export
    csv_rows = []
    for c in range(N_CLASSES):
        m = metrics["per_class"][c]
        csv_rows.append({"class_id": c, "class_name": CLASSES[c],
                         "iou": m["iou"], "precision": m["precision"],
                         "recall": m["recall"], "f1": m["f1"],
                         "tp": m["tp"], "fp": m["fp"], "fn": m["fn"]})
    csv_rows.append({"class_id": "macro", "class_name": "MACRO",
                     "iou": metrics["macro"]["iou"], "precision": metrics["macro"]["precision"],
                     "recall": metrics["macro"]["recall"], "f1": metrics["macro"]["f1"],
                     "tp": "", "fp": "", "fn": ""})
    st.download_button(
        "💾 Download Metrics CSV",
        pd.DataFrame(csv_rows).to_csv(index=False),
        f"hyderabad_{model_type.lower()}_eval_{tag}.csv",
        "text/csv",
        key=f"dl_{tag}"
    )

    return metrics


# ─── Run Evaluation ──────────────────────────────────────────────────────────
if st.button(f"▶ Run Hyderabad {model_type} Evaluation", type="primary"):

    # ── U-Net branch ──────────────────────────────────────────────────────────
    if model_type == "U-Net":
        with st.spinner("Loading U-Net..."):
            try:
                unet = load_unet_hyderabad(str(region.unet_path))
                st.success(
                    f"U-Net loaded — input: {unet.input_shape}, output: {unet.output_shape}"
                )
            except Exception as e:
                st.error(f"Failed to load U-Net: {e}")
                st.stop()

        all_metrics = []
        progress = st.progress(0)
        status   = st.empty()

        for idx, year in enumerate(selected_years):
            status.text(f"U-Net → {year} ({idx+1}/{len(selected_years)})")

            # Load & prepare image
            try:
                img = load_and_resize_image(
                    str(region.images[year]),
                    MODEL_INPUT_SIZE,
                    UNET_BANDS,
                    mean=GLOBAL_MEAN,  # 👈 ADD THIS
                    std=GLOBAL_STD  # 👈 ADD THIS
                )

            except Exception as e:
                st.warning(f"Could not load image for {year}: {e}")
                progress.progress((idx + 1) / len(selected_years))
                continue

            # Predict
            pred_raw   = unet.predict(np.expand_dims(img, 0), verbose=0)[0]  # (256,256,4)
            pred_prob  = pred_raw  # softmax output
            pred_label = np.argmax(pred_prob, axis=-1).astype(np.uint8)      # (256,256)

            # Load GT
            try:
                gt_eval = load_and_resize_mask(str(region.masks[year]), MODEL_INPUT_SIZE)
            except Exception as e:
                st.warning(f"Could not load GT mask for {year}: {e}")
                progress.progress((idx + 1) / len(selected_years))
                continue

            st.markdown(f"---\n### Year {year}")
            m = show_results(gt_eval, pred_label, pred_prob, str(year))
            m["year"] = year
            all_metrics.append(m)
            progress.progress((idx + 1) / len(selected_years))

        progress.empty()
        status.empty()

        # Trend chart across years
        if len(all_metrics) > 1:
            st.markdown("---")
            st.subheader("📈 Performance Trends Across Years")
            fig_t, ax_t = plt.subplots(figsize=(10, 5))
            years_plot = [m["year"] for m in all_metrics]
            for metric_name in ["accuracy", "macro_iou", "macro_f1"]:
                vals = []
                for m in all_metrics:
                    if metric_name == "accuracy":
                        vals.append(m["accuracy"])
                    elif metric_name == "macro_iou":
                        vals.append(m["macro"]["iou"])
                    elif metric_name == "macro_f1":
                        vals.append(m["macro"]["f1"])
                ax_t.plot(years_plot, vals, marker="o",
                          label=metric_name.replace("_", " ").title())
            ax_t.set_ylim(0, 1.05)
            ax_t.legend()
            ax_t.grid(True, alpha=0.3)
            ax_t.set_xlabel("Year")
            ax_t.set_ylabel("Score")
            ax_t.set_title("Hyderabad U-Net — Multi-Year Performance")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig_t)

            # Per-class trends
            st.subheader("📈 Per-Class IoU Trends")
            fig_c, ax_c = plt.subplots(figsize=(10, 5))
            for c in range(N_CLASSES):
                iou_vals = [m["per_class"][c]["iou"] for m in all_metrics]
                ax_c.plot(years_plot, iou_vals, marker="s",
                          color=CLASS_COLORS[c], label=f"{c}: {CLASSES[c]}")
            ax_c.set_ylim(0, 1.05)
            ax_c.legend()
            ax_c.grid(True, alpha=0.3)
            ax_c.set_xlabel("Year")
            ax_c.set_ylabel("IoU")
            ax_c.set_title("Per-Class IoU — Hyderabad U-Net")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig_c)

    # ── ConvLSTM branch ───────────────────────────────────────────────────────
    else:
        with st.spinner("Loading ConvLSTM (Keras)..."):
            try:
                convlstm = load_convlstm_hyderabad(str(region.convlstm_path))
                st.success(
                    f"ConvLSTM loaded — input: {convlstm.input_shape}, output: {convlstm.output_shape}"
                )
            except Exception as e:
                st.error(f"Failed to load ConvLSTM: {e}")
                st.stop()

        progress = st.progress(0)
        status   = st.empty()

        # Build sequence: (1, 8, 256, 256, 4) — one-hot encoded masks
        seq = []
        for i, y in enumerate(input_years):
            status.text(f"Loading mask {y}...")
            mask_r = load_and_resize_mask(str(region.masks[y]), MODEL_INPUT_SIZE)
            onehot = mask_to_onehot(mask_r, N_CLASSES)  # (256, 256, 4)
            seq.append(onehot)
            progress.progress((i + 1) / (len(input_years) + 2))

        seq_np = np.stack(seq)[np.newaxis]  # (1, T, 256, 256, 4)

        # If model expects exactly CONVLSTM_TIMESTEPS, pad or truncate
        T_actual = seq_np.shape[1]
        if T_actual < CONVLSTM_TIMESTEPS:
            pad = np.zeros((1, CONVLSTM_TIMESTEPS - T_actual, *seq_np.shape[2:]), dtype=np.float32)
            seq_np = np.concatenate([pad, seq_np], axis=1)
            st.warning(f"Padded sequence from {T_actual} to {CONVLSTM_TIMESTEPS} timesteps with zeros.")
        elif T_actual > CONVLSTM_TIMESTEPS:
            seq_np = seq_np[:, -CONVLSTM_TIMESTEPS:]

        # Predict
        status.text(f"ConvLSTM forward pass → predicting {target_year}...")
        pred_raw   = convlstm.predict(seq_np, verbose=0)[0]          # (256,256,4)
        pred_prob  = pred_raw
        pred_label = np.argmax(pred_prob, axis=-1).astype(np.uint8)  # (256,256)

        progress.progress((len(input_years) + 1) / (len(input_years) + 2))

        # Load GT
        status.text(f"Loading GT mask {target_year}...")
        if target_year not in region.mask_years:
            st.error(f"No GT mask for {target_year} — cannot compute metrics.")
            st.stop()
        gt_eval = load_and_resize_mask(str(region.masks[target_year]), MODEL_INPUT_SIZE)

        progress.progress(1.0)
        progress.empty()
        status.empty()

        st.markdown("---")
        st.subheader(f"📊 ConvLSTM Results — Predicting {target_year} from {input_years[0]}–{input_years[-1]}")
        show_results(gt_eval, pred_label, pred_prob,
                     f"{target_year} (from {input_years[0]}-{input_years[-1]})")

