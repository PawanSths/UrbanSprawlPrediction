"""
core/loaders.py — Cached model loaders used by all pages.
Supports both PyTorch (.pth) and Keras (.h5) models with multiple architectures.
"""

from __future__ import annotations

import streamlit as st
import torch
import torch.nn as nn
from pathlib import Path


@st.cache_resource(show_spinner="Loading U-Net model …")
def load_unet(model_path: str):
    """
    Load a Keras U-Net .h5 file. Cached so it loads once per app lifecycle.
    """
    from tensorflow.keras.models import load_model
    return load_model(model_path, compile=False)


@st.cache_resource(show_spinner="Loading ConvLSTM model …")
def load_convlstm(model_path: str):
    """
    Load ConvLSTM from either PyTorch (.pth) or Keras (.h5) checkpoint.
    Returns a tuple: (model, framework) where framework is 'pytorch' or 'keras'.

    Supports multiple architectures:
    - Kathmandu: Binary urban prediction (UrbanSprawlConvLSTM)
    - Hyderabad: 4-class segmentation (Keras .h5)
    - Pokhara: 3-class temporal prediction (PokharaConvLSTM)
    - TaxiBJ: Multi-layer ConvLSTM (TaxiBJConvLSTM)
    """
    path = Path(model_path)

    if path.suffix.lower() == ".h5":
        # Load Keras model
        from tensorflow.keras.models import load_model
        model = load_model(model_path, compile=False)
        return model, "keras"

    elif path.suffix.lower() == ".pth":
        device = get_device()

        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)

            # Strategy 1: Complete model object
            if isinstance(checkpoint, nn.Module):
                model = checkpoint
                model.eval()
                return model.to(device), "pytorch"

            # Strategy 2: State dict - detect architecture
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get("model_state_dict", checkpoint)

                # Get list of keys for architecture detection
                keys = list(state_dict.keys())

                # ─── Architecture Detection ───

                # 1. TaxiBJ ConvLSTM (conv_x, conv_h, conv_o structure)
                if "cell_list.0.conv_x.0.weight" in keys:
                    st.info("📦 Detected: TaxiBJ ConvLSTM")

                    try:
                        from models import TaxiBJConvLSTM
                    except ImportError:
                        st.error("❌ TaxiBJConvLSTM not found in models.py. Add it first.")
                        raise

                    # Determine parameters from state dict
                    num_layers = len([k for k in keys if k.startswith("cell_list.") and k.endswith(".conv_x.0.weight")])

                    # Get hidden channels from first layer
                    first_conv_weight = state_dict["cell_list.0.conv_x.0.weight"]
                    hidden_channels = first_conv_weight.shape[0] // 4  # Divided by 4 for LSTM gates

                    # Get input channels
                    input_channels = first_conv_weight.shape[1]

                    # Check if has classifier
                    has_classifier = any("classifier" in k for k in keys)
                    num_classes = 3
                    if has_classifier:
                        classifier_keys = [k for k in keys if "classifier" in k and "weight" in k]
                        if classifier_keys:
                            num_classes = state_dict[classifier_keys[0]].shape[0]

                    st.info(f"   📊 Layers: {num_layers} | Hidden: {hidden_channels} | Input: {input_channels} | Classes: {num_classes}")

                    model = TaxiBJConvLSTM(
                        input_channels=input_channels,
                        hidden_channels=hidden_channels,
                        num_layers=num_layers,
                        num_classes=num_classes
                    )

                    # Load weights
                    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

                    if missing_keys:
                        st.warning(f"⚠️  Missing keys: {len(missing_keys)}")
                    if unexpected_keys:
                        st.caption(f"ℹ️  Unexpected keys: {len(unexpected_keys)} (ignored)")

                    st.success("✅ Loaded TaxiBJ ConvLSTM weights")

                # 2. Pokhara ConvLSTM (encoder-decoder with standard ConvLSTM)
                elif any("encoder" in k for k in keys) and "decoder" in " ".join(keys):
                    st.info("📦 Detected: Pokhara ConvLSTM (3-class temporal)")

                    try:
                        from models import PokharaConvLSTM
                    except ImportError:
                        st.error("❌ PokharaConvLSTM not found in models.py")
                        raise

                    # Instantiate with typical Pokhara config
                    model = PokharaConvLSTM(
                        input_channels=1,
                        encoder_channels=[32, 64],
                        convlstm_hidden=64,
                        num_classes=3
                    )

                    # Load weights
                    model.load_state_dict(state_dict, strict=False)
                    st.success("✅ Loaded Pokhara ConvLSTM weights")

                # 3. Kathmandu ConvLSTM (binary urban prediction)
                elif "convlstm.cell_list.0.conv.weight" in keys:
                    st.info("📦 Detected: Kathmandu ConvLSTM (binary urban)")

                    try:
                        from models import UrbanSprawlConvLSTM
                    except ImportError:
                        st.error("❌ UrbanSprawlConvLSTM not found in models.py")
                        raise

                    # Typical Kathmandu config
                    model = UrbanSprawlConvLSTM(
                        input_channels=1,
                        hidden_dim=32,
                        num_layers=2,
                        output_channels=1
                    )

                    model.load_state_dict(state_dict)
                    st.success("✅ Loaded Kathmandu ConvLSTM weights")

                # 4. Alternative encoder-decoder architecture
                elif "encoder.encoder.0.weight" in keys:
                    st.info("📦 Detected: Encoder-Decoder ConvLSTM")

                    try:
                        from models import EncoderDecoderConvLSTM
                    except ImportError:
                        # Fallback: try Pokhara architecture
                        from models import PokharaConvLSTM as EncoderDecoderConvLSTM

                    model = EncoderDecoderConvLSTM(
                        input_channels=1,
                        hidden_dim=64
                    )

                    model.load_state_dict(state_dict, strict=False)
                    st.success("✅ Loaded Encoder-Decoder ConvLSTM weights")

                else:
                    # Unknown architecture - show keys for debugging
                    st.error(
                        f"❌ Unknown ConvLSTM architecture!\n\n"
                        f"**Found keys:**\n{keys[:10]}\n\n"
                        f"Add the matching architecture to `models.py` and update `core/loaders.py`"
                    )
                    raise ValueError(f"Unknown architecture. First keys: {keys[:10]}")

                model.eval()
                return model.to(device), "pytorch"

        except Exception as e:
            st.error(f"❌ Failed to load PyTorch model: {e}")
            st.error(
                f"**Model path:** {model_path}\n\n"
                f"**Troubleshooting:**\n"
                f"1. Ensure the model architecture exists in `models.py`\n"
                f"2. Check that the state_dict keys match the architecture\n"
                f"3. Verify the model was saved correctly during training\n"
                f"4. Run `python manual_taxibj_test.py` to analyze the model structure"
            )
            raise

    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Expected .pth or .h5")


def get_device() -> torch.device:
    """Get available PyTorch device (CUDA if available, else CPU)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


@st.cache_resource(show_spinner="Loading model configuration...")
def get_model_info(model_path: str) -> dict:
    """
    Get model metadata without fully loading it.
    Returns dict with architecture type, parameters, etc.
    """
    path = Path(model_path)

    if not path.exists():
        return {"error": "File not found"}

    info = {
        "path": str(path),
        "size_mb": path.stat().st_size / (1024 * 1024),
        "format": path.suffix.lower()
    }

    if path.suffix.lower() == ".pth":
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

            if isinstance(checkpoint, nn.Module):
                info["type"] = "Complete model object"
                info["parameters"] = sum(p.numel() for p in checkpoint.parameters())
            elif isinstance(checkpoint, dict):
                state_dict = checkpoint.get("model_state_dict", checkpoint)
                keys = list(state_dict.keys())

                info["num_keys"] = len(keys)
                info["sample_keys"] = keys[:5]

                # Detect architecture
                if "cell_list.0.conv_x.0.weight" in keys:
                    info["architecture"] = "TaxiBJ ConvLSTM"
                elif any("encoder" in k for k in keys):
                    info["architecture"] = "Pokhara ConvLSTM"
                elif "convlstm.cell_list.0.conv.weight" in keys:
                    info["architecture"] = "Kathmandu ConvLSTM"
                else:
                    info["architecture"] = "Unknown"

        except Exception as e:
            info["error"] = str(e)

    elif path.suffix.lower() == ".h5":
        info["format"] = "Keras/TensorFlow"
        info["architecture"] = "U-Net or ConvLSTM"

    return info