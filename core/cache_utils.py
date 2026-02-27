"""
cache_utils.py — Caching functions for model predictions
Speeds up repeated predictions significantly
"""

import hashlib
import numpy as np
import streamlit as st
import tensorflow as tf
from pathlib import Path
from core.loaders import load_unet


def _get_patch_hash(patch: np.ndarray) -> str:
    """
    Generate a hash of the patch for cache keys.
    Uses only the first 1000 bytes to speed up hashing.
    """
    return hashlib.md5(patch[:, :, 0].tobytes()[:1000]).hexdigest()[:8]


@st.cache_resource
def _load_model_cached(model_path: str):
    """
    Load and cache model at module level.
    Streamlit caches this automatically.
    """
    return load_unet(model_path)


@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_multiclass_predictions(
        model_path: str,
        patch: np.ndarray,
        region_name: str = "default",
        year: int = 2023
) -> tuple:
    """
    Get multiclass predictions with caching.

    Args:
        model_path: Path to multiclass U-Net model (.h5)
        patch: Image patch [256, 256, num_bands]
        region_name: Region identifier for cache key
        year: Year for cache key

    Returns:
        (probs, classes, confidence)
            - probs: [256, 256, num_classes] float probabilities
            - classes: [256, 256] uint8 class indices
            - confidence: [256, 256] float max probability per pixel
    """
    model = _load_model_cached(model_path)

    # Get probabilities
    probs = model.predict(patch[np.newaxis, ...], verbose=0)[0]  # [256, 256, num_classes]

    # Get class predictions (argmax)
    classes = np.argmax(probs, axis=-1).astype("uint8")

    # Get confidence scores (max probability)
    confidence = np.max(probs, axis=-1)

    return probs, classes, confidence


@st.cache_data(ttl=3600)
def get_binary_predictions(
        model_path: str,
        patch: np.ndarray,
        region_name: str = "default",
        year: int = 2023,
        threshold: float = 0.5
) -> tuple:
    """
    Get binary predictions with caching.

    Args:
        model_path: Path to binary U-Net model (.h5)
        patch: Image patch [256, 256, num_bands]
        region_name: Region identifier for cache key
        year: Year for cache key
        threshold: Threshold for binary classification

    Returns:
        (probs, binary_mask, confidence)
            - probs: [256, 256] float sigmoid output
            - binary_mask: [256, 256] uint8 (0 or 1)
            - confidence: [256, 256] float probability
    """
    model = _load_model_cached(model_path)

    # Get probabilities [256, 256, 1]
    probs = model.predict(patch[np.newaxis, ...], verbose=0)[0, :, :, 0]

    # Apply threshold
    binary_mask = (probs >= threshold).astype("uint8")

    return probs, binary_mask, probs