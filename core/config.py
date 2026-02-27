"""
core/config.py — Auto-discovers regions with proper historical/predicted separation
Now supports mixed binary/multiclass U-Net models with region-specific band counts
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─── Root paths ───
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

# ─── File-name patterns ───
IMAGE_PATTERNS = [
    re.compile(r"sentinel_\w+_(\d{4})\.tif$", re.IGNORECASE),
    re.compile(r"(\d{4})image\.tif$", re.IGNORECASE),
]

MASK_PATTERN = re.compile(r"mask_(\d{4})\.tif$", re.IGNORECASE)
UNET_BINARY_PATTERN = re.compile(r".*unet.*binary.*\.h5$", re.IGNORECASE)
UNET_MULTICLASS_PATTERN = re.compile(r".*unet.*multiclass.*\.h5$", re.IGNORECASE)
UNET_PATTERN = re.compile(r".*unet.*\.h5$", re.IGNORECASE)  # Fallback
CONVLSTM_PATTERN = re.compile(r".*convlstm.*\.(pth|h5)$", re.IGNORECASE)

# ─── Default map centers per region ───
KNOWN_MAP_CENTERS: Dict[str, Tuple[float, float, int]] = {
    "kathmandu": (27.70, 85.32, 12),
    "hyderabad": (17.385, 78.4867, 12),
    "pokhara": (28.2096, 83.9856, 13),
}

KNOWN_MAP_BOUNDS: Dict[str, Tuple[List[float], List[float]]] = {
    "kathmandu": ([27.55, 85.15], [27.85, 85.55]),
    "hyderabad": ([17.20, 78.20], [17.55, 78.70]),
    "pokhara": ([28.10, 83.90], [28.30, 84.10]),
}

# ─── CRITICAL: Define last historical year per region ───
LAST_HISTORICAL_YEAR = {
    "kathmandu": 2025,
    "hyderabad": 2026,
    "pokhara": 2024,
}

# ─── Region-specific configurations ───
# Key: region_name
# Value: {
#     "model_type": "binary" | "multiclass",
#     "num_bands": int,
#     "class_names": [optional list for multiclass],
#     "class_colors": [optional list for multiclass],
# }
REGION_SPECS: Dict[str, Dict] = {
    "kathmandu": {
        "model_type": "binary",
        "num_bands": 11,
    },
    "pokhara": {
        "model_type": "binary",
        "num_bands": 11,
    },
    "hyderabad": {
        "model_type": "multiclass",
        "num_bands": 10,  # ← Hyderabad has 10 bands
        "class_names": ["Urban", "Forest", "Water", "Barren"],
        "class_colors": ["#800000", "#228B22", "#00BFFF", "#D2B48C"],
    },
}


@dataclass
class RegionConfig:
    """Everything the app needs for one region — built automatically."""

    name: str
    display_name: str
    data_dir: Path

    images: Dict[int, Path] = field(default_factory=dict)
    masks: Dict[int, Path] = field(default_factory=dict)

    # Binary or Multiclass U-Net
    unet_binary_path: Optional[Path] = None
    unet_multiclass_path: Optional[Path] = None

    # ConvLSTM (can work with either binary or multiclass)
    convlstm_path: Optional[Path] = None

    image_prefix: str = ""
    image_suffix: str = ".tif"

    # Model type and band count
    model_type: str = "binary"  # "binary" or "multiclass"
    num_bands: int = 11

    # For multiclass only
    class_names: Optional[List[str]] = None
    class_colors: Optional[List[str]] = None

    @property
    def image_years(self) -> List[int]:
        return sorted(self.images.keys())

    @property
    def mask_years(self) -> List[int]:
        """All mask years (historical + predicted)"""
        return sorted(self.masks.keys())

    @property
    def historical_mask_years(self) -> List[int]:
        """Only ground truth masks (before last_historical_year)"""
        cutoff = LAST_HISTORICAL_YEAR.get(self.name, 2024)
        return sorted([y for y in self.masks.keys() if y <= cutoff])

    @property
    def predicted_mask_years(self) -> List[int]:
        """Only predicted masks (after last_historical_year)"""
        cutoff = LAST_HISTORICAL_YEAR.get(self.name, 2024)
        return sorted([y for y in self.masks.keys() if y > cutoff])

    # ─── U-Net Model Properties ───
    @property
    def has_binary_unet(self) -> bool:
        """Check if binary U-Net exists"""
        return (
            self.unet_binary_path is not None
            and self.unet_binary_path.exists()
        )

    @property
    def has_multiclass_unet(self) -> bool:
        """Check if multiclass U-Net exists"""
        return (
            self.unet_multiclass_path is not None
            and self.unet_multiclass_path.exists()
        )

    @property
    def has_unet(self) -> bool:
        """Check if ANY U-Net model exists"""
        return self.has_binary_unet or self.has_multiclass_unet

    @property
    def unet_path(self) -> Optional[Path]:
        """Get the appropriate U-Net path based on model_type"""
        if self.model_type == "multiclass":
            return self.unet_multiclass_path
        else:
            return self.unet_binary_path

    # ─── ConvLSTM Properties ───
    @property
    def has_convlstm(self) -> bool:
        return self.convlstm_path is not None and self.convlstm_path.exists()

    @property
    def has_pipeline(self) -> bool:
        """U-Net → ConvLSTM pipeline available"""
        return self.has_unet and self.has_convlstm

    # ─── Available Modes ───
    @property
    def available_modes(self) -> List[str]:
        modes = []
        if self.has_binary_unet and self.model_type == "binary":
            modes.append("Binary U-Net (Urban Detection)")
        if self.has_multiclass_unet and self.model_type == "multiclass":
            modes.append("Multiclass U-Net (Land Cover)")
        if self.has_convlstm:
            modes.append("ConvLSTM Future Prediction")
        if self.has_pipeline:
            modes.append("U-Net → ConvLSTM Pipeline")
        return modes

    # ─── Map Properties ───
    @property
    def map_center(self) -> Tuple[float, float]:
        info = KNOWN_MAP_CENTERS.get(self.name, (0.0, 0.0, 12))
        return (info[0], info[1])

    @property
    def map_zoom(self) -> int:
        info = KNOWN_MAP_CENTERS.get(self.name, (0.0, 0.0, 12))
        return info[2]

    @property
    def map_bounds(self) -> Optional[Tuple[List[float], List[float]]]:
        return KNOWN_MAP_BOUNDS.get(self.name)

    # ─── Path Accessors ───
    def image_path(self, year: int) -> Path:
        return self.images[year]

    def mask_path(self, year: int) -> Path:
        return self.masks[year]


def _match_year(filename: str, patterns: list) -> Optional[int]:
    """Try each pattern; return year if matched, else None."""
    for pat in patterns:
        m = pat.match(filename)
        if m:
            return int(m.group(1))
    return None


def _find_model(directory: Path, pattern: re.Pattern) -> Optional[Path]:
    """Return the first file matching pattern, or None."""
    if not directory.exists():
        return None
    for f in sorted(directory.iterdir()):
        if pattern.match(f.name):
            return f
    return None


def _detect_image_prefix(directory: Path) -> str:
    """Detect image naming prefix for path construction."""
    for f in directory.iterdir():
        if re.match(r"sentinel_\w+_\d{4}\.tif$", f.name, re.IGNORECASE):
            m = re.match(r"(sentinel_\w+_)\d{4}\.tif$", f.name, re.IGNORECASE)
            if m:
                return m.group(1)
    return ""


def discover_regions() -> Dict[str, RegionConfig]:
    """
    Walk data/ and build a RegionConfig for every subfolder
    that contains at least one image or mask file.

    Auto-detects binary vs. multiclass U-Net models.
    Respects REGION_SPECS for model_type and num_bands.
    """
    regions: Dict[str, RegionConfig] = {}

    if not DATA_DIR.exists():
        return regions

    for entry in sorted(DATA_DIR.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue

        name = entry.name.lower()

        # Scan for images and masks
        images: Dict[int, Path] = {}
        masks: Dict[int, Path] = {}

        for f in sorted(entry.iterdir()):
            if not f.is_file():
                continue

            # Try image patterns
            year = _match_year(f.name, IMAGE_PATTERNS)
            if year is not None:
                images[year] = f
                continue

            # Try mask pattern
            m = MASK_PATTERN.match(f.name)
            if m:
                masks[int(m.group(1))] = f

        # ─── Find U-Net Models (binary + multiclass) ───
        unet_binary_path = _find_model(entry, UNET_BINARY_PATTERN)
        unet_multiclass_path = _find_model(entry, UNET_MULTICLASS_PATTERN)

        # Fallback: if no specific binary/multiclass, check generic UNET_PATTERN
        if not unet_binary_path and not unet_multiclass_path:
            generic_unet = _find_model(entry, UNET_PATTERN)
            # Assume binary if only one model found
            if generic_unet:
                unet_binary_path = generic_unet

        # Find ConvLSTM model
        convlstm_path = _find_model(entry, CONVLSTM_PATTERN)

        # Detect naming convention
        prefix = _detect_image_prefix(entry)

        # ─── Apply region-specific specs ───
        spec = REGION_SPECS.get(name, {})
        model_type = spec.get("model_type", "binary")
        num_bands = spec.get("num_bands", 11)
        class_names = spec.get("class_names", None)
        class_colors = spec.get("class_colors", None)

        # Only register if there's at least some data
        if images or masks:
            regions[name] = RegionConfig(
                name=name,
                display_name=name.replace("_", " ").title(),
                data_dir=entry,
                images=images,
                masks=masks,
                unet_binary_path=unet_binary_path,
                unet_multiclass_path=unet_multiclass_path,
                convlstm_path=convlstm_path,
                image_prefix=prefix,
                model_type=model_type,
                num_bands=num_bands,
                class_names=class_names,
                class_colors=class_colors,
            )

    return regions


# ─── Module-level singleton ───
REGIONS: Dict[str, RegionConfig] = discover_regions()