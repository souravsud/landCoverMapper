"""
pipeline/preprocessor.py

Loads the input image (PNG or GeoTIFF) and applies any
preprocessing before tiling. Keeps georeferencing metadata
if available so the output can be written as a proper GeoTIFF.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
from PIL import Image


class ImagePreprocessor:
    """
    Loads an image from disk (PNG or GeoTIFF) and exposes:
        .image      → (H, W, 3) uint8 RGB numpy array
        .meta       → dict with georef info if available, else None
        .filepath   → original path
    """

    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        self.image: np.ndarray | None = None
        self.meta: dict | None = None

    def load(self) -> np.ndarray:
        suffix = self.filepath.suffix.lower()

        if suffix in (".tif", ".tiff"):
            self._load_geotiff()
        else:
            self._load_plain_image()

        print(f"[Preprocessor] Loaded {self.filepath.name} "
              f"— shape {self.image.shape}, dtype {self.image.dtype}")
        return self.image

    # ------------------------------------------------------------------ #

    def _load_plain_image(self):
        img = Image.open(self.filepath).convert("RGB")
        self.image = np.array(img, dtype=np.uint8)
        self.meta = None

    def _load_geotiff(self):
        try:
            import rasterio
            with rasterio.open(self.filepath) as src:
                # Read first 3 bands as RGB
                bands = min(src.count, 3)
                data = src.read(list(range(1, bands + 1)))  # (C, H, W)
                self.image = np.moveaxis(data, 0, -1)       # (H, W, C)

                # Ensure uint8
                if self.image.dtype != np.uint8:
                    mn, mx = self.image.min(), self.image.max()
                    self.image = ((self.image - mn) / (mx - mn + 1e-6) * 255).astype(np.uint8)

                if self.image.shape[2] == 1:
                    self.image = np.repeat(self.image, 3, axis=2)

                self.meta = {
                    "crs": src.crs,
                    "transform": src.transform,
                    "driver": "GTiff",
                    "dtype": "uint8",
                    "count": 1,         # label map = single band
                    "height": src.height,
                    "width": src.width,
                    "nodata": 255,
                }
        except ImportError:
            print("[Preprocessor] rasterio not found — falling back to PIL for TIFF")
            self._load_plain_image()

    # ------------------------------------------------------------------ #

    def normalize_float(self, image: np.ndarray) -> np.ndarray:
        """Returns (H, W, 3) float32 in [0, 1]."""
        return image.astype(np.float32) / 255.0

    def to_pil(self, image: np.ndarray | None = None) -> Image.Image:
        img = image if image is not None else self.image
        return Image.fromarray(img)
