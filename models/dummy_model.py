"""
models/dummy_model.py

A fast, dependency-free model for testing the pipeline end-to-end.
Segments by simple RGB colour thresholding — not accurate, but
useful to verify tiling / stitching / output without waiting for
real model downloads.
"""

from __future__ import annotations

import numpy as np

from .base_model import BaseSegmentationModel


class DummyModel(BaseSegmentationModel):

    def load(self) -> None:
        print("[Dummy] No weights to load — using colour thresholding.")
        self._loaded = True

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Very rough RGB heuristics — purely for pipeline testing."""
        H, W = image.shape[:2]
        r = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        b = image[:, :, 2].astype(np.float32)

        label_map = np.zeros((H, W), dtype=np.int32)  # 0 = background

        # Water: high blue, low red
        water_mask = (b > 120) & (b > r + 20) & (b > g + 10)
        label_map[water_mask] = 7

        # Vegetation (any): green channel dominant
        veg_mask = (g > r + 10) & (g > b + 5)
        label_map[veg_mask] = 1  # vegetation_low default

        # Vegetation high: dark saturated greens (trees cast shadow → darker)
        veg_high = veg_mask & (g < 160) & (g > 60)
        label_map[veg_high] = 3

        # Roads: grey (R≈G≈B, mid-range brightness)
        grey = (np.abs(r - g) < 20) & (np.abs(g - b) < 20)
        road_mask = grey & (r > 80) & (r < 200)
        label_map[road_mask] = 5

        # Buildings: bright, not green
        bright = (r + g + b) > 500
        not_green = g < r + 15
        building_mask = bright & not_green & ~road_mask
        label_map[building_mask] = 4

        # Bare/open ground: brownish, not classified yet
        open_mask = (r > g) & (r > b) & (label_map == 0)
        label_map[open_mask] = 6

        return label_map
