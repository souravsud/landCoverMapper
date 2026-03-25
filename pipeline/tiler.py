"""
pipeline/tiler.py

Handles:
  - Splitting a large image into overlapping tiles
  - Stitching predicted masks back using weighted blending
    (centre of each tile gets full weight; edges taper off so
     seams don't appear in the final map)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Generator

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class Tile:
    image: np.ndarray       # (H, W, 3) uint8
    x: int                  # top-left col in original image
    y: int                  # top-left row in original image
    tile_w: int             # actual tile width  (may be < tile_size at edges)
    tile_h: int             # actual tile height (may be < tile_size at edges)
    pad_right: int          # pixels padded on right  (to reach tile_size)
    pad_bottom: int         # pixels padded on bottom (to reach tile_size)


class Tiler:
    def __init__(self, tile_size: int = 1024, overlap: int = 128,
                 blend_mode: str = "gaussian"):
        assert overlap < tile_size, "overlap must be smaller than tile_size"
        self.tile_size = tile_size
        self.overlap = overlap
        self.blend_mode = blend_mode

    # ------------------------------------------------------------------ #
    #  Forward pass: image → tiles                                         #
    # ------------------------------------------------------------------ #
    def tile_image(self, image: np.ndarray) -> Generator[Tile, None, None]:
        """
        Yields Tile objects covering the full image with overlap.
        Each tile is padded to exactly (tile_size, tile_size) so model
        input is always uniform.
        """
        H, W = image.shape[:2]
        stride = self.tile_size - self.overlap

        ys = list(range(0, H, stride))
        xs = list(range(0, W, stride))

        # Ensure last tile reaches the image edge
        if ys[-1] + self.tile_size < H:
            ys.append(H - self.tile_size)
        if xs[-1] + self.tile_size < W:
            xs.append(W - self.tile_size)

        for y in ys:
            for x in xs:
                y = max(0, min(y, H - 1))
                x = max(0, min(x, W - 1))

                y1 = y
                y2 = min(y + self.tile_size, H)
                x1 = x
                x2 = min(x + self.tile_size, W)

                crop = image[y1:y2, x1:x2]
                tile_h, tile_w = crop.shape[:2]

                # Pad to tile_size x tile_size (bottom-right padding)
                pad_bottom = self.tile_size - tile_h
                pad_right  = self.tile_size - tile_w

                if pad_bottom > 0 or pad_right > 0:
                    crop = np.pad(
                        crop,
                        ((0, pad_bottom), (0, pad_right), (0, 0)),
                        mode="reflect"
                    )

                yield Tile(
                    image=crop,
                    x=x1, y=y1,
                    tile_w=tile_w, tile_h=tile_h,
                    pad_right=pad_right, pad_bottom=pad_bottom
                )

    def count_tiles(self, image_shape: tuple[int, int]) -> int:
        """Returns how many tiles will be produced (useful for progress bars)."""
        H, W = image_shape[:2]
        stride = self.tile_size - self.overlap
        ny = len(range(0, H, stride))
        nx = len(range(0, W, stride))
        return ny * nx

    # ------------------------------------------------------------------ #
    #  Inverse pass: predicted masks → stitched output                    #
    # ------------------------------------------------------------------ #
    def stitch(
        self,
        tiles_and_masks: list[tuple[Tile, np.ndarray]],
        original_shape: tuple[int, int],
        num_classes: int,
    ) -> np.ndarray:
        """
        Combines per-tile predicted masks into a full-resolution label map.

        Args:
            tiles_and_masks : list of (Tile, mask) where mask is (H, W) int32
            original_shape  : (H, W) of the original image
            num_classes     : total number of classes (including 0=background)

        Returns:
            label_map : (H, W) int32 — argmax class id per pixel
        """
        H, W = original_shape
        # Accumulate soft scores per class
        score_map = np.zeros((H, W, num_classes), dtype=np.float32)
        weight_map = np.zeros((H, W), dtype=np.float32)

        weight_kernel = self._make_weight_kernel(self.tile_size, self.blend_mode)

        for tile, mask in tiles_and_masks:
            # Strip padding from mask
            valid_mask = mask[:tile.tile_h, :tile.tile_w]
            valid_w    = weight_kernel[:tile.tile_h, :tile.tile_w]

            y1, y2 = tile.y, tile.y + tile.tile_h
            x1, x2 = tile.x, tile.x + tile.tile_w

            # Convert int label map → one-hot, then weight-accumulate
            one_hot = np.eye(num_classes, dtype=np.float32)[valid_mask]  # (H,W,C)
            score_map[y1:y2, x1:x2] += one_hot * valid_w[:, :, np.newaxis]
            weight_map[y1:y2, x1:x2] += valid_w

        # Normalise and take argmax
        weight_map = np.maximum(weight_map, 1e-6)
        score_map /= weight_map[:, :, np.newaxis]
        label_map = np.argmax(score_map, axis=-1).astype(np.int32)
        return label_map

    # ------------------------------------------------------------------ #
    #  Weight kernel                                                       #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _make_weight_kernel(size: int, mode: str = "gaussian") -> np.ndarray:
        """
        Returns a (size, size) float32 weight kernel.
        Pixels near the centre get weight ~1.0, edges taper toward 0.
        """
        if mode == "hard":
            return np.ones((size, size), dtype=np.float32)

        elif mode == "linear":
            ramp = np.minimum(
                np.arange(1, size + 1),
                np.arange(size, 0, -1)
            ).astype(np.float32)
            kernel = np.outer(ramp, ramp)
            kernel /= kernel.max()
            return kernel

        else:  # gaussian (default)
            centre = np.zeros((size, size), dtype=np.float32)
            mid = size // 2
            centre[mid, mid] = 1.0
            kernel = gaussian_filter(centre, sigma=size / 6)
            kernel /= kernel.max()
            return kernel
