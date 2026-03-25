"""
utils/io.py

Saves the final label map as:
  - GeoTIFF (with georeferencing if source had it)
  - Plain PNG (fallback)
"""

from __future__ import annotations
from pathlib import Path

import numpy as np


def save_label_map(
    label_map: np.ndarray,
    output_path: str | Path,
    meta: dict | None = None,
) -> None:
    """
    Args:
        label_map   : (H, W) int32
        output_path : .tif or .png
        meta        : rasterio metadata dict from preprocessor (or None)
    """
    output_path = Path(output_path)
    suffix = output_path.suffix.lower()

    if suffix in (".tif", ".tiff") and meta is not None:
        _save_geotiff(label_map, output_path, meta)
    else:
        _save_png(label_map, output_path)


def _save_geotiff(label_map: np.ndarray, path: Path, meta: dict) -> None:
    try:
        import rasterio
        meta = meta.copy()
        meta.update({
            "dtype": "uint8",
            "count": 1,
            "height": label_map.shape[0],
            "width":  label_map.shape[1],
        })
        with rasterio.open(path, "w", **meta) as dst:
            dst.write(label_map.astype(np.uint8), 1)
        print(f"[IO] Saved GeoTIFF label map → {path}")
    except ImportError:
        print("[IO] rasterio not available — saving as PNG instead")
        _save_png(label_map, path.with_suffix(".png"))


def _save_png(label_map: np.ndarray, path: Path) -> None:
    from PIL import Image
    img = Image.fromarray(label_map.astype(np.uint8))
    img.save(path)
    print(f"[IO] Saved PNG label map → {path}")


def save_per_class_masks(
    label_map: np.ndarray,
    classes: list[dict],
    output_dir: Path,
) -> None:
    """
    Saves one RGBA PNG per class.

    Each image is the same size as the label map.  Pixels belonging to the
    class are filled with the class color at full opacity; all other pixels
    are fully transparent.  This makes it easy to overlay the masks on the
    original image in any viewer.

    Files are named ``{id:02d}_{name}.png`` so they sort by class ID in the
    file browser.
    """
    from PIL import Image
    output_dir.mkdir(parents=True, exist_ok=True)
    H, W = label_map.shape

    for cls in classes:
        binary = label_map == cls["id"]          # (H, W) bool

        r, g, b = cls["color"]
        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        rgba[binary] = [r, g, b, 255]            # class pixels: opaque color
        # non-class pixels remain [0, 0, 0, 0]   → fully transparent

        fname = f"{cls['id']:02d}_{cls['name']}.png"
        Image.fromarray(rgba, mode="RGBA").save(output_dir / fname)

    print(f"[IO] Per-class masks saved → {output_dir}  ({len(classes)} files)")
