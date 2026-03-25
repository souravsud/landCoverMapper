"""
utils/visualization.py

Converts integer label maps → coloured PNG / overlay images.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def label_map_to_color(
    label_map: np.ndarray,
    classes: list[dict],
    background_color: tuple = (30, 30, 30),
) -> np.ndarray:
    """
    Args:
        label_map : (H, W) int32 with class ids
        classes   : list of class dicts from config (each has 'id' and 'color')

    Returns:
        (H, W, 3) uint8 RGB colour image
    """
    H, W = label_map.shape
    color_map = np.full((H, W, 3), background_color, dtype=np.uint8)

    id_to_color = {c["id"]: tuple(c["color"]) for c in classes}

    for class_id, color in id_to_color.items():
        mask = label_map == class_id
        color_map[mask] = color

    return color_map


def overlay_on_image(
    image: np.ndarray,
    color_map: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Blends original image with coloured label map.
    alpha=0 → original image only; alpha=1 → label map only
    """
    assert 0.0 <= alpha <= 1.0
    blended = (image.astype(np.float32) * (1 - alpha) +
               color_map.astype(np.float32) * alpha).astype(np.uint8)
    return blended


def draw_legend(
    classes: list[dict],
    tile_size: int = 200,
) -> Image.Image:
    """Creates a small legend image showing class colours."""
    row_h = 30
    pad = 10
    h = row_h * (len(classes) + 1) + pad
    w = 300

    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    draw.text((pad, pad // 2), "Land Use Legend", fill=(0, 0, 0), font=font)

    for i, cls in enumerate(classes):
        y = pad + (i + 1) * row_h
        color = tuple(cls["color"])
        draw.rectangle([pad, y, pad + 20, y + 20], fill=color, outline=(0, 0, 0))
        draw.text((pad + 28, y + 3), cls["name"], fill=(0, 0, 0), font=font)

    return img


def save_visualization(
    label_map: np.ndarray,
    original_image: np.ndarray,
    classes: list[dict],
    output_path: str,
    alpha: float = 0.5,
    include_legend: bool = True,
) -> None:
    color_map = label_map_to_color(label_map, classes)
    overlay   = overlay_on_image(original_image, color_map, alpha=alpha)

    result = Image.fromarray(overlay)

    if include_legend:
        legend = draw_legend(classes)
        # Paste legend top-right
        lw, lh = legend.size
        rw, rh = result.size
        if lh < rh and lw < rw:
            result.paste(legend, (rw - lw - 10, 10))

    result.save(output_path)
    print(f"[Viz] Saved visualization → {output_path}")
