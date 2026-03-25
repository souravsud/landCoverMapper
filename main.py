"""
main.py — Land Use Mapper
─────────────────────────
Usage:
    python main.py --input /path/to/orthomosaic.png
    python main.py --input /path/to/orthomosaic.png --model dummy   (for testing)
    python main.py --input /path/to/orthomosaic.png --model langsam
    python main.py --input /path/to/orthomosaic.png --config my_config.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from pipeline.preprocessor import ImagePreprocessor
from pipeline.tiler import Tiler
from models import get_model
from utils.visualization import save_visualization
from utils.io import save_label_map, save_per_class_masks


# ─────────────────────────────────────────────────────────────── #

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run(input_path: str, config: dict, model_override: str | None = None) -> None:
    t0 = time.time()

    # Allow --model flag to override config
    if model_override:
        config["active_model"] = model_override

    output_dir = Path(config.get("output_dir", "./output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(input_path).stem

    # ── 1. Load image ──────────────────────────────────────────── #
    print("\n[1/4] Loading image ...")
    preprocessor = ImagePreprocessor(input_path)
    image = preprocessor.load()          # (H, W, 3) uint8
    H, W  = image.shape[:2]
    print(f"      {W} × {H} px")

    # ── 2+3. Tile & Inference ──────────────────────────────────── #
    model = get_model(config)
    model.load()

    if model.handles_full_image:
        # Model manages its own internal tiling with logit-level blending —
        # bypassing the outer Tiler eliminates the extra seam-artifact layer
        # that would otherwise be introduced by blending hard one-hot labels.
        print(f"\n[2/4] Tiling — skipped (model handles full image internally)")
        print(f"\n[3/4] Running model: {config['active_model']} ...")
        label_map = model.predict(image)

        print(f"\n[4/4] Stitching — skipped (done inside model)")

    else:
        # Standard outer-tiler path for models that expect fixed-size crops
        print("\n[2/4] Tiling ...")
        tiler = Tiler(
            tile_size  = config.get("tile_size",  1024),
            overlap    = config.get("overlap",     256),
            blend_mode = config.get("blend_mode", "gaussian"),
        )
        tiles = list(tiler.tile_image(image))
        print(f"      {len(tiles)} tiles "
              f"(tile_size={tiler.tile_size}, overlap={tiler.overlap})")

        print(f"\n[3/4] Running model: {config['active_model']} ...")
        tiles_and_masks = []
        for tile in tqdm(tiles, desc="Inference", unit="tile"):
            mask = model.predict(tile.image)
            tiles_and_masks.append((tile, mask))

        print("\n[4/4] Stitching ...")
        label_map = tiler.stitch(
            tiles_and_masks,
            original_shape=(H, W),
            num_classes=model.num_classes,
        )

    # ── Save outputs ───────────────────────────────────────────── #
    # Use model.get_classes() so models like FlairHub that have a fixed
    # checkpoint-defined class space (different from config["classes"]) are
    # visualised and exported correctly.
    classes = model.get_classes()

    if config.get("save_label_map", True):
        ext = "tif" if config.get("output_format", "tiff") == "tiff" else "png"
        save_label_map(
            label_map,
            output_dir / f"{stem}_labels.{ext}",
            meta=preprocessor.meta,
        )

    if config.get("save_visualization", True):
        save_visualization(
            label_map,
            image,
            classes,
            output_path=str(output_dir / f"{stem}_visualization.png"),
            alpha=0.5,
        )

    if config.get("save_per_class_masks", False):
        save_per_class_masks(label_map, classes, output_dir / f"{stem}_masks")

    elapsed = time.time() - t0
    print(f"\n✓ Done in {elapsed:.1f}s — outputs in {output_dir}/")

    # ── Quick class coverage summary ───────────────────────────── #
    total_px = H * W
    print("\nClass coverage:")
    # Print a background row only when the model's class list does NOT already
    # include id=0 (e.g. LangSAM/dummy where 0 means "unlabelled").
    if not any(c["id"] == 0 for c in classes):
        print(f"  {'background':20s}: {(label_map == 0).sum() / total_px * 100:5.1f}%")
    for cls in classes:
        pct = (label_map == cls["id"]).sum() / total_px * 100
        print(f"  {cls['name']:20s}: {pct:5.1f}%")


# ─────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Land Use Mapper")
    parser.add_argument("--input",  required=True, help="Path to input orthomosaic")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--model",  default=None,
                        help="Override active_model (dummy | langsam | sam2)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run(args.input, cfg, model_override=args.model)
