"""
models/langsam_model.py

Segmentation via LangSAM (GroundingDINO + SAM).
  - Runs a text prompt per class (e.g. "tree forest canopy")
  - Combines all class masks using priority ordering
  - Zero-shot — no labels needed

Install:
    pip install git+https://github.com/luca-medeiros/lang-segment-anything.git
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .base_model import BaseSegmentationModel


class LangSAMModel(BaseSegmentationModel):

    def __init__(self, config: dict):
        super().__init__(config)
        self.sam_model = None

        # Confidence threshold for GroundingDINO detections
        self.box_threshold  = config.get("langsam", {}).get("box_threshold",  0.30)
        self.text_threshold = config.get("langsam", {}).get("text_threshold", 0.25)

    def load(self) -> None:
        try:
            from lang_sam import LangSAM
        except ImportError:
            raise ImportError(
                "LangSAM not installed.\n"
                "Run: pip install git+https://github.com/luca-medeiros/lang-segment-anything.git"
            )
        print("[LangSAM] Loading model (first run downloads ~2 GB) ...")
        self.sam_model = LangSAM()
        self._loaded = True
        print("[LangSAM] Ready.")

    def predict(self, image: np.ndarray) -> np.ndarray:
        self._ensure_loaded()

        H, W = image.shape[:2]
        # Combined mask: 0 = background/unlabelled
        combined = np.zeros((H, W), dtype=np.int32)

        pil_image = Image.fromarray(image)

        # Process in ascending priority so high-priority classes overwrite
        for cls in self._build_priority_order():
            prompt = cls["prompt"]
            class_id = cls["id"]

            try:
                masks, boxes, phrases, logits = self.sam_model.predict(
                    pil_image,
                    prompt,
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                )
            except Exception as e:
                print(f"[LangSAM] Warning — prompt '{prompt}' failed: {e}")
                continue

            if masks is None or len(masks) == 0:
                continue

            # Each mask is (1, H, W) bool tensor
            for mask_tensor in masks:
                mask_np = mask_tensor.squeeze().cpu().numpy().astype(bool)
                combined[mask_np] = class_id

        return combined
