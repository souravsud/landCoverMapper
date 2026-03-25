"""
models/sam2_model.py

Segment Anything Model 2 (Meta) backend.

Approach: SAM2 "segment everything" → cluster segments by colour/texture
          → assign class labels to clusters → map back to pixels.

This is a semi-manual step: after first run you inspect the clusters
and assign class ids in config. Subsequent runs use those assignments.

Install:
    pip install git+https://github.com/facebookresearch/segment-anything-2.git
    # Download checkpoint: https://ai.meta.com/sam2/
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from .base_model import BaseSegmentationModel


class SAM2Model(BaseSegmentationModel):

    def __init__(self, config: dict):
        super().__init__(config)
        cfg = config.get("sam2", {})
        self.checkpoint  = cfg.get("checkpoint", "sam2_hiera_large.pt")
        self.model_cfg   = cfg.get("model_cfg",  "sam2_hiera_l.yaml")
        self.n_clusters  = cfg.get("n_clusters",  len(config["classes"]) + 2)
        # Optional: dict mapping cluster_id → class_id, populated after first run
        self.cluster_map: dict[int, int] = cfg.get("cluster_map", {})
        self.predictor = None

    def load(self) -> None:
        try:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        except ImportError:
            raise ImportError(
                "SAM2 not installed.\n"
                "Run: pip install git+https://github.com/facebookresearch/segment-anything-2.git"
            )

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[SAM2] Loading on {device} ...")

        sam2 = build_sam2(self.model_cfg, self.checkpoint, device=device)
        self.predictor = SAM2AutomaticMaskGenerator(sam2)
        self._loaded = True
        print("[SAM2] Ready.")

    def predict(self, image: np.ndarray) -> np.ndarray:
        self._ensure_loaded()

        H, W = image.shape[:2]

        # 1. Generate all segments
        masks_data = self.predictor.generate(image)  # list of dicts
        if not masks_data:
            return np.zeros((H, W), dtype=np.int32)

        # 2. For each segment, compute mean RGB colour
        segment_colours = []
        for m in masks_data:
            seg_mask = m["segmentation"]  # (H, W) bool
            mean_rgb = image[seg_mask].mean(axis=0) if seg_mask.any() else np.array([0, 0, 0])
            segment_colours.append(mean_rgb)

        segment_colours = np.array(segment_colours)

        # 3. Cluster segments by colour → pseudo land-use groups
        n = min(self.n_clusters, len(segment_colours))
        km = KMeans(n_clusters=n, random_state=0, n_init="auto")
        cluster_ids = km.fit_predict(segment_colours)

        # 4. Map cluster → class id (from config, else 0=unknown)
        label_map = np.zeros((H, W), dtype=np.int32)
        for seg_idx, m in enumerate(masks_data):
            seg_mask  = m["segmentation"]
            cluster   = int(cluster_ids[seg_idx])
            class_id  = self.cluster_map.get(cluster, 0)
            label_map[seg_mask] = class_id

        return label_map

    # ------------------------------------------------------------------ #
    #  Helper: after a first run, print cluster colours so user can
    #  fill in cluster_map in config.yaml
    # ------------------------------------------------------------------ #
    def inspect_clusters(self, image: np.ndarray) -> None:
        """Run this once to see what each cluster looks like, then fill config."""
        self._ensure_loaded()
        masks_data = self.predictor.generate(image)
        colours = np.array([
            image[m["segmentation"]].mean(axis=0)
            if m["segmentation"].any() else [0, 0, 0]
            for m in masks_data
        ])
        n = min(self.n_clusters, len(colours))
        km = KMeans(n_clusters=n, random_state=0, n_init="auto")
        km.fit(colours)
        print("[SAM2] Cluster centre colours (R, G, B):")
        for i, centre in enumerate(km.cluster_centers_):
            print(f"  cluster {i}: RGB {centre.astype(int).tolist()}")
        print("Add 'cluster_map: {cluster_id: class_id, ...}' to sam2: in config.yaml")
