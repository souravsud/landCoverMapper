"""
models/base_model.py

Abstract interface that every segmentation backend must implement.
Swap models by subclassing this and pointing config.yaml → active_model.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np


class BaseSegmentationModel(ABC):
    """
    Contract:
        load()     — download weights / initialise, called once
        predict()  — (H, W, 3) uint8 → (H, W) int32 label map
    """

    def __init__(self, config: dict):
        """
        config : full parsed config dict (see config.yaml)
                 models can read config['classes'], config['active_model'], etc.
        """
        self.config = config
        self.classes = config["classes"]          # list of class dicts
        self.num_classes = len(self.classes) + 1  # +1 for background (id=0)
        self._loaded = False

    # ── Must implement ────────────────────────────────────────────────── #

    @abstractmethod
    def load(self) -> None:
        """Load weights, initialise GPU resources, etc."""
        ...

    @abstractmethod
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Args:
            image : (H, W, 3) uint8 RGB tile

        Returns:
            mask  : (H, W) int32
                    0 = unlabelled / background
                    1..N = class ids as defined in config['classes']
        """
        ...

    # ── Optional override ─────────────────────────────────────────────── #

    #: When True, ``predict()`` accepts the full-resolution image directly and
    #: handles its own internal tiling.  ``main.py`` skips the outer Tiler
    #: entirely for such models, eliminating an extra blending step that would
    #: otherwise introduce visible seam artifacts.
    handles_full_image: bool = False

    def predict_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """
        Default: runs predict() one by one.
        Override for true batched inference if your model supports it.
        """
        return [self.predict(img) for img in images]

    # ── Helpers available to all subclasses ───────────────────────────── #

    def get_classes(self) -> list[dict]:
        """
        Returns the list of class dicts that this model's label map uses.

        Each dict must contain at least:
            id    (int)   – integer value used in the label map
            name  (str)   – human-readable name
            color (list)  – [R, G, B] uint8 color for visualization

        The default implementation returns ``config["classes"]`` (the classes
        defined in config.yaml).  Override this in subclasses whose class
        space is determined by the checkpoint rather than the config
        (e.g. FlairHubModel).
        """
        return self.classes

    def _build_priority_order(self) -> list[dict]:
        """Returns classes sorted lowest priority first (so high-priority overwrites)."""
        return sorted(self.classes, key=lambda c: c.get("priority", 0))

    def _class_by_id(self, class_id: int) -> dict | None:
        for c in self.classes:
            if c["id"] == class_id:
                return c
        return None

    def _ensure_loaded(self):
        if not self._loaded:
            self.load()
            self._loaded = True
