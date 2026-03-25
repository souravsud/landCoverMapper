"""
models/__init__.py

Model registry — add new backends here.
config.yaml: active_model: <key>
"""

from .base_model import BaseSegmentationModel
from .dummy_model import DummyModel
from .langsam_model import LangSAMModel
from .sam2_model import SAM2Model


REGISTRY: dict[str, type[BaseSegmentationModel]] = {
    "dummy":   DummyModel,
    "langsam": LangSAMModel,
    "sam2":    SAM2Model,
}


def get_model(config: dict) -> BaseSegmentationModel:
    key = config.get("active_model", "dummy").lower()
    if key not in REGISTRY:
        raise ValueError(
            f"Unknown model '{key}'. Available: {list(REGISTRY.keys())}"
        )
    return REGISTRY[key](config)
