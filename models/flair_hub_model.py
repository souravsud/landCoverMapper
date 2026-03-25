import timm
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
import segmentation_models_pytorch as smp
from models.base_model import BaseSegmentationModel

# Normalization constants from model card
FLAIR_MEAN = np.array([105.66, 111.35, 102.18], dtype=np.float32)
FLAIR_STD  = np.array([52.23,  45.62,  44.30],  dtype=np.float32)

HF_REPO      = "IGNF/FLAIR-HUB_LC-A_RGB_swintiny-upernet"
WEIGHTS_FILE = "FLAIR-HUB_LC-A_RGB_swintiny-upernet.safetensors"

ENCODER_PREFIX = "model.encoders.AERIAL_RGBI.seg_model."
DECODER_PREFIX = "model.main_decoders.AERIAL_LABEL-COSIA.seg_model."

# Must be divisible by patch_size(4) * window_size(7) = 28
# 448 = 16 * 28
TILE_SIZE   = 448
NUM_CLASSES = 19

FLAIR_CLASSES = {
    0:  "building",           1:  "greenhouse",
    2:  "swimming_pool",      3:  "impervious_surface",
    4:  "pervious_surface",   5:  "bare_soil",
    6:  "water",              7:  "snow",
    8:  "herbaceous_vegetation", 9: "agricultural_land",
    10: "plowed_land",        11: "vineyard",
    12: "deciduous",          13: "coniferous",
    14: "brushwood",          15: "clear_cut",
    16: "ligneous",           17: "mixed",
    18: "undefined",
}


class FlairHubModel(BaseSegmentationModel):

    def load(self):
        # Build SMP UperNet with Swin-Tiny encoder
        self.model = smp.UPerNet(
            encoder_name="tu-swin_tiny_patch4_window7_224",
            encoder_weights=None,
            in_channels=3,
            classes=NUM_CLASSES,
        )

        # Replace the internal timm model with dynamic_img_size=True so that
        # Swin recomputes attention masks per forward pass (instead of fixed 224).
        # out_indices must match what SMP's UperNet encoder expects: [0,1,2,3]
        dynamic_swin = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=False,
            features_only=True,
            dynamic_img_size=True,
            out_indices=[0, 1, 2, 3],
        )
        self.model.encoder.model = dynamic_swin

        # Download weights
        print(f"Downloading weights from {HF_REPO} ...")
        weights_path = hf_hub_download(repo_id=HF_REPO, filename=WEIGHTS_FILE)

        # Remap checkpoint keys to SMP layout:
        #   encoder: model.encoders.AERIAL_RGBI.seg_model.model.* -> encoder.model.*
        #   decoder: model.main_decoders.AERIAL_LABEL-COSIA.seg_model.decoder.* -> decoder.*
        #   head:    model.main_decoders.AERIAL_LABEL-COSIA.seg_model.segmentation_head.* -> segmentation_head.*
        remapped = {}
        with safe_open(weights_path, framework="pt") as f:
            for k in f.keys():
                if k.startswith(ENCODER_PREFIX):
                    short = k[len(ENCODER_PREFIX):]   # model.layers_0...
                    remapped["encoder." + short] = f.get_tensor(k)
                elif k.startswith(DECODER_PREFIX):
                    short = k[len(DECODER_PREFIX):]   # decoder.* or segmentation_head.*
                    remapped[short] = f.get_tensor(k)
                # skip criterion.* and fusion_handler.*

        # Skip shape-mismatched keys instead of crashing
        model_state = self.model.state_dict()
        filtered = {}
        skipped  = []
        for k, v in remapped.items():
            if k in model_state and model_state[k].shape != v.shape:
                skipped.append(f"  SKIP {k}: ckpt={v.shape} vs model={model_state[k].shape}")
            else:
                filtered[k] = v

        if skipped:
            print("Shape-mismatched keys skipped (random init):")
            for s in skipped:
                print(s)

        missing, unexpected = self.model.load_state_dict(filtered, strict=False)
        print(f"  Loaded: {len(filtered)} | Missing: {len(missing)} | Unexpected: {len(unexpected)}")

        self.model.eval()
        self.model.to(torch.device("cpu"))
        print("FLAIR-HUB model ready on CPU.")

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Args:
            image: H x W x 3 numpy uint8 array
        Returns:
            H x W integer label map (class indices 0-18)
        """
        h, w = image.shape[:2]
        img  = (image.astype(np.float32) - FLAIR_MEAN) / FLAIR_STD

        stride      = TILE_SIZE // 2   # 50% overlap for smoother borders
        logit_accum = np.zeros((NUM_CLASSES, h, w), dtype=np.float32)
        count_map   = np.zeros((h, w), dtype=np.float32)

        ys    = list(range(0, h, stride))
        xs    = list(range(0, w, stride))
        total = len(ys) * len(xs)
        done  = 0

        for y in ys:
            for x in xs:
                y2, x2 = min(y + TILE_SIZE, h), min(x + TILE_SIZE, w)
                tile   = img[y:y2, x:x2]
                th, tw = tile.shape[:2]

                # Pad to TILE_SIZE if near image edges
                pad_h, pad_w = TILE_SIZE - th, TILE_SIZE - tw
                if pad_h > 0 or pad_w > 0:
                    tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

                tensor = torch.from_numpy(tile.transpose(2, 0, 1)).unsqueeze(0)  # (1,3,448,448)

                with torch.no_grad():
                    logits = self.model(tensor).squeeze(0).numpy()  # (19,448,448)

                logit_accum[:, y:y2, x:x2] += logits[:, :th, :tw]
                count_map[y:y2, x:x2]       += 1.0

                done += 1
                print(f"  Tile {done}/{total}", end="\r")

        print()
        logit_accum /= np.maximum(count_map[np.newaxis], 1)
        return np.argmax(logit_accum, axis=0).astype(np.int64)