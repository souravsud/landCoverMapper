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

# Semantically-motivated colors for each FLAIR class (RGB uint8).
# These are used for per-class mask PNGs and the color visualization.
FLAIR_COLORS = {
    0:  [220,  60,  60],   # building             – red
    1:  [255, 182, 193],   # greenhouse           – light pink
    2:  [  0, 200, 255],   # swimming_pool        – cyan
    3:  [128, 128, 128],   # impervious_surface   – mid grey
    4:  [180, 160, 120],   # pervious_surface     – light tan
    5:  [139,  90,  43],   # bare_soil            – brown
    6:  [ 30, 100, 220],   # water                – blue
    7:  [230, 230, 255],   # snow                 – pale blue-white
    8:  [144, 238, 144],   # herbaceous_vegetation– light green
    9:  [200, 220, 100],   # agricultural_land    – yellow-green
    10: [100,  70,  30],   # plowed_land          – dark brown
    11: [130,  60, 150],   # vineyard             – purple
    12: [ 34, 139,  34],   # deciduous            – medium green
    13: [  0,  80,  30],   # coniferous           – dark green
    14: [120, 120,  50],   # brushwood            – olive
    15: [200, 180, 120],   # clear_cut            – pale tan
    16: [ 80,  80,  20],   # ligneous             – dark olive
    17: [ 60, 120,  60],   # mixed                – muted green
    18: [ 60,  60,  60],   # undefined            – dark grey
}

# Canonical class list used by get_classes() / save_per_class_masks / visualization.
FLAIR_CLASS_LIST = [
    {"id": idx, "name": name, "color": FLAIR_COLORS[idx]}
    for idx, name in FLAIR_CLASSES.items()
]


class FlairHubModel(BaseSegmentationModel):

    #: predict() accepts the full image and handles its own internal tiling,
    #: so the outer Tiler in main.py is bypassed entirely.
    handles_full_image: bool = True

    def __init__(self, config: dict):
        super().__init__(config)
        # Override: FlairHub's class space is fixed by the checkpoint (19
        # classes, indices 0-18) and is independent of config["classes"].
        # BaseSegmentationModel defaults to len(config["classes"]) + 1, which
        # is wrong here and causes an IndexError in tiler.stitch() when
        # np.eye(num_classes) is indexed with a value up to 18.
        self.num_classes = NUM_CLASSES

    def get_classes(self) -> list[dict]:
        """Return the 19 FLAIR-HUB classes with semantic colors."""
        return FLAIR_CLASS_LIST

    def load(self):
        # Build SMP UperNet with Swin-Tiny encoder.
        # in_channels=1 matches the checkpoint's decoder architecture: the
        # original FLAIR-HUB model was saved with a 1-channel placeholder for
        # the input-level FPN skip connection (decoder.fpn_stages.4.skip_conv).
        # That stage is never exercised during inference (see FPN zip logic), so
        # using 1 here has no effect on accuracy while avoiding a spurious
        # shape-mismatch warning during weight loading.
        # NOTE: The actual encoder model (swin, set below) is created fresh with
        # the default in_chans=3 and processes 3-channel RGB input at runtime.
        # The in_channels=1 here only affects the shape of the single unused
        # decoder weight (decoder.fpn_stages.4.skip_conv), not the inference path.
        self.model = smp.UPerNet(
            encoder_name="tu-swin_tiny_patch4_window7_224",
            encoder_weights=None,
            in_channels=1,   # matches checkpoint's decoder.fpn_stages.4.skip_conv (unused at inference)
            classes=NUM_CLASSES,
        )

        # TILE_SIZE must be divisible by patch_size(4) × window_size(7) = 28.
        # The only value that satisfies this AND is ≥ 224 (Swin's minimum) is
        # a multiple of 28.  Changing it without re-training breaks the model.
        assert TILE_SIZE % 28 == 0, (
            f"TILE_SIZE ({TILE_SIZE}) must be divisible by 28 "
            "(patch_size=4 x window_size=7) for Swin window-attention."
        )

        # Replace the internal timm model with a version fixed to TILE_SIZE so
        # that Swin's window-attention works on 448×448 tiles.
        # This approach (img_size=TILE_SIZE) works with all timm versions ≥0.9
        # and is preferred over dynamic_img_size=True, which is only available in
        # timm ≥1.0. Since all tiles are padded to exactly TILE_SIZE before
        # inference, fixing the size here is always correct.
        # out_indices must match what SMP's UPerNet encoder expects: [0,1,2,3]
        swin = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=False,
            features_only=True,
            img_size=TILE_SIZE,
            out_indices=[0, 1, 2, 3],
        )
        self.model.encoder.model = swin

        # Load weights: prefer a fine-tuned .pt checkpoint (set via
        # config key 'flairhub_weights') over the original HuggingFace release.
        custom_weights = self.config.get("flairhub_weights", None)
        if custom_weights and Path(custom_weights).exists():
            print(f"Loading fine-tuned weights from {custom_weights} ...")
            state = torch.load(custom_weights, map_location="cpu")
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            print(f"  Loaded: {len(state)} | Missing: {len(missing)} | "
                  f"Unexpected: {len(unexpected)}")
        else:
            if custom_weights:
                print(f"[Warning] flairhub_weights={custom_weights!r} not found — "
                      "falling back to HuggingFace pretrained weights.")

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
        Runs inference on an arbitrarily-sized image using overlapping
        448x448 sub-tiles.  Raw logits are accumulated with a 2-D Gaussian
        weight kernel so each pixel's prediction is dominated by sub-tiles
        in which it appears near the centre, eliminating seam artifacts.

        Args:
            image: H x W x 3 numpy uint8 array
        Returns:
            H x W int64 label map (class indices 0-18)
        """
        from scipy.ndimage import gaussian_filter

        h, w = image.shape[:2]
        img  = (image.astype(np.float32) - FLAIR_MEAN) / FLAIR_STD

        # Build a Gaussian weight kernel for one sub-tile (same approach as
        # the outer Tiler so blending is consistent across both tiling levels).
        _gauss_centre = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.float32)
        _gauss_centre[TILE_SIZE // 2, TILE_SIZE // 2] = 1.0
        weight_kernel = gaussian_filter(_gauss_centre, sigma=TILE_SIZE / 6)
        weight_kernel /= weight_kernel.max()   # peak = 1.0

        stride       = TILE_SIZE // 2          # 50 % overlap → 224 px
        logit_accum  = np.zeros((NUM_CLASSES, h, w), dtype=np.float32)
        weight_accum = np.zeros((h, w), dtype=np.float32)

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
                    tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)),
                                  mode="reflect")

                tensor = torch.from_numpy(
                    tile.transpose(2, 0, 1)
                ).unsqueeze(0)  # (1, 3, 448, 448)

                with torch.no_grad():
                    logits = self.model(tensor).squeeze(0).numpy()  # (19, 448, 448)

                # Gaussian-weighted accumulation (crop to valid region only)
                w_crop = weight_kernel[:th, :tw]
                logit_accum[:, y:y2, x:x2] += logits[:, :th, :tw] * w_crop
                weight_accum[y:y2, x:x2]   += w_crop

                done += 1
                print(f"  Sub-tile {done}/{total}", end="\r")

        print()
        weight_accum = np.maximum(weight_accum, 1e-6)
        logit_accum /= weight_accum[np.newaxis]
        return np.argmax(logit_accum, axis=0).astype(np.int64)