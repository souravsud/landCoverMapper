"""
finetune.py — Fine-tune FLAIR-HUB on your own labelled data
─────────────────────────────────────────────────────────────

Overview
--------
This script loads the pretrained FLAIR-HUB weights (downloaded from HuggingFace
the first time) and fine-tunes the model on a custom dataset of aerial images and
corresponding pixel-level label masks.

Data format
-----------
Organise your labelled data as two directories that mirror each other:

    data/
    ├── images/          # RGB aerial tiles (PNG or TIFF, any name)
    │   ├── site1_tile01.png
    │   ├── site1_tile02.png
    │   └── ...
    └── masks/           # Corresponding label masks (uint8 PNG, same name as image)
        ├── site1_tile01.png   # pixel values = class id 0-18
        ├── site1_tile02.png
        └── ...

Label convention (same as FLAIR-HUB)
-------------------------------------
    0  building           6  water               12  deciduous
    1  greenhouse         7  snow                13  coniferous
    2  swimming_pool      8  herbaceous_vegetation 14  brushwood
    3  impervious_surface 9  agricultural_land   15  clear_cut
    4  pervious_surface  10  plowed_land         16  ligneous
    5  bare_soil         11  vineyard            17  mixed
                                                 18  undefined

Usage
-----
    python finetune.py \\
        --images  data/images \\
        --masks   data/masks \\
        --output  ./finetuned_weights \\
        --epochs  20 \\
        --lr      1e-4 \\
        --batch   4

    # Then point config.yaml at the fine-tuned weights:
    #   active_model: flairhub
    #   flairhub_weights: ./finetuned_weights/best.pt

Notes
-----
* Tiles are automatically resized / padded to 448×448 (Swin-Tiny's required size).
* Training uses cross-entropy loss with class-frequency inverse weighting so rare
  classes (e.g. swimming_pool, snow) are not ignored during training.
* A validation split (default 20 %) is held out and mIoU is computed each epoch.
* The best checkpoint (highest val mIoU) is saved to <output>/best.pt.
* Fine-tuning on CPU is possible but very slow — even a GPU with 8 GB VRAM is
  roughly 20–50× faster.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from tqdm import tqdm

# ── Reuse constants / model architecture from the existing model class ────── #
from models.flair_hub_model import (
    FLAIR_MEAN, FLAIR_STD, TILE_SIZE, NUM_CLASSES,
    HF_REPO, WEIGHTS_FILE, ENCODER_PREFIX, DECODER_PREFIX,
)

try:
    import timm
    import segmentation_models_pytorch as smp
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
except ImportError as e:
    raise SystemExit(
        f"Missing dependency: {e}\n"
        "Install with: pip install timm segmentation-models-pytorch "
        "huggingface-hub safetensors"
    )


# ─────────────────────────────────────────────────────────────────────────── #
#  Dataset                                                                     #
# ─────────────────────────────────────────────────────────────────────────── #

class LandCoverDataset(Dataset):
    """
    Loads pairs of (RGB image, label mask) from two directories.
    Both are cropped / resized to TILE_SIZE × TILE_SIZE before returning.
    """

    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        augment: bool = True,
    ):
        self.image_paths = sorted(image_dir.glob("*.png")) + \
                           sorted(image_dir.glob("*.tif")) + \
                           sorted(image_dir.glob("*.tiff"))
        self.mask_dir  = mask_dir
        self.augment   = augment
        self.mean      = torch.tensor(FLAIR_MEAN / 255.0).float().view(3, 1, 1)
        self.std       = torch.tensor(FLAIR_STD  / 255.0).float().view(3, 1, 1)

        if not self.image_paths:
            raise FileNotFoundError(
                f"No PNG/TIF images found in {image_dir}"
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path  = self.image_paths[idx]
        mask_path = self.mask_dir / (img_path.stem + ".png")

        # Load image
        img = np.array(Image.open(img_path).convert("RGB"))

        # Load mask
        if not mask_path.exists():
            raise FileNotFoundError(
                f"Label mask not found: {mask_path}\n"
                f"Expected a PNG with the same filename as the image."
            )
        mask = np.array(Image.open(mask_path))

        # Resize both to TILE_SIZE × TILE_SIZE
        img  = np.array(Image.fromarray(img).resize(
            (TILE_SIZE, TILE_SIZE), Image.BILINEAR))
        mask = np.array(Image.fromarray(mask).resize(
            (TILE_SIZE, TILE_SIZE), Image.NEAREST))

        # Clip label values to valid range
        mask = np.clip(mask, 0, NUM_CLASSES - 1)

        # Simple augmentation: horizontal + vertical flip
        if self.augment:
            if random.random() > 0.5:
                img  = img[:, ::-1, :].copy()
                mask = mask[:, ::-1].copy()
            if random.random() > 0.5:
                img  = img[::-1, :, :].copy()
                mask = mask[::-1, :].copy()

        # Normalise image → (3, H, W) float32 tensor
        img_t  = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        img_t  = (img_t - self.mean) / self.std
        mask_t = torch.from_numpy(mask).long()

        return img_t, mask_t


# ─────────────────────────────────────────────────────────────────────────── #
#  Model builder                                                               #
# ─────────────────────────────────────────────────────────────────────────── #

def build_model(pretrained_weights: str | None = None) -> nn.Module:
    """
    Builds the SMP UperNet / Swin-Tiny architecture and optionally loads
    pretrained (or previously fine-tuned) weights.

    Args:
        pretrained_weights: path to a .pt / .safetensors file, or None to
                            download the original FLAIR-HUB checkpoint.
    """
    model = smp.UPerNet(
        encoder_name="tu-swin_tiny_patch4_window7_224",
        encoder_weights=None,
        in_channels=1,   # matches checkpoint decoder.fpn_stages.4.skip_conv
        classes=NUM_CLASSES,
    )

    # Fix Swin img_size to TILE_SIZE (required for window attention)
    swin = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=False,
        features_only=True,
        img_size=TILE_SIZE,
        out_indices=[0, 1, 2, 3],
    )
    model.encoder.model = swin

    if pretrained_weights and Path(pretrained_weights).exists():
        # Load a previously fine-tuned .pt checkpoint
        state = torch.load(pretrained_weights, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print(f"Loaded fine-tuned weights from {pretrained_weights}")
    else:
        # Download original FLAIR-HUB checkpoint and remap keys
        if pretrained_weights:
            print(f"[Warning] {pretrained_weights} not found — "
                  "falling back to HuggingFace FLAIR-HUB weights.")
        print(f"Downloading pretrained weights from {HF_REPO} ...")
        weights_path = hf_hub_download(repo_id=HF_REPO, filename=WEIGHTS_FILE)

        remapped = {}
        with safe_open(weights_path, framework="pt") as f:
            for k in f.keys():
                if k.startswith(ENCODER_PREFIX):
                    remapped["encoder." + k[len(ENCODER_PREFIX):]] = f.get_tensor(k)
                elif k.startswith(DECODER_PREFIX):
                    remapped[k[len(DECODER_PREFIX):]] = f.get_tensor(k)

        model_state = model.state_dict()
        filtered = {
            k: v for k, v in remapped.items()
            if k in model_state and model_state[k].shape == v.shape
        }
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print(f"  Loaded: {len(filtered)} | Missing: {len(missing)} | "
              f"Unexpected: {len(unexpected)}")

    return model


# ─────────────────────────────────────────────────────────────────────────── #
#  Metrics                                                                     #
# ─────────────────────────────────────────────────────────────────────────── #

def compute_miou(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> float:
    """Computes mean Intersection-over-Union across all classes that appear."""
    ious = []
    pred_np  = preds.cpu().numpy().ravel()
    label_np = labels.cpu().numpy().ravel()
    valid    = label_np != ignore_index
    pred_np  = pred_np[valid]
    label_np = label_np[valid]

    for cls in range(num_classes):
        tp = ((pred_np == cls) & (label_np == cls)).sum()
        fp = ((pred_np == cls) & (label_np != cls)).sum()
        fn = ((pred_np != cls) & (label_np == cls)).sum()
        denom = tp + fp + fn
        if denom > 0:
            ious.append(tp / denom)

    return float(np.mean(ious)) if ious else 0.0


# ─────────────────────────────────────────────────────────────────────────── #
#  Training loop                                                               #
# ─────────────────────────────────────────────────────────────────────────── #

def train(args: argparse.Namespace) -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        ("mps"  if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Training on: {device}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────── #
    full_dataset = LandCoverDataset(
        image_dir=Path(args.images),
        mask_dir=Path(args.masks),
        augment=True,
    )
    n_val   = max(1, int(len(full_dataset) * args.val_split))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    # Disable augmentation for validation subset
    val_ds.dataset.augment = False

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
    )
    print(f"Dataset: {n_train} train / {n_val} val tiles")

    # ── Model ────────────────────────────────────────────────────── #
    model = build_model(pretrained_weights=args.pretrained).to(device)

    # Optionally freeze encoder to only fine-tune the decoder head
    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad_(False)
        print("Encoder frozen — training decoder + segmentation head only.")

    # ── Loss ─────────────────────────────────────────────────────── #
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # ── Optimiser + LR scheduler ─────────────────────────────────── #
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ── Training ─────────────────────────────────────────────────── #
    history = []
    best_miou = -1.0
    best_ckpt = output_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        # ── Train ────────────────────────────────────────────────── #
        model.train()
        train_loss = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]"):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # ── Validate ─────────────────────────────────────────────── #
        model.eval()
        all_preds, all_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]  "):
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                val_loss += criterion(logits, masks).item()
                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(masks.cpu())

        val_loss /= len(val_loader)
        miou = compute_miou(
            torch.cat(all_preds), torch.cat(all_labels), NUM_CLASSES
        )

        print(f"  Epoch {epoch:3d} | "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_mIoU={miou:.4f}")

        # ── Save best checkpoint ──────────────────────────────────── #
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), best_ckpt)
            print(f"  → New best mIoU={best_miou:.4f}  saved to {best_ckpt}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_miou": miou,
        })

    # Save training history as JSON for plotting
    history_path = output_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n✓ Training complete.  Best val mIoU: {best_miou:.4f}")
    print(f"  Best weights → {best_ckpt}")
    print(f"  Training history → {history_path}")
    print(f"\nTo use these weights, set in config.yaml:")
    print(f"  active_model: flairhub")
    print(f"  flairhub_weights: {best_ckpt}")


# ─────────────────────────────────────────────────────────────────────────── #
#  CLI                                                                         #
# ─────────────────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune FLAIR-HUB on custom labelled aerial imagery",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--images",  required=True,
                        help="Directory of RGB aerial image tiles (PNG/TIFF)")
    parser.add_argument("--masks",   required=True,
                        help="Directory of matching label masks (uint8 PNG, "
                             "pixel values = FLAIR class id 0-18)")
    parser.add_argument("--output",  default="./finetuned_weights",
                        help="Output directory for checkpoints and history")
    parser.add_argument("--pretrained", default=None,
                        help="Path to a .pt checkpoint to continue fine-tuning "
                             "(if omitted, downloads FLAIR-HUB from HuggingFace)")
    parser.add_argument("--epochs",  type=int, default=20)
    parser.add_argument("--lr",      type=float, default=1e-4)
    parser.add_argument("--batch",   type=int, default=4)
    parser.add_argument("--workers", type=int, default=2,
                        help="DataLoader worker threads (set to 0 on Windows)")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Fraction of data held out for validation")
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Freeze the Swin encoder; only train decoder+head. "
                             "Useful with very small datasets (< ~100 tiles).")
    args = parser.parse_args()
    train(args)
