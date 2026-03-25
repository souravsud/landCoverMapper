# Land Use Mapper

Modular pipeline for aerial/satellite imagery → land use classification.
Swap models via `config.yaml` without touching anything else.

## Structure

```
landuse_mapper/
├── main.py                  ← entry point
├── config.yaml              ← all settings (tile size, classes, model)
├── requirements.txt
├── pipeline/
│   ├── tiler.py             ← tile + weighted stitch
│   └── preprocessor.py     ← image loading (PNG / GeoTIFF)
├── models/
│   ├── base_model.py        ← interface to implement
│   ├── dummy_model.py       ← colour heuristics (test pipeline fast)
│   ├── langsam_model.py     ← LangSAM zero-shot (recommended)
│   └── sam2_model.py        ← SAM2 segment-everything
└── utils/
    ├── visualization.py     ← coloured overlay + legend
    └── io.py                ← GeoTIFF / PNG output
```

## Setup (WSL / Linux)

```bash
# 1. Base deps
pip install -r requirements.txt

# 2. PyTorch — for AMD GPU (ROCm) on Linux:
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
# For CPU only:
# pip install torch torchvision

# 3. LangSAM (recommended first model)
pip install git+https://github.com/luca-medeiros/lang-segment-anything.git
```

> **AMD GPU note (Beelink SER9):** The Ryzen AI 9 HX 370 has an integrated Radeon 890M.
> ROCm support for integrated AMD GPUs in WSL is limited — if ROCm doesn't work,
> the code falls back to CPU automatically. LangSAM on CPU is slow (~1–2 min/tile)
> but functional. Consider running on a smaller tile of your image first to test.

## Usage

```bash
# Test pipeline quickly (no model download needed)
python main.py --input /path/to/image.png --model dummy

# Run LangSAM (downloads ~2 GB on first run)
python main.py --input /path/to/image.png --model langsam

# Use custom config
python main.py --input /path/to/image.png --config my_config.yaml
```

## Outputs (in ./output/)

| File | Description |
|---|---|
| `*_labels.tif` | Integer label map (class id per pixel) |
| `*_visualization.png` | Colour overlay on original image + legend |
| `*_masks/` | Per-class binary masks (if enabled in config) |

## Adding a New Model

1. Create `models/my_model.py` subclassing `BaseSegmentationModel`
2. Implement `load()` and `predict(image) → label_map`
3. Register it in `models/__init__.py`: `REGISTRY["mymodel"] = MyModel`
4. Set `active_model: mymodel` in `config.yaml`

## Class IDs

| ID | Name |
|---|---|
| 0 | background / unlabelled |
| 1 | vegetation_low |
| 2 | vegetation_medium |
| 3 | vegetation_high |
| 4 | building |
| 5 | road |
| 6 | open_ground |
| 7 | water |

Edit classes freely in `config.yaml` — the pipeline reads them dynamically.
