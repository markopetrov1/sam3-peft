# SAM LoRA for Remote Sensing Segmentation

Fine-tune SAM (Segment Anything Model) with LoRA adapters for aerial/satellite
semantic segmentation on ISPRS Potsdam and Vaihingen datasets.

## Project structure

```
.
├── train.py                    # LoRA fine-tuning training script
├── test.py                     # Evaluation script (mIoU, per-class IoU, OA)
├── train.sh                    # One-command train + eval
├── datasets.py                 # Potsdam / Vaihingen dataset loaders
├── sam_lora_image_encoder.py   # LoRA adapter for SAM image encoder
├── segment_anything_lora/      # Modified SAM (multi-class mask decoder)
├── utils/
│   └── losses.py               # DiceLoss, FocalLoss, etc.
├── pre_weight/                 # Place SAM checkpoint here (not tracked)
└── experiments/                # Training outputs (not tracked)
```

## Datasets

Both datasets follow MMSeg directory layout:

```
root/
  img_dir/train/*.png
  img_dir/val/*.png
  ann_dir/train/*.png
  ann_dir/val/*.png
```

Annotations are single-channel PNGs:  0 = unlabeled/ignore, 1-6 = class IDs.

| Dataset   | Classes | GSD  | Patch size |
|-----------|---------|------|------------|
| Potsdam   | 6       | 5 cm | 512×512    |
| Vaihingen | 6       | 9 cm | 512×512    |

**Classes (both):** impervious surface, building, low vegetation, tree, car, clutter

## Setup

```bash
# Optional: create a virtual environment
python -m venv .venv && source .venv/bin/activate  # Linux/macOS

# Install PyTorch (with CUDA if needed), then project deps
pip install torch torchvision  # or use --index-url for CUDA builds
pip install -r requirements.txt
```

## Quick start

1. **Download SAM checkpoint** and place at `pre_weight/sam_vit_b_01ec64.pth`

2. **Train + evaluate:**

```bash
./train.sh potsdam /path/to/potsdam_mmseg 0
./train.sh vaihingen /path/to/vaihingen_mmseg 0
```

3. **Or run scripts individually:**

```bash
# Train
python train.py --dataset potsdam --root /path/to/potsdam_mmseg --gpu 0

# Evaluate
python test.py --dataset potsdam --root /path/to/potsdam_mmseg \
               --checkpoint experiments/potsdam_lora/best.pth --gpu 0 --save_preds
```

## Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | potsdam | `potsdam` or `vaihingen` |
| `--root` | (required) | Path to MMSeg dataset root |
| `--image_size` | 1024 | SAM input resolution |
| `--num_classes` | 6 | Semantic classes (excluding ignore) |
| `--batch_size` | 2 | Batch size |
| `--max_iterations` | 20000 | Training iterations |
| `--lr` | 1e-3 | Peak learning rate (AdamW) |
| `--rank` | 4 | LoRA rank |
| `--val_iter` | 2000 | Validate every N iterations |

## How it works

1. SAM's image encoder (ViT-B) is frozen
2. Low-rank LoRA adapters are injected into every attention QKV layer
3. The mask decoder head is randomly initialised for `num_classes` output
4. Only LoRA weights + mask decoder are trained (CE + Dice loss)
5. Best checkpoint is selected by validation mIoU
