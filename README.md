# SAM LoRA for Remote Sensing Segmentation

Fine-tune SAM (Segment Anything Model) with LoRA adapters for aerial/satellite
semantic segmentation on ISPRS Potsdam and Vaihingen datasets.

## Project structure

```
.
├── train.py                    # Training script (epoch-based)
├── test.py                     # Evaluation script (mIoU, per-class IoU, OA)
├── train.sh                    # One-command train + eval
├── configs/
│   ├── potsdam.yaml            # Potsdam config
│   └── vaihingen.yaml          # Vaihingen config
├── datasets.py                 # Potsdam / Vaihingen dataset loaders
├── sam_lora_image_encoder.py   # LoRA adapter for SAM image encoder
├── segment_anything_lora/      # Modified SAM (multi-class mask decoder)
├── utils/
│   ├── config.py               # YAML config loader
│   ├── losses.py               # DiceLoss, FocalLoss, etc.
│   └── sam_checkpoint.py       # Auto-download SAM checkpoints
├── pre_weight/                 # SAM checkpoint (auto-downloaded, not tracked)
└── experiments/                # Training outputs (not tracked)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision
pip install -r requirements.txt
```

## Configuration

All settings live in a YAML config file. Copy and edit:

```bash
cp configs/potsdam.yaml configs/my_run.yaml
# edit configs/my_run.yaml — set dataset.root, training.epochs, etc.
```

Key sections in the YAML:

```yaml
experiment:
  name: potsdam_lora          # experiment output folder name
  gpu: "0"

dataset:
  type: potsdam               # potsdam | vaihingen
  root: /path/to/dataset      # MMSeg-format root
  image_size: 1024
  num_classes: 6

model:
  pretrain_model: vit_b       # vit_b | vit_l | vit_h
  sam_checkpoint: null         # null = auto-download
  rank: 4                      # LoRA rank

training:
  epochs: 50
  batch_size: 2
  lr: 1.0e-3
  warmup_epochs: 1
  save_every: 5               # checkpoint every N epochs
  val_every: 5                # validate every N epochs
```

## Quick start

1. Edit `configs/potsdam.yaml` — set `dataset.root` to your data path.

2. Train:

```bash
python train.py --config configs/potsdam.yaml
```

3. Evaluate:

```bash
python test.py --config configs/potsdam.yaml \
               --checkpoint experiments/potsdam_lora/best.pth
```

4. Or train + eval in one go:

```bash
./train.sh configs/potsdam.yaml
```

## CLI overrides

Override any config value from the command line without editing the YAML:

```bash
python train.py --config configs/potsdam.yaml \
    --override training.epochs=100 training.batch_size=4 training.lr=5e-4

./train.sh configs/potsdam.yaml training.epochs=100
```

## SAM checkpoint

The base SAM checkpoint (ViT-B, ~375 MB) is auto-downloaded from Meta on first
run to `pre_weight/`. To use a local file, set `model.sam_checkpoint` in YAML
or pass `--override model.sam_checkpoint=/path/to/sam.pth`.

## How it works

1. SAM's image encoder (ViT-B/L/H) is frozen
2. LoRA adapters are injected into every attention QKV layer
3. The mask decoder is randomly initialised for `num_classes` output channels
4. Only LoRA weights + mask decoder are trained (CE + Dice loss)
5. Best checkpoint is selected by validation mIoU

## Datasets

Both use MMSeg directory layout:

```
root/
  img_dir/train/*.png
  img_dir/val/*.png
  ann_dir/train/*.png       # pixel value = class ID
  ann_dir/val/*.png
```

0 = unlabeled (ignore), 1–6 = impervious surface, building, low vegetation, tree, car, clutter.
