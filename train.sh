#!/usr/bin/env bash
# SAM LoRA fine-tuning for remote sensing segmentation.
#
# Usage:
#   ./train.sh DATASET ROOT [GPU]
#
# Examples:
#   ./train.sh potsdam /data/potsdam_mmseg 0
#   ./train.sh vaihingen /data/vaihingen_mmseg 0
#
# The dataset root should be in MMSeg format:
#   root/
#     img_dir/train/*.png
#     img_dir/val/*.png
#     ann_dir/train/*.png
#     ann_dir/val/*.png

set -e
cd "$(dirname "$0")"

DATASET="${1:?Usage: $0 DATASET ROOT [GPU]}"
ROOT="${2:?Usage: $0 DATASET ROOT [GPU]}"
GPU="${3:-0}"

echo "=== SAM LoRA Training ==="
echo "Dataset:  $DATASET"
echo "Root:     $ROOT"
echo "GPU:      $GPU"
echo ""

# Train
python3 train.py \
    --dataset "$DATASET" \
    --root "$ROOT" \
    --gpu "$GPU" \
    --exp "${DATASET}_lora" \
    --num_classes 6 \
    --batch_size 2 \
    --max_iterations 20000 \
    --lr 1e-3 \
    --rank 4 \
    --save_iter 1000 \
    --val_iter 2000

# Evaluate best checkpoint
echo ""
echo "=== Evaluation ==="
python3 test.py \
    --dataset "$DATASET" \
    --root "$ROOT" \
    --gpu "$GPU" \
    --checkpoint "experiments/${DATASET}_lora/best.pth" \
    --save_preds \
    --output_dir "experiments/${DATASET}_lora/predictions"
