"""
Evaluate a trained SAM LoRA model on the validation set.

Computes mIoU, per-class IoU, overall accuracy (OA), and optionally
saves prediction PNGs.

Usage:
    python test.py --dataset potsdam --root /data/potsdam_mmseg \
                   --checkpoint experiments/sam_lora/best.pth --gpu 0
"""

import os
import sys
import argparse

import numpy as np
from PIL import Image as PILImage
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from sam_lora_image_encoder import LoRA_Sam
from segment_anything_lora import sam_model_registry
from datasets import create_dataset, DATASET_REGISTRY
from utils.sam_checkpoint import get_sam_checkpoint


parser = argparse.ArgumentParser(description="SAM LoRA evaluation for remote sensing")
parser.add_argument("--dataset", type=str, default="potsdam",
                    choices=list(DATASET_REGISTRY.keys()))
parser.add_argument("--root", type=str, required=True)
parser.add_argument("--split", type=str, default="val")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to LoRA .pth checkpoint")

parser.add_argument("--image_size", type=int, default=1024)
parser.add_argument("--num_classes", type=int, default=6)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--num_workers", type=int, default=4)

parser.add_argument("--rank", type=int, default=4)
parser.add_argument("--pretrain_model", type=str, default="vit_b",
                    choices=["vit_b", "vit_l", "vit_h"])
parser.add_argument("--sam_checkpoint", type=str, default=None,
                    help="Path to base SAM weights (default: auto-download to pre_weight/)")

parser.add_argument("--save_preds", action="store_true",
                    help="Save prediction PNGs to disk")
parser.add_argument("--output_dir", type=str, default="predictions")
parser.add_argument("--gpu", type=str, default="0")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


# -------------------------------------------------------------------------
# Data
# -------------------------------------------------------------------------

dataset = create_dataset(
    args.dataset, root=args.root, split=args.split,
    image_size=args.image_size, augment=False,
)
loader = DataLoader(
    dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_workers, pin_memory=True,
)

ignore_index = dataset.IGNORE_INDEX
num_output_channels = args.num_classes + 1
class_names = ["ignore"] + list(dataset.active_classes.values())

print(f"Dataset: {args.dataset} ({args.split}), {len(dataset)} samples")
print(f"Classes: {class_names}")


# -------------------------------------------------------------------------
# Model
# -------------------------------------------------------------------------

sam_checkpoint_path = get_sam_checkpoint(
    path=args.sam_checkpoint,
    model_type=args.pretrain_model,
    download=True,
)

model_sam, _ = sam_model_registry[args.pretrain_model](
    image_size=args.image_size,
    num_classes=args.num_classes,
    checkpoint=sam_checkpoint_path,
    pixel_mean=[0, 0, 0],
    pixel_std=[1, 1, 1],
)

model = LoRA_Sam(model_sam, args.rank).cuda()
model.load_lora_parameters(args.checkpoint)
print(f"Loaded LoRA checkpoint: {args.checkpoint}")
model.eval()

multimask_output = args.num_classes > 2


# -------------------------------------------------------------------------
# Inference + metrics
# -------------------------------------------------------------------------

confusion = np.zeros((num_output_channels, num_output_channels), dtype=np.int64)
sample_idx = 0

if args.save_preds:
    os.makedirs(args.output_dir, exist_ok=True)

with torch.no_grad():
    for batch in tqdm(loader, desc="Evaluating"):
        images = batch["image"].cuda()
        labels = batch["label"].cuda()

        outputs = model(images, multimask_output, args.image_size)
        preds = outputs["masks"].argmax(dim=1)  # [B, H, W]

        preds_np = preds.cpu().numpy()
        labels_np = labels.cpu().numpy()

        for b in range(preds_np.shape[0]):
            p, g = preds_np[b].ravel(), labels_np[b].ravel()
            valid = g != ignore_index
            p, g = p[valid], g[valid]
            np.add.at(confusion, (g, p), 1)

            if args.save_preds:
                pred_img = PILImage.fromarray(preds_np[b].astype(np.uint8))
                pred_img.save(os.path.join(args.output_dir, f"{sample_idx:05d}.png"))
            sample_idx += 1


# -------------------------------------------------------------------------
# Compute metrics
# -------------------------------------------------------------------------

per_class_iou = np.zeros(num_output_channels)
for c in range(num_output_channels):
    tp = confusion[c, c]
    fp = confusion[:, c].sum() - tp
    fn = confusion[c, :].sum() - tp
    denom = tp + fp + fn
    per_class_iou[c] = tp / denom if denom > 0 else 0.0

active_mask = confusion.sum(axis=1) > 0
active_mask[ignore_index] = False  # exclude ignore class from mIoU
miou = per_class_iou[active_mask].mean()
oa = np.diag(confusion).sum() / max(confusion.sum(), 1)

print("\n" + "=" * 60)
print(f"Results on {args.dataset} ({args.split})")
print("=" * 60)
print(f"{'Class':<25} {'IoU':>8}  {'Pixels':>12}")
print("-" * 60)
for c in range(num_output_channels):
    total = confusion[c, :].sum()
    if total > 0 or c == ignore_index:
        marker = " (ignore)" if c == ignore_index else ""
        print(f"{class_names[c]:<25} {per_class_iou[c]:>8.4f}  {total:>12,}{marker}")
print("-" * 60)
print(f"{'mIoU':<25} {miou:>8.4f}")
print(f"{'Overall Accuracy':<25} {oa:>8.4f}")
print("=" * 60)

if args.save_preds:
    print(f"\nPredictions saved to {args.output_dir}/")
