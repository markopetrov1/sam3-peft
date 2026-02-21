"""
Evaluate a trained SAM PEFT model on the validation set.

Reads the same YAML config used for training.  Only requires
--checkpoint to point to the saved weights.

Usage:
    python test.py --config configs/lora_potsdam.yaml --checkpoint experiments/lora_potsdam/best.pth
    python test.py --config configs/linear_probing_potsdam.yaml --checkpoint experiments/linear_probing_potsdam/best.pth
    python test.py --config configs/lora_potsdam.yaml --checkpoint experiments/lora_potsdam/best.pth --save_preds
"""

import os
import argparse

import numpy as np
from PIL import Image as PILImage
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from peft import build_peft_model
from segment_anything import sam_model_registry
from datasets import create_dataset
from utils.sam_checkpoint import get_sam_checkpoint
from utils.config import load_config


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="SAM PEFT evaluation")
parser.add_argument("--config", type=str, required=True,
                    help="Path to YAML config (same one used for training)")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to trained .pth checkpoint")
parser.add_argument("--split", type=str, default="val")
parser.add_argument("--save_preds", action="store_true",
                    help="Save prediction PNGs")
parser.add_argument("--output_dir", type=str, default=None,
                    help="Where to save predictions (default: <exp_dir>/predictions)")
cli = parser.parse_args()

cfg = load_config(cli.config)
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.experiment.gpu)


# -------------------------------------------------------------------------
# Data
# -------------------------------------------------------------------------

ds_cfg = cfg.dataset

dataset = create_dataset(
    ds_cfg.type, root=ds_cfg.root, split=cli.split,
    image_size=ds_cfg.image_size, augment=False,
    exclude_classes=ds_cfg.exclude_classes or None,
)
loader = DataLoader(
    dataset, batch_size=cfg.training.batch_size, shuffle=False,
    num_workers=cfg.training.num_workers, pin_memory=True,
)

ignore_index = dataset.IGNORE_INDEX
num_classes = len(dataset.active_classes)
num_output_channels = num_classes + 1
class_names = ["ignore"] + list(dataset.active_classes.values())

print(f"Dataset: {ds_cfg.type} ({cli.split}), {len(dataset)} samples")
print(f"Classes: {class_names}")


# -------------------------------------------------------------------------
# Model
# -------------------------------------------------------------------------

m_cfg = cfg.model

sam_ckpt = get_sam_checkpoint(
    path=m_cfg.sam_checkpoint,
    model_type=m_cfg.pretrain_model,
    download=True,
)

model_sam, _ = sam_model_registry[m_cfg.pretrain_model](
    image_size=ds_cfg.image_size,
    num_classes=num_classes,
    checkpoint=sam_ckpt,
    pixel_mean=[0, 0, 0],
    pixel_std=[1, 1, 1],
)

method = getattr(m_cfg, "method", "lora")

model = build_peft_model(
    model_sam,
    method=method,
    num_classes=num_classes,
    rank=getattr(m_cfg, "rank", 4),
    lora_layer=getattr(m_cfg, "lora_layer", None),
).cuda()

model.load_parameters(cli.checkpoint)
print(f"Loaded checkpoint ({method}): {cli.checkpoint}")
model.eval()

multimask_output = num_classes > 2


# -------------------------------------------------------------------------
# Inference + metrics
# -------------------------------------------------------------------------

confusion = np.zeros((num_output_channels, num_output_channels), dtype=np.int64)
sample_idx = 0

exp_dir = os.path.join(cfg.experiment.output_dir, cfg.experiment.name)
pred_dir = cli.output_dir or os.path.join(exp_dir, "predictions")
if cli.save_preds:
    os.makedirs(pred_dir, exist_ok=True)

with torch.no_grad():
    for batch in tqdm(loader, desc="Evaluating"):
        images = batch["image"].cuda()
        labels = batch["label"].cuda()

        outputs = model(images, multimask_output, ds_cfg.image_size)
        preds = outputs["masks"].argmax(dim=1)

        preds_np = preds.cpu().numpy()
        labels_np = labels.cpu().numpy()

        for b in range(preds_np.shape[0]):
            p, g = preds_np[b].ravel(), labels_np[b].ravel()
            valid = g != ignore_index
            np.add.at(confusion, (g[valid], p[valid]), 1)

            if cli.save_preds:
                PILImage.fromarray(preds_np[b].astype(np.uint8)).save(
                    os.path.join(pred_dir, f"{sample_idx:05d}.png")
                )
            sample_idx += 1


# -------------------------------------------------------------------------
# Compute metrics
# -------------------------------------------------------------------------

per_class_iou = np.zeros(num_output_channels)
for c in range(num_output_channels):
    tp = confusion[c, c]
    denom = confusion[:, c].sum() + confusion[c, :].sum() - tp
    per_class_iou[c] = tp / denom if denom > 0 else 0.0

active_mask = confusion.sum(axis=1) > 0
active_mask[ignore_index] = False
miou = per_class_iou[active_mask].mean()
oa = np.diag(confusion).sum() / max(confusion.sum(), 1)

print("\n" + "=" * 60)
print(f"Results — {ds_cfg.type} ({cli.split}), method={method}")
print("=" * 60)
print(f"{'Class':<25} {'IoU':>8}  {'Pixels':>12}")
print("-" * 60)
for c in range(num_output_channels):
    total = confusion[c, :].sum()
    if total > 0 or c == ignore_index:
        tag = " (ignore)" if c == ignore_index else ""
        print(f"{class_names[c]:<25} {per_class_iou[c]:>8.4f}  {total:>12,}{tag}")
print("-" * 60)
print(f"{'mIoU':<25} {miou:>8.4f}")
print(f"{'Overall Accuracy':<25} {oa:>8.4f}")
print("=" * 60)

if cli.save_preds:
    print(f"\nPredictions saved to {pred_dir}/")
