"""
Evaluate a trained SAM3 PEFT model on the test set (unseen holdout).

Reads the same YAML config used for training. Only requires
--checkpoint to point to saved weights.

Always saves 10 random sample visualizations (input | ground truth | prediction)
with dataset class colors to eval_dir/visualizations/.
"""

import os
import random
import argparse

import numpy as np
from PIL import Image as PILImage
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from peft import build_peft_model
from datasets import create_dataset
from utils.config import load_config
from utils.run_log import setup_run_log, timestamp

# Dataset class colors (ignore + 6 ISPRS classes: impervious, building, low veg, tree, car, clutter)
CLASS_PALETTE = [
    (64, 64, 64),    # 0 ignore
    (255, 255, 255), # 1 impervious surface
    (0, 0, 255),    # 2 building
    (0, 255, 0),    # 3 low vegetation
    (0, 128, 0),    # 4 tree
    (255, 255, 0),  # 5 car
    (255, 0, 0),    # 6 clutter
]


def mask_to_rgb(mask: np.ndarray, palette: list, num_classes: int) -> np.ndarray:
    """Convert class index mask [H,W] to RGB [H,W,3] using palette."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(min(num_classes + 1, len(palette))):
        rgb[mask == c] = palette[c]
    # Any out-of-range class id
    for c in range(len(palette), mask.max() + 1):
        rgb[mask == c] = (128, 128, 128)
    return rgb


parser = argparse.ArgumentParser(description="SAM3 PEFT evaluation")
parser.add_argument("--config", type=str, required=True,
                    help="Path to YAML config (same one used for training)")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to trained .pth checkpoint")
parser.add_argument("--split", type=str, default="test",
                    help="Split to evaluate on: test (unseen, default) or val")
parser.add_argument("--save_preds", action="store_true",
                    help="Save prediction PNGs")
parser.add_argument("--output_dir", type=str, default=None,
                    help="Where to save predictions (default: <exp_dir>/predictions)")
cli = parser.parse_args()

cfg = load_config(cli.config)
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.experiment.gpu)

# Timestamped eval run dir so multiple eval runs don't overwrite; full log to file
exp_dir_base = os.path.join(cfg.experiment.output_dir, cfg.experiment.name)
eval_dir = setup_run_log(
    exp_dir_base,
    f"eval_{timestamp()}",
    log_filename="eval.log",
    use_logging=False,
)
print(f"Eval run directory: {eval_dir}")

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

m_cfg = cfg.model
method = getattr(m_cfg, "method", "sam3_lora")
model = build_peft_model(
    sam_model=None,
    method=method,
    num_classes=num_classes,
    image_size=ds_cfg.image_size,
    sam3_checkpoint=getattr(m_cfg, "sam3_checkpoint", None),
    bpe_path=getattr(m_cfg, "bpe_path", None),
    # LoRA kwargs
    rank=getattr(m_cfg, "rank", 8),
    alpha=getattr(m_cfg, "alpha", 16),
    dropout=getattr(m_cfg, "dropout", 0.0),
    # Adapter kwargs
    scale_factor=getattr(m_cfg, "scale_factor", 32),
    input_type=getattr(m_cfg, "input_type", "fft"),
    freq_nums=getattr(m_cfg, "freq_nums", 0.25),
    prompt_type=getattr(m_cfg, "prompt_type", "highpass"),
    tuning_stage=getattr(m_cfg, "tuning_stage", "1234"),
    handcrafted_tune=getattr(m_cfg, "handcrafted_tune", True),
    embedding_tune=getattr(m_cfg, "embedding_tune", True),
    adaptor=getattr(m_cfg, "adaptor", "adaptor"),
).cuda()

model.load_parameters(cli.checkpoint)
print(f"Loaded checkpoint ({method}): {cli.checkpoint}")
model.eval()

multimask_output = num_classes > 2

confusion = np.zeros((num_output_channels, num_output_channels), dtype=np.int64)
sample_idx = 0

pred_dir = cli.output_dir or os.path.join(eval_dir, "predictions")
if cli.save_preds:
    os.makedirs(pred_dir, exist_ok=True)

use_amp = getattr(cfg.training, "amp", True)

with torch.no_grad():
    for batch in tqdm(loader, desc="Evaluating"):
        images = batch["image"].cuda()
        labels = batch["label"].cuda()

        with torch.amp.autocast("cuda", enabled=use_amp):
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

# Visualize 10 random samples (input | ground truth | prediction) with dataset class colors
vis_dir = os.path.join(eval_dir, "visualizations")
os.makedirs(vis_dir, exist_ok=True)
n_vis = min(10, len(dataset))
vis_indices = random.sample(range(len(dataset)), n_vis)
for i, idx in enumerate(vis_indices):
    sample = dataset[idx]
    image = sample["image"].unsqueeze(0).cuda()
    label_np = sample["label"].numpy()
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(image, multimask_output, ds_cfg.image_size)
    pred_np = out["masks"].argmax(dim=1).squeeze(0).cpu().numpy()
    img_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 255).astype(np.uint8)
    gt_rgb = mask_to_rgb(label_np, CLASS_PALETTE, num_output_channels)
    pred_rgb = mask_to_rgb(pred_np, CLASS_PALETTE, num_output_channels)
    composite = np.concatenate([img_np, gt_rgb, pred_rgb], axis=1)
    PILImage.fromarray(composite).save(os.path.join(vis_dir, f"sample_{i:02d}_idx_{idx}.png"))
print(f"Saved {n_vis} visualizations to {vis_dir}/")

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
