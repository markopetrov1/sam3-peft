"""
SAM3 PEFT fine-tuning for remote sensing semantic segmentation.

Supported methods:
    - sam3_lora
    - sam3_linear_probing
    - sam3_adapter

Usage:
    python train.py --config configs/sam3_lora_potsdam.yaml
    python train.py --config configs/sam3_linear_probing_potsdam.yaml
    python train.py --config configs/sam3_adapter_potsdam.yaml
"""

import os
import random
import argparse
import logging
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
from tensorboardX import SummaryWriter
from tqdm import tqdm

from peft import build_peft_model
from datasets import create_dataset
from utils.config import load_config
from utils.losses import DiceLoss
from utils.run_log import setup_run_log, timestamp


parser = argparse.ArgumentParser(description="SAM3 PEFT training")
parser.add_argument("--config", type=str, required=True,
                    help="Path to YAML config file")
parser.add_argument("--gpu", type=str, default=None,
                    help="GPU id (overrides config, e.g. --gpu 1)")
cli = parser.parse_args()

cfg = load_config(cli.config)

os.environ["CUDA_VISIBLE_DEVICES"] = cli.gpu if cli.gpu is not None else str(cfg.experiment.gpu)

exp_dir = setup_run_log(
    cfg.experiment.output_dir,
    f"{cfg.experiment.name}_{timestamp()}",
    log_filename="train.log",
    use_logging=True,
)
logging.info(f"Run directory: {exp_dir}")
logging.info(f"Config:\n{cfg}")

seed = cfg.experiment.seed
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def worker_init_fn(worker_id):
    random.seed(seed + worker_id)

ds_cfg = cfg.dataset

train_ds = create_dataset(
    ds_cfg.type, root=ds_cfg.root, split="train",
    image_size=ds_cfg.image_size, augment=ds_cfg.augment,
    exclude_classes=ds_cfg.exclude_classes or None,
)
val_ds = create_dataset(
    ds_cfg.type, root=ds_cfg.root, split="val",
    image_size=ds_cfg.image_size, augment=False,
    exclude_classes=ds_cfg.exclude_classes or None,
)

t_cfg = cfg.training

train_loader = DataLoader(
    train_ds, batch_size=t_cfg.batch_size, shuffle=True,
    num_workers=t_cfg.num_workers, pin_memory=True,
    worker_init_fn=worker_init_fn, drop_last=True,
)
val_loader = DataLoader(
    val_ds, batch_size=t_cfg.batch_size, shuffle=False,
    num_workers=t_cfg.num_workers, pin_memory=True,
)

ignore_index = train_ds.IGNORE_INDEX
num_classes = len(train_ds.active_classes)
num_output_channels = num_classes + 1

if hasattr(ds_cfg, "num_classes") and ds_cfg.num_classes != num_classes:
    logging.warning(
        f"Config num_classes={ds_cfg.num_classes} does not match dataset active classes={num_classes}. "
        f"Using inferred value: {num_classes}."
    )

steps_per_epoch = len(train_loader)
total_steps = t_cfg.epochs * steps_per_epoch

logging.info(f"Train: {len(train_ds)} samples ({steps_per_epoch} steps/epoch)")
logging.info(f"Val:   {len(val_ds)} samples")
logging.info(f"Epochs: {t_cfg.epochs}, Total steps: {total_steps}")
logging.info(f"Active classes ({num_classes}): {train_ds.active_classes}")


m_cfg = cfg.model
method = getattr(m_cfg, "method", "sam3_lora")
if method not in ("sam3_lora", "sam3_linear_probing", "sam3_adapter"):
    raise ValueError(
        f"Unsupported method '{method}'. Use: sam3_lora, sam3_linear_probing, or sam3_adapter"
    )

model = build_peft_model(
    sam_model=None,
    method=method,
    num_classes=num_classes,
    image_size=ds_cfg.image_size,
    sam3_checkpoint=getattr(m_cfg, "sam3_checkpoint", None),
    bpe_path=getattr(m_cfg, "bpe_path", None),
    # LoRA kwargs (ignored by other methods)
    rank=getattr(m_cfg, "rank", 8),
    alpha=getattr(m_cfg, "alpha", 16),
    dropout=getattr(m_cfg, "dropout", 0.0),
    # Adapter kwargs (ignored by other methods)
    scale_factor=getattr(m_cfg, "scale_factor", 32),
    input_type=getattr(m_cfg, "input_type", "fft"),
    freq_nums=getattr(m_cfg, "freq_nums", 0.25),
    prompt_type=getattr(m_cfg, "prompt_type", "highpass"),
    tuning_stage=getattr(m_cfg, "tuning_stage", "1234"),
    handcrafted_tune=getattr(m_cfg, "handcrafted_tune", True),
    embedding_tune=getattr(m_cfg, "embedding_tune", True),
    adaptor=getattr(m_cfg, "adaptor", "adaptor"),
).cuda()

if cfg.resume.checkpoint:
    model.load_parameters(cfg.resume.checkpoint)
    logging.info(f"Resumed from {cfg.resume.checkpoint}")

model.train()
multimask_output = num_classes > 2

# Reset peak memory stats so we measure only this run
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
training_start_time = time.perf_counter()

# Always log parameter counts before training
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
pct = 100.0 * trainable / total if total else 0.0
logging.info(f"Method: {method}")
logging.info(
    f"Parameters (before training) — trainable: {trainable:,} / total: {total:,} ({pct:.2f}%)"
)
print(f"\nParameters — trainable: {trainable:,} / total: {total:,} ({pct:.2f}%)\n")

use_amp = getattr(t_cfg, "amp", True)
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
logging.info(f"AMP (mixed precision): {'ON' if use_amp else 'OFF'}")

warmup_steps = t_cfg.warmup_epochs * steps_per_epoch

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=t_cfg.lr,
    betas=(0.9, 0.999),
    weight_decay=t_cfg.weight_decay,
)

ce_loss_fn = CrossEntropyLoss(ignore_index=ignore_index)
dice_loss_fn = DiceLoss(num_output_channels)


def get_lr(step):
    """Linear warmup -> cosine decay."""
    if warmup_steps > 0 and step < warmup_steps:
        return t_cfg.lr * step / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return t_cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))

class_names = ["ignore"] + list(train_ds.active_classes.values())

@torch.no_grad()
def validate(epoch):
    model.eval()
    total_inter = torch.zeros(num_output_channels, device="cuda")
    total_union = torch.zeros(num_output_channels, device="cuda")
    total_correct = 0
    total_pixels = 0

    for batch in val_loader:
        images = batch["image"].cuda()
        labels = batch["label"].cuda()

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(images, multimask_output, ds_cfg.image_size)
        preds = outputs["masks"].argmax(dim=1)

        valid = labels != ignore_index
        total_correct += ((preds == labels) & valid).sum().item()
        total_pixels += valid.sum().item()

        for c in range(num_output_channels):
            pred_c = (preds == c) & valid
            label_c = (labels == c) & valid
            total_inter[c] += (pred_c & label_c).sum()
            total_union[c] += (pred_c | label_c).sum()

    ious = total_inter / (total_union + 1e-8)
    active = total_union > 0
    active[ignore_index] = False
    miou = ious[active].mean().item() if torch.any(active) else 0.0
    oa = total_correct / max(total_pixels, 1)

    per_class = {class_names[i]: f"{ious[i].item():.4f}"
                 for i in range(num_output_channels) if active[i]}

    logging.info(f"[Val epoch {epoch}] mIoU: {miou:.4f}, OA: {oa:.4f}")
    logging.info(f"  Per-class IoU: {per_class}")

    model.train()
    return miou, oa

writer = SummaryWriter(os.path.join(exp_dir, "tb_logs"))
start_epoch = cfg.resume.epoch
global_step = start_epoch * steps_per_epoch
best_miou = 0.0
best_oa = 0.0  # OA = Overall Accuracy (pixel-wise); value at epoch where best mIoU was achieved
mem_sum_mb = 0.0
mem_n_samples = 0

# Early stopping: stop if val mIoU does not improve for this many validation checks (0 = disabled)
early_stopping_patience = getattr(t_cfg, "early_stopping_patience", 0)
epochs_without_improvement = 0
stopped_early = False

if early_stopping_patience > 0:
    logging.info(f"Early stopping: patience={early_stopping_patience} (val mIoU)")

for epoch in range(start_epoch, t_cfg.epochs):
    model.train()
    epoch_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{t_cfg.epochs}", leave=True)
    for step, batch in enumerate(pbar):
        images = batch["image"].cuda()
        labels = batch["label"].cuda()

        # ---------- forward (under AMP autocast) ----------
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(images, multimask_output, ds_cfg.image_size)
            masks = outputs["masks"]
            masks_soft = F.softmax(masks, dim=1)

            loss_ce = ce_loss_fn(masks, labels)
            loss_dice = dice_loss_fn(masks_soft, labels.unsqueeze(1))
            loss = 0.5 * (loss_ce + loss_dice)

        if not torch.isfinite(loss):
            logging.warning(
                f"Non-finite loss at epoch={epoch+1}, step={step}. "
                f"Skipping this batch. ce={loss_ce.item():.6f}, dice={loss_dice.item():.6f}"
            )
            optimizer.zero_grad(set_to_none=True)
            continue

        # ---------- backward (AMP-scaled) ----------
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        grad_clip_norm = getattr(t_cfg, "grad_clip_norm", None)
        if grad_clip_norm is not None and grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

        scaler.step(optimizer)
        scaler.update()

        # ---------- LR schedule ----------
        global_step += 1
        lr = get_lr(global_step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        epoch_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

        # Sample GPU memory for average (training-step level)
        if torch.cuda.is_available():
            mem_sum_mb += torch.cuda.memory_allocated() / (1024 ** 2)
            mem_n_samples += 1

        if step % t_cfg.log_every == 0:
            writer.add_scalar("loss/total", loss.item(), global_step)
            writer.add_scalar("loss/ce", loss_ce.item(), global_step)
            writer.add_scalar("loss/dice", loss_dice.item(), global_step)
            writer.add_scalar("lr", lr, global_step)

    avg_loss = epoch_loss / steps_per_epoch
    logging.info(f"Epoch {epoch+1}/{t_cfg.epochs} — avg loss: {avg_loss:.4f}, lr: {lr:.2e}")
    writer.add_scalar("epoch/loss", avg_loss, epoch + 1)

    # Validation
    if (epoch + 1) % t_cfg.val_every == 0 or (epoch + 1) == t_cfg.epochs:
        miou, oa = validate(epoch + 1)
        writer.add_scalar("val/mIoU", miou, epoch + 1)
        writer.add_scalar("val/OA", oa, epoch + 1)
        if miou > best_miou:
            best_miou = miou
            best_oa = oa
            epochs_without_improvement = 0
            path = os.path.join(exp_dir, "best.pth")
            model.save_parameters(path)
            logging.info(f"New best mIoU={best_miou:.4f}, OA={best_oa:.4f} → {path}")
        else:
            epochs_without_improvement += 1

        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            logging.info(
                f"Early stopping at epoch {epoch + 1}: no val mIoU improvement for "
                f"{early_stopping_patience} validation(s). Best mIoU={best_miou:.4f} at earlier epoch."
            )
            stopped_early = True
            break

    # Save checkpoint
    if (epoch + 1) % t_cfg.save_every == 0 or (epoch + 1) == t_cfg.epochs:
        path = os.path.join(exp_dir, f"epoch_{epoch+1}.pth")
        model.save_parameters(path)
        logging.info(f"Checkpoint → {path}")

# When early stopping triggered, restore best checkpoint so last.pth equals best
if stopped_early:
    best_path = os.path.join(exp_dir, "best.pth")
    if os.path.isfile(best_path):
        model.load_parameters(best_path)
        logging.info(f"Restored best checkpoint for final save: {best_path}")

path = os.path.join(exp_dir, "last.pth")
model.save_parameters(path)
writer.close()

# ---------- Training summary ----------
# OA = Overall Accuracy: fraction of (non-ignore) pixels predicted correctly.
training_elapsed_s = time.perf_counter() - training_start_time
peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
avg_mem_mb = mem_sum_mb / mem_n_samples if mem_n_samples else 0.0

def _format_duration(seconds):
    s = int(seconds)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"

completed_epochs = epoch + 1  # last epoch we ran (loop variable is 0-based)
epochs_line = f"Epochs:              {completed_epochs}/{t_cfg.epochs}"
if stopped_early:
    epochs_line += " (early stopping)"
else:
    epochs_line += " (completed)"

summary_lines = [
    "",
    "=" * 60,
    "TRAINING SUMMARY",
    "=" * 60,
    f"Method:              {method}",
    f"Experiment:          {cfg.experiment.name}",
    f"Dataset:             {ds_cfg.type} (train={len(train_ds)}, val={len(val_ds)})",
    epochs_line,
    f"Batch size:          {t_cfg.batch_size}",
    f"Trainable params:    {trainable:,}",
    f"Total params:        {total:,}",
    f"Trainable %:        {pct:.2f}%",
    f"Best mIoU:           {best_miou:.4f}",
    f"Best OA (overall acc): {best_oa:.4f}",
    f"Peak GPU memory:     {peak_mem_mb:.1f} MiB",
    f"Avg GPU memory:      {avg_mem_mb:.1f} MiB",
    f"Total time:          {_format_duration(training_elapsed_s)} ({training_elapsed_s:.1f} s)",
    "=" * 60,
]
for line in summary_lines:
    logging.info(line)
print("\n" + "\n".join(summary_lines) + "\n")

# Write same summary to a dedicated file for downstream scripts
summary_path = os.path.join(exp_dir, "training_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines) + "\n")
logging.info(f"Summary written to {summary_path}")
