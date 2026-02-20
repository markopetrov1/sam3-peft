"""
SAM LoRA fine-tuning for remote sensing semantic segmentation.

Epoch-based training driven by a YAML config file.

Usage:
    python train.py --config configs/potsdam.yaml
    python train.py --config configs/vaihingen.yaml
    python train.py --config configs/potsdam.yaml --override training.epochs=100
"""

import os
import sys
import random
import argparse
import logging
import math

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
from tensorboardX import SummaryWriter
from tqdm import tqdm

from sam_lora_image_encoder import LoRA_Sam
from segment_anything_lora import sam_model_registry
from datasets import create_dataset
from utils.losses import DiceLoss
from utils.sam_checkpoint import get_sam_checkpoint
from utils.config import load_config


# -------------------------------------------------------------------------
# CLI (minimal — everything else lives in the YAML)
# -------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="SAM LoRA training")
parser.add_argument("--config", type=str, required=True,
                    help="Path to YAML config file")
parser.add_argument("--override", nargs="*", default=[],
                    help="Override config values, e.g. training.epochs=100 dataset.root=/data/x")
cli = parser.parse_args()

overrides = {}
for item in cli.override:
    key, val = item.split("=", 1)
    # auto-cast numbers and booleans
    for cast in (int, float):
        try:
            val = cast(val)
            break
        except ValueError:
            pass
    if isinstance(val, str):
        if val.lower() == "true":
            val = True
        elif val.lower() == "false":
            val = False
        elif val.lower() == "null" or val.lower() == "none":
            val = None
    overrides[key] = val

cfg = load_config(cli.config, overrides)


# -------------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------------

os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.experiment.gpu)

exp_dir = os.path.join(cfg.experiment.output_dir, cfg.experiment.name)
os.makedirs(exp_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(exp_dir, "train.log"),
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
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


# -------------------------------------------------------------------------
# Data
# -------------------------------------------------------------------------

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


# -------------------------------------------------------------------------
# Model
# -------------------------------------------------------------------------

m_cfg = cfg.model

sam_ckpt = get_sam_checkpoint(
    path=m_cfg.sam_checkpoint,
    model_type=m_cfg.pretrain_model,
    download=True,
)
if sam_ckpt is None:
    logging.warning("No SAM checkpoint — training from random init.")

model_sam, _ = sam_model_registry[m_cfg.pretrain_model](
    image_size=ds_cfg.image_size,
    num_classes=num_classes,
    checkpoint=sam_ckpt,
    pixel_mean=[0, 0, 0],
    pixel_std=[1, 1, 1],
)

model = LoRA_Sam(model_sam, m_cfg.rank).cuda()

if cfg.resume.checkpoint:
    model.load_lora_parameters(cfg.resume.checkpoint)
    logging.info(f"Resumed from {cfg.resume.checkpoint}")

model.train()
multimask_output = num_classes > 2

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
logging.info(f"Parameters — trainable: {trainable:,} / total: {total:,} "
             f"({100 * trainable / total:.2f}%)")


# -------------------------------------------------------------------------
# Optimizer & losses
# -------------------------------------------------------------------------

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
    """Linear warmup → cosine decay."""
    if warmup_steps > 0 and step < warmup_steps:
        return t_cfg.lr * step / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return t_cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# -------------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------------

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


# -------------------------------------------------------------------------
# Training loop
# -------------------------------------------------------------------------

writer = SummaryWriter(os.path.join(exp_dir, "tb_logs"))
start_epoch = cfg.resume.epoch
global_step = start_epoch * steps_per_epoch
best_miou = 0.0

for epoch in range(start_epoch, t_cfg.epochs):
    model.train()
    epoch_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{t_cfg.epochs}", leave=True)
    for step, batch in enumerate(pbar):
        images = batch["image"].cuda()
        labels = batch["label"].cuda()

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

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        grad_clip_norm = getattr(t_cfg, "grad_clip_norm", None)
        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

        optimizer.step()

        global_step += 1
        lr = get_lr(global_step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        epoch_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

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
            path = os.path.join(exp_dir, "best.pth")
            model.save_lora_parameters(path)
            logging.info(f"New best mIoU={best_miou:.4f} → {path}")

    # Save checkpoint
    if (epoch + 1) % t_cfg.save_every == 0 or (epoch + 1) == t_cfg.epochs:
        path = os.path.join(exp_dir, f"epoch_{epoch+1}.pth")
        model.save_lora_parameters(path)
        logging.info(f"Checkpoint → {path}")

path = os.path.join(exp_dir, "last.pth")
model.save_lora_parameters(path)
logging.info(f"Training complete. Best mIoU: {best_miou:.4f}")
writer.close()
