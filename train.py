"""
SAM LoRA fine-tuning for remote sensing semantic segmentation.

Trains LoRA adapters on the SAM image encoder while jointly training
the mask decoder. Supports Potsdam and Vaihingen datasets.

Usage:
    python train.py --dataset potsdam --root /data/potsdam_mmseg --gpu 0
    python train.py --dataset vaihingen --root /data/vaihingen_mmseg --exp vai_lora
"""

import os
import sys
import random
import argparse
import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
from tensorboardX import SummaryWriter

from sam_lora_image_encoder import LoRA_Sam
from segment_anything_lora import sam_model_registry
from datasets import create_dataset, DATASET_REGISTRY
from utils.losses import DiceLoss
from utils.sam_checkpoint import get_sam_checkpoint


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="SAM LoRA fine-tuning for remote sensing")
parser.add_argument("--dataset", type=str, default="potsdam",
                    choices=list(DATASET_REGISTRY.keys()),
                    help="Dataset name")
parser.add_argument("--root", type=str, required=True,
                    help="Path to MMSeg-format dataset root")
parser.add_argument("--exp", type=str, default="sam_lora",
                    help="Experiment name (used for output directory)")
parser.add_argument("--output_dir", type=str, default="experiments",
                    help="Parent directory for experiment outputs")

parser.add_argument("--image_size", type=int, default=1024,
                    help="SAM input resolution")
parser.add_argument("--num_classes", type=int, default=6,
                    help="Number of semantic classes (excluding ignore)")
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--num_workers", type=int, default=4)

parser.add_argument("--max_iterations", type=int, default=20000)
parser.add_argument("--lr", type=float, default=1e-3,
                    help="Peak learning rate (AdamW)")
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--warmup_iters", type=int, default=250,
                    help="Linear warmup iterations (0 to disable)")

parser.add_argument("--rank", type=int, default=4, help="LoRA rank")
parser.add_argument("--pretrain_model", type=str, default="vit_b",
                    choices=["vit_b", "vit_l", "vit_h"])
parser.add_argument("--sam_checkpoint", type=str, default=None,
                    help="Path to pretrained SAM checkpoint (default: auto-download to pre_weight/)")

parser.add_argument("--save_iter", type=int, default=1000,
                    help="Save checkpoint every N iterations")
parser.add_argument("--log_iter", type=int, default=50,
                    help="Log metrics every N iterations")
parser.add_argument("--val_iter", type=int, default=2000,
                    help="Run validation every N iterations")

parser.add_argument("--load", type=str, default=None,
                    help="Path to LoRA checkpoint to resume from")
parser.add_argument("--load_iter", type=int, default=0,
                    help="Iteration number of loaded checkpoint")

parser.add_argument("--seed", type=int, default=1337)
parser.add_argument("--gpu", type=str, default="0")
args = parser.parse_args()


# -------------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------------

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

snapshot_path = os.path.join(args.output_dir, args.exp)
os.makedirs(snapshot_path, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(snapshot_path, "train.log"),
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info(f"Args: {args}")

cudnn.benchmark = False
cudnn.deterministic = True
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


# -------------------------------------------------------------------------
# Data
# -------------------------------------------------------------------------

train_ds = create_dataset(
    args.dataset, root=args.root, split="train",
    image_size=args.image_size, augment=True,
)
val_ds = create_dataset(
    args.dataset, root=args.root, split="val",
    image_size=args.image_size, augment=False,
)

train_loader = DataLoader(
    train_ds, batch_size=args.batch_size, shuffle=True,
    num_workers=args.num_workers, pin_memory=True,
    worker_init_fn=worker_init_fn, drop_last=True,
)
val_loader = DataLoader(
    val_ds, batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_workers, pin_memory=True,
)

ignore_index = train_ds.IGNORE_INDEX
num_output_channels = args.num_classes + 1  # +1 for the extra mask token

logging.info(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")
logging.info(f"Classes: {args.num_classes}, Ignore index: {ignore_index}, "
             f"Output channels: {num_output_channels}")


# -------------------------------------------------------------------------
# Model
# -------------------------------------------------------------------------

sam_checkpoint_path = get_sam_checkpoint(
    path=args.sam_checkpoint,
    model_type=args.pretrain_model,
    download=True,
)
if sam_checkpoint_path is None:
    logging.warning("No SAM checkpoint loaded; image encoder will train from random init.")

model_sam, img_embedding_size = sam_model_registry[args.pretrain_model](
    image_size=args.image_size,
    num_classes=args.num_classes,
    checkpoint=sam_checkpoint_path,
    pixel_mean=[0, 0, 0],
    pixel_std=[1, 1, 1],
)

model = LoRA_Sam(model_sam, args.rank).cuda()

if args.load:
    model.load_lora_parameters(args.load)
    logging.info(f"Loaded LoRA checkpoint from {args.load}")

model.train()

multimask_output = args.num_classes > 2

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
logging.info(f"Trainable: {trainable_params:,} / Total: {total_params:,} "
             f"({100 * trainable_params / total_params:.2f}%)")


# -------------------------------------------------------------------------
# Optimizer & losses
# -------------------------------------------------------------------------

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=args.lr if args.warmup_iters == 0 else args.lr / args.warmup_iters,
    betas=(0.9, 0.999),
    weight_decay=args.weight_decay,
)

ce_loss_fn = CrossEntropyLoss(ignore_index=ignore_index)
dice_loss_fn = DiceLoss(num_output_channels)


# -------------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, iteration):
    model.eval()
    total_inter = torch.zeros(num_output_channels, device="cuda")
    total_union = torch.zeros(num_output_channels, device="cuda")
    total_correct = 0
    total_pixels = 0

    for batch in loader:
        images = batch["image"].cuda()
        labels = batch["label"].cuda()

        outputs = model(images, multimask_output, args.image_size)
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
    active_mask = total_union > 0
    miou = ious[active_mask].mean().item()
    oa = total_correct / max(total_pixels, 1)

    class_names = ["ignore"] + list(train_ds.active_classes.values())
    per_class = {class_names[i]: f"{ious[i].item():.4f}"
                 for i in range(num_output_channels) if active_mask[i]}

    logging.info(f"[Val @ iter {iteration}] mIoU: {miou:.4f}, OA: {oa:.4f}")
    logging.info(f"  Per-class IoU: {per_class}")

    model.train()
    return miou, oa


# -------------------------------------------------------------------------
# Training loop
# -------------------------------------------------------------------------

writer = SummaryWriter(os.path.join(snapshot_path, "tb_logs"))
logging.info(f"{len(train_loader)} iterations per epoch")

iter_num = args.load_iter
best_miou = 0.0

for epoch in range(99999):
    for batch in train_loader:
        images = batch["image"].cuda()
        labels = batch["label"].cuda()

        outputs = model(images, multimask_output, args.image_size)
        masks = outputs["masks"]
        masks_soft = F.softmax(masks, dim=1)

        loss_ce = ce_loss_fn(masks, labels)
        loss_dice = dice_loss_fn(masks_soft, labels.unsqueeze(1))
        loss = 0.5 * (loss_ce + loss_dice)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # LR schedule: linear warmup then polynomial decay
        iter_num += 1
        if args.warmup_iters > 0 and iter_num <= args.warmup_iters:
            lr = args.lr * (iter_num / args.warmup_iters)
        else:
            progress = (iter_num - args.warmup_iters) / max(args.max_iterations - args.warmup_iters, 1)
            lr = args.lr * (1.0 - progress) ** 0.9
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Logging
        if iter_num % args.log_iter == 0:
            logging.info(
                f"iter {iter_num}/{args.max_iterations} | "
                f"loss={loss.item():.4f} ce={loss_ce.item():.4f} "
                f"dice={loss_dice.item():.4f} lr={lr:.6f}"
            )
            writer.add_scalar("loss/total", loss.item(), iter_num)
            writer.add_scalar("loss/ce", loss_ce.item(), iter_num)
            writer.add_scalar("loss/dice", loss_dice.item(), iter_num)
            writer.add_scalar("lr", lr, iter_num)

        # Validation
        if iter_num % args.val_iter == 0:
            miou, oa = validate(model, val_loader, iter_num)
            writer.add_scalar("val/mIoU", miou, iter_num)
            writer.add_scalar("val/OA", oa, iter_num)
            if miou > best_miou:
                best_miou = miou
                path = os.path.join(snapshot_path, "best.pth")
                model.save_lora_parameters(path)
                logging.info(f"New best mIoU={best_miou:.4f}, saved to {path}")

        # Save checkpoint
        if iter_num % args.save_iter == 0:
            path = os.path.join(snapshot_path, f"iter_{iter_num}.pth")
            model.save_lora_parameters(path)
            logging.info(f"Saved checkpoint to {path}")

        if iter_num >= args.max_iterations:
            break
    if iter_num >= args.max_iterations:
        break

# Final save
path = os.path.join(snapshot_path, f"iter_{iter_num}.pth")
model.save_lora_parameters(path)
logging.info(f"Training complete. Final checkpoint: {path}")
logging.info(f"Best validation mIoU: {best_miou:.4f}")
writer.close()
