"""
Run evaluation on a trained SAM3 PEFT model and save the confusion matrix
(both raw counts .npy and a matplotlib plot .png / .pdf).

Usage:
  python scripts/compute_confusion_matrix.py \
      --config configs/sam3_adapter_uavid.yaml \
      --checkpoint experiments/sam3_adapter_uavid/.../best.pth \
      --output_dir experiments/confusion_matrices/uavid_adapter
"""

import os
import sys
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Repo imports
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from peft import build_peft_model
from datasets import create_dataset
from utils.config import load_config


def _draw_confusion(ax, cm, class_names):
    """Row-normalised confusion matrix (diagonal = recall)."""
    cm = cm.astype(np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)
    n = cm_norm.shape[0]
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    thresh = 0.5
    for i in range(n):
        for j in range(n):
            v = cm_norm[i, j]
            color = "white" if v > thresh else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=color, fontsize=9)
    return im


def _draw_iou_bars(ax, per_class_iou, class_names, miou):
    """Horizontal IoU bars, same y-order as confusion matrix rows."""
    n = len(class_names)
    y = np.arange(n)
    cmap = plt.get_cmap("Blues")
    colors = [cmap(0.35 + 0.55 * v) for v in per_class_iou]
    ax.barh(y, per_class_iou, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(class_names)
    ax.invert_yaxis()  # match imshow ordering (class 0 at top)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("IoU")
    ax.axvline(miou, color="red", linestyle="--", linewidth=1.2,
               label=f"mIoU = {miou:.3f}")
    for yi, v in zip(y, per_class_iou):
        ax.text(min(v + 0.02, 0.98), yi, f"{v:.2f}",
                va="center", ha="left", fontsize=9)
    ax.legend(loc="lower right", fontsize=9)


def plot_confusion_and_iou(cm, class_names, per_class_iou, miou, title, outfile):
    """Combined figure: row-normalised confusion matrix (left) + per-class IoU bars (right)."""
    n = len(class_names)
    fig, (ax_cm, ax_iou) = plt.subplots(
        1, 2,
        figsize=(max(10, 1.1 * n + 6), max(5, 0.7 * n + 2)),
        gridspec_kw={"width_ratios": [2.2, 1]},
    )
    im = _draw_confusion(ax_cm, cm, class_names)
    fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    ax_cm.set_title("Confusion matrix (row-normalised, diagonal = recall)")

    _draw_iou_bars(ax_iou, per_class_iou, class_names, miou)
    ax_iou.set_title("Per-class IoU")

    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compute & save confusion matrix")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--title", type=str, default=None,
                        help="Title for the plot")
    parser.add_argument("--drop_ignore", action="store_true", default=True,
                        help="Drop ignore class from the plotted matrix")
    cli = parser.parse_args()

    cfg = load_config(cli.config)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.experiment.gpu)

    os.makedirs(cli.output_dir, exist_ok=True)

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
    print(f"Classes: {class_names}  (ignore_index={ignore_index})")

    m_cfg = cfg.model
    method = getattr(m_cfg, "method", "sam3_lora")
    model = build_peft_model(
        sam_model=None,
        method=method,
        num_classes=num_classes,
        image_size=ds_cfg.image_size,
        sam3_checkpoint=getattr(m_cfg, "sam3_checkpoint", None),
        bpe_path=getattr(m_cfg, "bpe_path", None),
        rank=getattr(m_cfg, "rank", 8),
        alpha=getattr(m_cfg, "alpha", 16),
        dropout=getattr(m_cfg, "dropout", 0.0),
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
    use_amp = getattr(cfg.training, "amp", True)

    confusion = np.zeros((num_output_channels, num_output_channels), dtype=np.int64)

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

    # --- Metrics for sanity check ---
    per_class_iou = np.zeros(num_output_channels)
    for c in range(num_output_channels):
        tp = confusion[c, c]
        denom = confusion[:, c].sum() + confusion[c, :].sum() - tp
        per_class_iou[c] = tp / denom if denom > 0 else 0.0
    active_mask = confusion.sum(axis=1) > 0
    active_mask[ignore_index] = False
    miou = per_class_iou[active_mask].mean()
    oa = np.diag(confusion).sum() / max(confusion.sum(), 1)
    print(f"mIoU={miou:.4f}  OA={oa:.4f}")

    # --- Save ---
    np.save(os.path.join(cli.output_dir, "confusion.npy"), confusion)
    with open(os.path.join(cli.output_dir, "classes.txt"), "w") as f:
        for name in class_names:
            f.write(name + "\n")

    # --- Plot: drop ignore + any classes with zero ground-truth pixels ---
    if cli.drop_ignore:
        keep = [c for c in range(num_output_channels)
                if c != ignore_index and confusion[c, :].sum() > 0]
    else:
        keep = list(range(num_output_channels))

    cm_plot = confusion[np.ix_(keep, keep)]
    names_plot = [class_names[c] for c in keep]
    iou_plot = per_class_iou[keep]
    miou_plot = iou_plot.mean() if len(iou_plot) else 0.0

    title = cli.title or f"{ds_cfg.type} / {method}"
    plot_confusion_and_iou(
        cm_plot, names_plot, iou_plot, miou_plot,
        title=title,
        outfile=os.path.join(cli.output_dir, "confusion_matrix.png"),
    )
    print(f"Saved confusion matrix & IoU figure to {cli.output_dir}/")


if __name__ == "__main__":
    main()
