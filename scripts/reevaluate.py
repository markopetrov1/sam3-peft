#!/usr/bin/env python3
"""
Protocol-correct re-evaluation of a trained SAM3-PEFT checkpoint.

Motivation
----------
`test.py` computes metrics at the network's working resolution (1008x1008), against a
ground-truth mask that has itself been resized to 1008x1008 with nearest-neighbour
interpolation. For datasets whose native tiles are smaller than 1008 (Potsdam, Vaihingen,
Massachusetts, WHU, LoveDA) this upsamples the ground truth; for UAVid, whose native frames
are 3840x2160, it *downsamples* the ground truth by ~4x and additionally distorts the aspect
ratio. Neither is the protocol used by the published benchmarks.

This script re-scores the same checkpoints without retraining, reporting metrics at BOTH:

  * ``resized``  -- 1008x1008, reproducing the original protocol (for continuity), and
  * ``native``   -- the native ground-truth resolution, obtained by bilinearly upsampling the
                    72x72 logits to the native mask size and taking the argmax there.

It additionally accumulates, at native resolution:

  * a per-image confusion matrix, enabling image-level bootstrap confidence intervals and
    paired tests between methods without repeated training runs, and
  * confusion matrices restricted to a boundary band and to the class interior, which
    quantifies how much of the error is attributable to the effective output stride of 14
    rather than to semantic confusion.

Outputs an ``.npz`` holding the per-image confusion matrices and a JSON summary.

Usage
-----
    python scripts/reevaluate.py --config configs/sam3_lora_potsdam.yaml \
        --checkpoint experiments/.../best.pth --out-dir analysis/lora_potsdam
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Run from anywhere: put the repository root first on sys.path so that `datasets` resolves to
# this repository's datasets.py and not to the HuggingFace `datasets` package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from tqdm import tqdm

from datasets import create_dataset
from peft import build_peft_model
from utils.config import load_config

# Ground-truth pixels within this many pixels of a label boundary are counted as the
# "boundary band"; the rest are the "interior". Measured at native resolution.
BOUNDARY_RADIUS = 3


def boundary_band(label: np.ndarray, radius: int = BOUNDARY_RADIUS) -> np.ndarray:
    """Boolean mask of pixels within `radius` of a change in label value."""
    lab = label.astype(np.uint8)
    k3 = np.ones((3, 3), np.uint8)
    edges = (cv2.dilate(lab, k3) != lab) | (cv2.erode(lab, k3) != lab)
    if radius > 1:
        kr = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
        edges = cv2.dilate(edges.astype(np.uint8), kr).astype(bool)
    return edges


def accumulate(conf: np.ndarray, gt: np.ndarray, pred: np.ndarray, ignore_index: int) -> None:
    """Add one image's (gt, pred) pairs into `conf`, skipping ignored ground truth.

    Uses bincount over the flattened (gt, pred) index rather than np.add.at, which is orders of
    magnitude slower on the millions of pixels a single 4K frame contributes. The result is
    identical; only the speed differs.
    """
    n = conf.shape[0]
    valid = gt != ignore_index
    if not valid.any():
        return
    g = gt[valid].astype(np.int64)
    p = pred[valid].astype(np.int64)
    if g.max(initial=0) >= n or p.max(initial=0) >= n:
        raise ValueError(f"label out of range for a {n}-channel confusion matrix: "
                         f"gt max {g.max(initial=0)}, pred max {p.max(initial=0)}")
    conf += np.bincount(g * n + p, minlength=n * n).reshape(n, n)


def foreground_index(class_names: list) -> int | None:
    """Channel holding the foreground class of a binary benchmark.

    The loaders declare binary datasets as {0: background, 255: building/road}, and the label
    remap sends those to contiguous ids 1 and 2, so the foreground sits at channel 2 and channel 1
    is background. Selecting channel 1 would report the background score, which on these
    background-dominated scenes is far higher than the foreground score.
    """
    named = [(i, str(n).lower()) for i, n in enumerate(class_names)]
    fg = [i for i, n in named if n not in ("ignore", "background", "unused")]
    return fg[0] if len(fg) == 1 else None


def metrics_from_confusion(conf: np.ndarray, ignore_index: int, class_names=None) -> dict:
    """mIoU / OA / per-class IoU, F1 and foreground IoU from a confusion matrix.

    Mirrors test.py: classes absent from the ground truth are dropped from the mean, and the
    ignore channel never contributes even though predictions may fall into it (such
    predictions are counted as errors for the true class, as they are in test.py).
    """
    n = conf.shape[0]
    iou = np.zeros(n)
    f1 = np.zeros(n)
    for c in range(n):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        iou[c] = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        f1[c] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    active = conf.sum(axis=1) > 0
    active[ignore_index] = False
    total = conf.sum()
    fg = foreground_index(class_names) if class_names else None
    return {
        "miou": float(iou[active].mean()) if active.any() else 0.0,
        "oa": float(np.diag(conf).sum() / total) if total > 0 else 0.0,
        "mean_f1": float(f1[active].mean()) if active.any() else 0.0,
        "per_class_iou": {int(c): float(iou[c]) for c in range(n) if active[c]},
        "per_class_f1": {int(c): float(f1[c]) for c in range(n) if active[c]},
        # For binary benchmarks the foreground is the single non-background evaluated class.
        "foreground_class": (class_names[fg] if (fg is not None and class_names) else None),
        "foreground_iou": (float(iou[fg]) if fg is not None and fg < n and active[fg] else None),
        "foreground_f1": (float(f1[fg]) if fg is not None and fg < n and active[fg] else None),
        "n_active_classes": int(active.sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--root", default=None, help="Override dataset.root from the config")
    ap.add_argument("--gpu", default=None, help="Override experiment.gpu")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="Evaluate only the first N images (smoke test)")
    ap.add_argument("--band-swap", default=None,
                    choices=["nir_to_red", "drop_nir", "rgb_shuffle"],
                    help="Input-channel ablation, used as the Vaihingen NIR control")
    ap.add_argument("--score-ignore-channel", action="store_true",
                    help="Score channel 0 as an ordinary class instead of discarding it. On UAVid "
                         "the ignore label is exactly the background-clutter class, and the Dice "
                         "term of the training loss supervises channel 0 on that mask, so channel 0 "
                         "is a de facto clutter predictor. This flag yields the 8-class mIoU that "
                         "the official UAVid protocol uses, whereas the default 7-class score "
                         "excludes clutter.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu if args.gpu is not None else cfg.experiment.gpu)
    os.makedirs(args.out_dir, exist_ok=True)

    ds_cfg = cfg.dataset
    root = args.root or ds_cfg.root
    dataset = create_dataset(
        ds_cfg.type, root=root, split=args.split,
        image_size=ds_cfg.image_size, augment=False,
        exclude_classes=ds_cfg.exclude_classes or None,
    )
    ignore_index = dataset.IGNORE_INDEX
    # When channel 0 is scored as an ordinary class, no pixel is discarded and no channel is
    # excluded from the mean, so the accumulation and metric helpers are given an index that
    # cannot occur in a label map.
    score_ignore = -1 if args.score_ignore_channel else ignore_index
    # Sentinel marking "outside the region of interest" for the boundary/interior split. It must
    # be a value accumulate() drops and one no label can take; when the ignore channel is scored
    # as a class it cannot be the ignore index itself.
    band_sentinel = 255 if args.score_ignore_channel else ignore_index
    num_classes = len(dataset.active_classes)
    n_ch = num_classes + 1
    class_names = ["ignore"] + list(dataset.active_classes.values())

    m_cfg = cfg.model
    method = getattr(m_cfg, "method", "sam3_lora")
    model = build_peft_model(
        sam_model=None, method=method, num_classes=num_classes,
        image_size=ds_cfg.image_size,
        sam3_checkpoint=getattr(m_cfg, "sam3_checkpoint", None),
        bpe_path=getattr(m_cfg, "bpe_path", None),
        rank=getattr(m_cfg, "rank", 8), alpha=getattr(m_cfg, "alpha", 16),
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
    model.load_parameters(args.checkpoint)
    model.eval()

    pairs = dataset.pairs[: args.limit] if args.limit else dataset.pairs
    n_img = len(pairs)

    conf_resized = np.zeros((n_ch, n_ch), dtype=np.int64)
    conf_native = np.zeros((n_ch, n_ch), dtype=np.int64)
    conf_boundary = np.zeros((n_ch, n_ch), dtype=np.int64)
    conf_interior = np.zeros((n_ch, n_ch), dtype=np.int64)
    per_image = np.zeros((n_img, n_ch, n_ch), dtype=np.int64)
    names = []

    torch.cuda.reset_peak_memory_stats()
    t_infer = 0.0

    with torch.no_grad():
        for i in tqdm(range(0, n_img, args.batch_size), desc=os.path.basename(args.out_dir)):
            batch = pairs[i : i + args.batch_size]
            imgs, native_gts = [], []
            for img_path, ann_path in batch:
                pil = PILImage.open(img_path).convert("RGB")
                raw = np.array(PILImage.open(ann_path))
                if raw.ndim == 3:
                    raw = raw[:, :, 0]
                gt = np.full_like(raw, ignore_index, dtype=np.uint8)
                for old_id, new_id in dataset.class_id_remap.items():
                    gt[raw == old_id] = new_id
                native_gts.append(gt)

                arr = np.array(pil.resize((ds_cfg.image_size, ds_cfg.image_size),
                                          PILImage.BILINEAR), dtype=np.float32)
                if args.band_swap == "nir_to_red":
                    # Vaihingen is delivered IRRG: channel 0 is NIR. Replace it with the red
                    # channel, removing all near-infrared information while keeping statistics.
                    arr[:, :, 0] = arr[:, :, 1]
                elif args.band_swap == "drop_nir":
                    arr[:, :, 0] = 0.0
                elif args.band_swap == "rgb_shuffle":
                    arr = arr[:, :, ::-1].copy()
                imgs.append(torch.from_numpy(arr).permute(2, 0, 1))

            x = torch.stack(imgs).cuda(non_blocking=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            # Mixed precision matches the precision the original evaluation ran under
            # (test.py wraps the forward pass in torch.amp.autocast with training.amp true).
            with torch.amp.autocast("cuda", enabled=bool(getattr(cfg.training, "amp", True))):
                out = model(x, num_classes > 2, ds_cfg.image_size)
            torch.cuda.synchronize()
            t_infer += time.perf_counter() - t0

            low = out["low_res_logits"].float()          # [B, C+1, 72, 72]
            pred_resized = out["masks"].argmax(dim=1).cpu().numpy()

            for j, gt_native in enumerate(native_gts):
                names.append(os.path.basename(batch[j][0]))

                # (a) original protocol: ground truth resized to 1008 with nearest neighbour
                gt_resized = np.array(PILImage.fromarray(gt_native).resize(
                    (ds_cfg.image_size, ds_cfg.image_size), PILImage.NEAREST))
                accumulate(conf_resized, gt_resized, pred_resized[j], score_ignore)

                # (b) protocol-correct: logits upsampled to the native ground-truth size
                h, w = gt_native.shape
                logit = F.interpolate(low[j : j + 1], size=(h, w), mode="bilinear",
                                      align_corners=False)
                pred_native = logit.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

                idx = i + j
                accumulate(per_image[idx], gt_native, pred_native, score_ignore)
                conf_native += per_image[idx]

                band = boundary_band(gt_native)
                # Pixels outside the region of interest are set to the same sentinel that
                # accumulate() drops, so that both the out-of-region pixels and (when the ignore
                # channel is not being scored) the ignored pixels are excluded.
                accumulate(conf_boundary, np.where(band, gt_native, band_sentinel),
                           pred_native, band_sentinel)
                accumulate(conf_interior, np.where(~band, gt_native, band_sentinel),
                           pred_native, band_sentinel)

    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "method": method,
        "dataset": ds_cfg.type,
        "split": args.split,
        "band_swap": args.band_swap,
        "n_images": n_img,
        "class_names": class_names,
        "ignore_index": ignore_index,
        "score_ignore_channel": bool(args.score_ignore_channel),
        "resized_1008": metrics_from_confusion(conf_resized, score_ignore, class_names),
        "native": metrics_from_confusion(conf_native, score_ignore, class_names),
        "boundary_band": metrics_from_confusion(conf_boundary, score_ignore, class_names),
        "interior": metrics_from_confusion(conf_interior, score_ignore, class_names),
        "throughput_img_per_s": n_img / t_infer if t_infer > 0 else None,
        "peak_gpu_mib_inference": torch.cuda.max_memory_allocated() / 2**20,
        "checkpoint_bytes": os.path.getsize(args.checkpoint),
    }

    np.savez_compressed(
        os.path.join(args.out_dir, "confusions.npz"),
        per_image=per_image, names=np.array(names),
        conf_resized=conf_resized, conf_native=conf_native,
        conf_boundary=conf_boundary, conf_interior=conf_interior,
        class_names=np.array(class_names),
    )
    with open(os.path.join(args.out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps({k: v for k, v in summary.items()
                      if k in ("method", "dataset", "n_images")}, indent=2))
    print(f"  resized 1008 mIoU: {summary['resized_1008']['miou']:.4f}  "
          f"OA {summary['resized_1008']['oa']:.4f}")
    print(f"  native       mIoU: {summary['native']['miou']:.4f}  "
          f"OA {summary['native']['oa']:.4f}")
    print(f"  boundary     mIoU: {summary['boundary_band']['miou']:.4f}   "
          f"interior mIoU: {summary['interior']['miou']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
