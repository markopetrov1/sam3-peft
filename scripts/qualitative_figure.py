#!/usr/bin/env python3
"""
Build a qualitative comparison figure: image | ground truth | linear probing | LoRA | adapter.

Predictions are produced by upsampling each model's 72x72 logits to the native ground-truth
resolution, which is the same protocol `scripts/reevaluate.py` reports, so the panels agree with
the tables. Inference only -- nothing is trained here.

Tiles can be chosen by name, or automatically: `--select disagreement` picks the tiles where the
three methods disagree most (the informative cases for a method comparison), `--select worst`
picks the tiles with the lowest LoRA IoU (failure cases), and `--select random` samples uniformly.

Usage
-----
    python scripts/qualitative_figure.py --dataset potsdam \
        --root /path/to/potsdam_mmseg --out figures/qualitative_potsdam.pdf --n 4
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import create_dataset          # noqa: E402
from peft import build_peft_model            # noqa: E402
from utils.config import load_config         # noqa: E402

METHODS = [("sam3_linear_probing", "Linear probing"),
           ("sam3_lora", "LoRA"),
           ("sam3_adapter", "Adapter")]

# Per-dataset display palettes, index 0 = ignore.
PALETTES = {
    "potsdam": [(60, 60, 60), (255, 255, 255), (0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0)],
    "vaihingen": [(60, 60, 60), (255, 255, 255), (0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0)],
    "uavid": [(60, 60, 60), (128, 0, 0), (128, 64, 128), (0, 128, 0), (128, 128, 0),
              (64, 0, 128), (192, 0, 192), (64, 64, 0)],
    "loveda": [(60, 60, 60), (255, 255, 255), (255, 0, 0), (255, 255, 0), (0, 0, 255),
               (159, 129, 183), (0, 255, 0), (255, 195, 128)],
    "whu_building": [(60, 60, 60), (0, 0, 0), (255, 60, 60)],
    "massachusetts_buildings": [(60, 60, 60), (0, 0, 0), (255, 60, 60)],
    "massachusetts_roads": [(60, 60, 60), (0, 0, 0), (255, 60, 60)],
}

ROOTS = {
    "potsdam": "potsdam_mmseg", "vaihingen": "vaihingen_mmseg", "uavid": "uavid",
    "loveda": "loveDA_mmseg", "whu_building": "WHU",
    "massachusetts_buildings": "massachusetts_buildings_mmseg",
    "massachusetts_roads": "massachusetts_roads_mmseg",
}


def colourise(mask: np.ndarray, palette) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c, col in enumerate(palette):
        rgb[mask == c] = col
    return rgb


def find_checkpoint(exp: str, repo: str) -> str | None:
    direct = os.path.join(repo, "experiments", exp, "best.pth")
    if os.path.isfile(direct):
        return direct
    hits = glob.glob(os.path.join(repo, "experiments", exp, "*", "best.pth"))
    return hits[0] if hits else None


def per_image_iou(gt: np.ndarray, pred: np.ndarray, n_ch: int, ignore: int) -> float:
    valid = gt != ignore
    ious = []
    for c in range(n_ch):
        if c == ignore:
            continue
        g, p = (gt == c) & valid, (pred == c) & valid
        union = (g | p).sum()
        if g.sum() > 0:
            ious.append((g & p).sum() / union if union else 0.0)
    return float(np.mean(ious)) if ious else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(ROOTS))
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--pool", type=int, default=60, help="Candidate tiles to score before choosing")
    ap.add_argument("--select", default="disagreement",
                    choices=["disagreement", "worst", "random"])
    ap.add_argument("--names", nargs="*", default=None, help="Explicit tile filenames to render")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    rng = np.random.default_rng(args.seed)

    cfg = load_config(os.path.join(args.repo, "configs", f"sam3_lora_{args.dataset}.yaml"))
    ds_cfg = cfg.dataset
    dataset = create_dataset(ds_cfg.type, root=args.root, split=args.split,
                             image_size=ds_cfg.image_size, augment=False,
                             exclude_classes=ds_cfg.exclude_classes or None)
    ignore = dataset.IGNORE_INDEX
    n_cls = len(dataset.active_classes)
    n_ch = n_cls + 1
    names = ["ignore"] + list(dataset.active_classes.values())
    # Pad rather than truncate: UAVid's head carries a ninth, never-populated channel, so n_ch can
    # exceed the number of colours the palette defines.
    palette = list(PALETTES[args.dataset])[:n_ch]
    while len(palette) < n_ch:
        palette.append((200, 200, 200))

    if args.names:
        chosen = [p for p in dataset.pairs if os.path.basename(p[0]) in set(args.names)]
    else:
        idx = rng.choice(len(dataset.pairs), size=min(args.pool, len(dataset.pairs)), replace=False)
        chosen = [dataset.pairs[i] for i in sorted(idx)]

    # Run all three methods over the candidate pool.
    preds = {}
    for method, _ in METHODS:
        exp = f"{method}_{args.dataset}"
        ck = find_checkpoint(exp, args.repo)
        if ck is None:
            print(f"!! no checkpoint for {exp}; skipping")
            continue
        mcfg = load_config(os.path.join(args.repo, "configs", f"{exp}.yaml")).model
        model = build_peft_model(
            sam_model=None, method=method, num_classes=n_cls, image_size=ds_cfg.image_size,
            sam3_checkpoint=getattr(mcfg, "sam3_checkpoint", None),
            bpe_path=getattr(mcfg, "bpe_path", None),
            rank=getattr(mcfg, "rank", 8), alpha=getattr(mcfg, "alpha", 16),
            dropout=getattr(mcfg, "dropout", 0.0),
            scale_factor=getattr(mcfg, "scale_factor", 32),
            input_type=getattr(mcfg, "input_type", "fft"),
            freq_nums=getattr(mcfg, "freq_nums", 0.25),
            prompt_type=getattr(mcfg, "prompt_type", "highpass"),
            tuning_stage=getattr(mcfg, "tuning_stage", "1234"),
            handcrafted_tune=getattr(mcfg, "handcrafted_tune", True),
            embedding_tune=getattr(mcfg, "embedding_tune", True),
            adaptor=getattr(mcfg, "adaptor", "adaptor"),
        ).cuda()
        model.load_parameters(ck)
        model.eval()
        out = []
        with torch.no_grad():
            for img_path, ann_path in chosen:
                pil = PILImage.open(img_path).convert("RGB")
                raw = np.array(PILImage.open(ann_path))
                if raw.ndim == 3:
                    raw = raw[:, :, 0]
                arr = np.array(pil.resize((ds_cfg.image_size, ds_cfg.image_size),
                                          PILImage.BILINEAR), dtype=np.float32)
                x = torch.from_numpy(arr).permute(2, 0, 1)[None].cuda()
                low = model(x, n_cls > 2, ds_cfg.image_size)["low_res_logits"].float()
                up = F.interpolate(low, size=raw.shape, mode="bilinear", align_corners=False)
                out.append(up.argmax(dim=1)[0].cpu().numpy().astype(np.uint8))
        preds[method] = out
        del model
        torch.cuda.empty_cache()

    # Ground truth at native resolution.
    gts, imgs = [], []
    for img_path, ann_path in chosen:
        raw = np.array(PILImage.open(ann_path))
        if raw.ndim == 3:
            raw = raw[:, :, 0]
        gt = np.full_like(raw, ignore, dtype=np.uint8)
        for old, new in dataset.class_id_remap.items():
            gt[raw == old] = new
        gts.append(gt)
        imgs.append(np.array(PILImage.open(img_path).convert("RGB")))

    # Rank the pool and keep the most informative tiles.
    scores = []
    for k in range(len(chosen)):
        ious = {m: per_image_iou(gts[k], preds[m][k], n_ch, ignore) for m in preds}
        if args.select == "disagreement":
            s = max(ious.values()) - min(ious.values())
        elif args.select == "worst":
            s = -ious.get("sam3_lora", 0.0)
        else:
            s = rng.random()
        scores.append((s, k, ious))
    scores.sort(reverse=True)
    # Prefer tiles from distinct scenes. UAVid tiles are named by sequence and ISPRS tiles by
    # orthophoto, so without this the highest-disagreement tiles often all come from one scene and
    # the figure shows the same place three times.
    def scene_of(path):
        base = os.path.basename(path)
        return base.split("_")[0]

    keep, seen_scenes = [], set()
    for _, k, ious in scores:
        s = scene_of(chosen[k][0])
        if s in seen_scenes:
            continue
        keep.append((k, ious))
        seen_scenes.add(s)
        if len(keep) >= args.n:
            break
    for _, k, ious in scores:            # top up if there were not enough distinct scenes
        if len(keep) >= args.n:
            break
        if k not in [kk for kk, _ in keep]:
            keep.append((k, ious))

    ncol = 2 + len(preds)
    fig, axes = plt.subplots(len(keep), ncol, figsize=(2.35 * ncol, 2.45 * len(keep)))
    if len(keep) == 1:
        axes = axes[None, :]
    titles = ["Image", "Ground truth"] + [lbl for m, lbl in METHODS if m in preds]

    for r, (k, ious) in enumerate(keep):
        panels = [imgs[k], colourise(gts[k], palette)] + \
                 [colourise(preds[m][k], palette) for m, _ in METHODS if m in preds]
        for c, panel in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(panel)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
            if r == 0:
                ax.set_title(titles[c], fontsize=9)
            if c >= 2:
                m = [mm for mm, _ in METHODS if mm in preds][c - 2]
                ax.set_xlabel(f"IoU {ious[m]:.3f}", fontsize=7.5, labelpad=1.5)
        axes[r, 0].set_ylabel(os.path.basename(chosen[k][0])[:18], fontsize=6.5)

    # Outline every swatch: several palettes use white or near-white for a class, which would
    # otherwise be invisible against the figure background.
    # Skip channels that carry no ground truth anywhere in the panel, which removes UAVid's inert
    # ninth channel from the legend rather than presenting it as a class.
    present = {int(c) for g in gts for c in np.unique(g)}
    handles = [mpatches.Patch(facecolor=np.array(palette[c]) / 255, edgecolor="0.35",
                              linewidth=0.6, label=names[c])
               for c in range(1, n_ch)
               if c in present and str(names[c]).lower() != "unused"]
    handles.append(mpatches.Patch(facecolor=np.array(palette[0]) / 255, edgecolor="0.35",
                                  linewidth=0.6, label="excluded (ignore)"))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 8),
               fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=220, bbox_inches="tight")
    print(f"wrote {args.out}")
    for k, ious in keep:
        print(f"  {os.path.basename(chosen[k][0]):40s} " +
              "  ".join(f"{m.replace('sam3_','')}={v:.3f}" for m, v in ious.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
