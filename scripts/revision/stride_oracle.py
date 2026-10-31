#!/usr/bin/env python3
"""
A model-free reference for the boundary analysis: what does an output stride of 14 cost on its own?

The oracle is given the true class composition of every cell of the 72x72 output grid, that is the
fraction of each class among the valid reference pixels the cell covers after the image has been
resized to the 1008x1008 network input exactly as the models see it. Those fractions are then
treated as the logits are treated: bilinearly resampled to the native reference resolution, with
the argmax taken there. The result is a prediction that is perfect at the resolution of the grid
and limited only by the grid, so its boundary-band and interior scores separate what the stride
removes from what the adaptation gets wrong. It is a reference point, not a strict upper bound: a
trained head can produce sharper transitions than class fractions do.

The same input handling as the models is reproduced: `stretch` resizes the whole image (or frame)
to 1008x1008; `tiled` uses the 512x512 tile grid of the training split and averages overlapping
tiles, as eval_conventions.py does for UAVid.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage

from datasets import create_dataset  # noqa: E402
from scripts.revision.eval_conventions import BAND_RADII, boundary_band, confusion, tile_grid  # noqa: E402

GRID, SIZE, TILE = 72, 1008, 512


def fractions(gt_u8: np.ndarray, n_ch: int, ignore: int) -> torch.Tensor:
    """[n_ch, 72, 72] class fractions of a reference raster seen at 1008x1008 (nearest resize)."""
    g = np.array(PILImage.fromarray(gt_u8).resize((SIZE, SIZE), PILImage.NEAREST))
    g = torch.from_numpy(g.astype(np.int64))
    oh = F.one_hot(g, n_ch).permute(2, 0, 1).float()
    oh[ignore] = 0  # ignored pixels carry no class evidence
    cnt = F.avg_pool2d(oh[None], SIZE // GRID)[0]
    tot = cnt.sum(0, keepdim=True)
    return torch.where(tot > 0, cnt / tot.clamp(min=1e-9), torch.zeros_like(cnt))


def oracle_logits(gt_u8: np.ndarray, n_ch: int, ignore: int, mode: str) -> torch.Tensor:
    h, w = gt_u8.shape
    if mode == "stretch":
        f = fractions(gt_u8, n_ch, ignore)
        return F.interpolate(f[None], size=(h, w), mode="bilinear", align_corners=False)[0]
    acc = torch.zeros((n_ch, h, w))
    cnt = torch.zeros((1, h, w))
    for x0, y0, x1, y1 in tile_grid(h, w):
        f = fractions(np.ascontiguousarray(gt_u8[y0:y1, x0:x1]), n_ch, ignore)
        acc[:, y0:y1, x0:x1] += F.interpolate(f[None], size=(TILE, TILE), mode="bilinear",
                                              align_corners=False)[0]
        cnt[:, y0:y1, x0:x1] += 1
    return acc / cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--mode", default="stretch", choices=["stretch", "tiled"])
    ap.add_argument("--ann-dir", default=None, help="alternative reference tiles with the same names")
    ap.add_argument("--exclude", default="", help="comma list of raw ids excluded (e.g. 6 for ISPRS)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    excl = [int(x) for x in a.exclude.split(",") if x]
    ds = create_dataset(a.dataset, root=a.root, split=a.split, image_size=SIZE, augment=False,
                        exclude_classes=excl or None)
    ignore = ds.IGNORE_INDEX
    names_active = ["ignore"] + list(ds.active_classes.values())
    n_ch = len(names_active)
    eval_ch = [c for c in range(1, n_ch) if names_active[c] != "unused"]
    store, names = {}, []
    for i, (img_path, ann_path) in enumerate(ds.pairs):
        if a.ann_dir:
            ann_path = os.path.join(a.ann_dir, os.path.basename(ann_path))
        raw = np.array(PILImage.open(ann_path))
        raw = raw[:, :, 0] if raw.ndim == 3 else raw
        gt = np.full(raw.shape, ignore, np.uint8)
        for old, new in ds.class_id_remap.items():
            gt[raw == old] = new
        lg = oracle_logits(gt, n_ch, ignore, a.mode)
        pred = lg.argmax(0)
        gt_t = torch.from_numpy(gt)
        mats = {"full": confusion(gt_t, pred, n_ch, ignore)}
        for r in BAND_RADII:
            band = boundary_band(gt_t, r)
            mats[f"band{r}"] = confusion(torch.where(band, gt_t, ignore), pred, n_ch, ignore)
            mats[f"int{r}"] = confusion(torch.where(~band, gt_t, ignore), pred, n_ch, ignore)
        for k, m in mats.items():
            store.setdefault(k, np.zeros((len(ds.pairs), n_ch, n_ch), np.int64))[i] = m
        names.append(os.path.basename(img_path))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, names=np.array(names), class_names=np.array(names_active), **store)
    summ = {}
    for k, arr in store.items():
        c = arr.sum(0).astype(np.float64)
        tp = np.diag(c); den = c.sum(0) + c.sum(1) - tp
        iou = np.divide(tp, den, out=np.zeros(n_ch), where=den > 0)
        cls = [x for x in eval_ch if c[x].sum() > 0]
        summ[k] = float(iou[cls].mean())
    json.dump(summ, open(a.out.replace(".npz", ".json"), "w"), indent=2)
    print(a.dataset, a.mode, json.dumps({k: round(v, 4) for k, v in summ.items()}))


if __name__ == "__main__":
    main()
