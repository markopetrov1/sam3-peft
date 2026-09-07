#!/usr/bin/env python3
"""
Near-duplicate audit across a benchmark's train and test splits.

Why a duplicate test rather than an adjacency test. We first tried to detect whether test tiles are
physically contiguous with training tiles by correlating their outer pixel rows. That test is not
usable at this scale: calibrated on ISPRS Potsdam, where true adjacency is known from the tile
coordinates, genuinely adjacent edges correlate at a median of 0.957 while unrelated edges reach
0.868 at the 99.99th percentile, and since every test tile is compared against thousands of training
tiles the false matches swamp the true ones. Controlling the family-wise error rate pushes the
threshold to 0.98 and drops the detector's sensitivity to about 21%, so neither a permissive nor a
strict threshold yields a trustworthy per-tile answer. We report that negative result rather than
selecting whichever threshold tells the more convenient story.

Near-duplicate detection does separate cleanly, because a duplicated or heavily overlapping tile is
nearly identical to its counterpart while unrelated aerial tiles are not. This script computes a
64-bit perceptual hash (DCT-based pHash) for every tile and reports, for each test tile, the smallest
Hamming distance to any training tile. Identical content gives distance 0, and unrelated tiles
typically sit near 32, which is the expected distance between independent random hashes.

Usage
-----
    python scripts/duplicate_audit.py --dataset whu_building --root /path/to/WHU \
        --out analysis/duplicates_whu.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image as PILImage
from scipy.fftpack import dct
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LAYOUTS = {
    "whu_building": lambda root, s: os.path.join(root, s, "Image"),
    "potsdam": lambda root, s: os.path.join(root, "img_dir", s),
    "vaihingen": lambda root, s: os.path.join(root, "img_dir", s),
    "massachusetts_buildings": lambda root, s: os.path.join(root, "img_dir", s),
    "massachusetts_roads": lambda root, s: os.path.join(root, "img_dir", s),
    "loveda": lambda root, s: os.path.join(root, "img_dir", s),
}

HASH_SIDE = 8      # 8x8 low-frequency block -> 64-bit hash
DCT_SIDE = 32


def phash(path: str) -> np.ndarray:
    im = PILImage.open(path).convert("L").resize((DCT_SIDE, DCT_SIDE), PILImage.BILINEAR)
    a = np.asarray(im, dtype=np.float64)
    d = dct(dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")[:HASH_SIDE, :HASH_SIDE]
    flat = d.flatten()
    # Exclude the DC term from the median so that overall brightness does not dominate.
    med = np.median(flat[1:])
    return (flat > med).astype(np.uint8)


def load(root: str, dataset: str, split: str):
    d = LAYOUTS[dataset](root, split)
    paths = sorted(sum([glob.glob(os.path.join(d, e))
                        for e in ("*.png", "*.jpg", "*.tif", "*.tiff")], []))
    if not paths:
        return [], np.zeros((0, HASH_SIDE * HASH_SIDE), np.uint8)
    H = np.stack([phash(p) for p in tqdm(paths, desc=f"{dataset}/{split}", leave=False)])
    return paths, H


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(LAYOUTS))
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-split", default="test")
    ap.add_argument("--near-threshold", type=int, default=5,
                    help="Hamming distance at or below which two tiles are called near-duplicates")
    args = ap.parse_args()

    tr_paths, Htr = load(args.root, args.dataset, "train")
    te_paths, Hte = load(args.root, args.dataset, args.test_split)
    if not tr_paths or not te_paths:
        print(f"{args.dataset}: missing split, nothing to do")
        return 1
    print(f"{args.dataset}: {len(tr_paths)} train, {len(te_paths)} {args.test_split}")

    # Hamming distance via matrix algebra: d = a + b - 2 a.b for 0/1 vectors.
    A = Hte.astype(np.int16)
    B = Htr.astype(np.int16)
    sa = A.sum(1, keepdims=True)
    sb = B.sum(1, keepdims=True).T
    best = np.full(len(te_paths), 64, dtype=np.int16)
    best_idx = np.zeros(len(te_paths), dtype=np.int64)
    step = 256
    for s in range(0, len(te_paths), step):
        D = sa[s : s + step] + sb - 2 * (A[s : s + step] @ B.T)
        j = D.argmin(axis=1)
        m = D[np.arange(D.shape[0]), j]
        best[s : s + step] = m.astype(np.int16)
        best_idx[s : s + step] = j

    near = best <= args.near_threshold
    exact = best == 0
    out = {
        "dataset": args.dataset, "test_split": args.test_split,
        "n_train": len(tr_paths), "n_test": len(te_paths),
        "near_threshold": args.near_threshold,
        "n_test_exact_duplicate_of_train": int(exact.sum()),
        "n_test_near_duplicate_of_train": int(near.sum()),
        "frac_test_near_duplicate_of_train": float(near.mean()),
        "min_distance_distribution": {
            "min": int(best.min()), "p01": float(np.percentile(best, 1)),
            "median": float(np.median(best)), "mean": float(best.mean()),
        },
        "examples": [
            {"test": os.path.basename(te_paths[i]),
             "nearest_train": os.path.basename(tr_paths[best_idx[i]]),
             "hamming": int(best[i])}
            for i in np.argsort(best)[:15]
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"  exact duplicates: {out['n_test_exact_duplicate_of_train']}, "
          f"near (<= {args.near_threshold}): {out['n_test_near_duplicate_of_train']} "
          f"({100*out['frac_test_near_duplicate_of_train']:.2f}%)")
    print(f"  min-distance median {out['min_distance_distribution']['median']:.1f}, "
          f"min {out['min_distance_distribution']['min']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
