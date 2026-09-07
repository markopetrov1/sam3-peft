#!/usr/bin/env python3
"""
Calibrate an image-edge adjacency detector, then apply it to a split whose tile geometry is unknown.

An earlier, uncalibrated version of this test simply thresholded the correlation between the outer
pixel lines of two tiles. That is not sound: aerial imagery is smooth, so unrelated tiles correlate
substantially, and the chance of a spurious match grows with the number of training tiles a test
tile is compared against. A benchmark with a large training set therefore looks "leakier" than a
small one purely as an artefact.

This script fixes that by calibrating on ISPRS Potsdam, where true adjacency is known from the tile
filenames: tiles are cut from named orthophotos on a regular grid, so two tiles are genuinely
contiguous exactly when they come from the same orthophoto and their grid offsets differ by one step
along one axis. Correlations for those known-adjacent pairs give the positive distribution, and pairs
drawn from different orthophotos give the negative distribution. The decision threshold is set at a
chosen false-positive rate on the negatives, and its true-positive rate is reported so the detector's
power is known.

The calibrated threshold is then applied to WHU Building, whose tiles were renamed when the split was
rebuilt and whose grid geometry is therefore not recoverable from filenames.

Usage
-----
    python scripts/calibrate_adjacency.py --potsdam-root /path/to/potsdam_mmseg \
        --whu-root /path/to/WHU --out analysis/adjacency_calibrated.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
from PIL import Image as PILImage
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

POTSDAM_RE = re.compile(r"^(?P<ortho>\d+_\d+)_(?P<x>\d+)_(?P<y>\d+)_(?P<x2>\d+)_(?P<y2>\d+)\.png$")
STEP = 512


def edges(path: str) -> np.ndarray:
    """Standardised outer pixel lines: [top, bottom, left, right], each length 512."""
    a = np.asarray(PILImage.open(path).convert("RGB"), dtype=np.float32).mean(axis=2) / 255.0
    out = []
    for line in (a[0, :], a[-1, :], a[:, 0], a[:, -1]):
        v = line
        if v.shape[0] != 512:
            v = np.interp(np.linspace(0, v.shape[0] - 1, 512), np.arange(v.shape[0]), v)
        v = v - v.mean()
        out.append(v / (np.linalg.norm(v) + 1e-8))
    return np.stack(out).astype(np.float32)


TOP, BOTTOM, LEFT, RIGHT = 0, 1, 2, 3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--potsdam-root", required=True)
    ap.add_argument("--whu-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-false-per-tile", type=float, default=0.01,
                    help="Expected number of false edge matches allowed per test tile")
    ap.add_argument("--n-neg", type=int, default=5000000)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # ---- Calibration on Potsdam, pooling all splits so that adjacency is densely sampled. ----
    paths = []
    for split in ("train", "val", "test"):
        paths += sorted(glob.glob(os.path.join(args.potsdam_root, "img_dir", split, "*.png")))
    meta, keep = [], []
    for p in paths:
        m = POTSDAM_RE.match(os.path.basename(p))
        if m:
            meta.append((m.group("ortho"), int(m.group("x")), int(m.group("y"))))
            keep.append(p)
    print(f"Potsdam calibration tiles: {len(keep)}")

    E = np.stack([edges(p) for p in tqdm(keep, desc="potsdam edges")])
    index = {(o, x, y): i for i, (o, x, y) in enumerate(meta)}

    # Positives: same orthophoto, one grid step apart, comparing the two touching edges.
    # The filename encodes (x, y), that is (column, row), not (row, column): verified directly on
    # 2_10, where the bottom-to-top correlation of two tiles one step apart in the second
    # coordinate is 0.985 and the right-to-left correlation one step apart in the first is 0.939.
    # So the first parsed offset advances columns and the second advances rows.
    pos = []
    for i, (o, x, y) in enumerate(meta):
        j = index.get((o, x + STEP, y))
        if j is not None:                       # i is immediately left of j
            pos.append(float(abs(E[i, RIGHT] @ E[j, LEFT])))
        j = index.get((o, x, y + STEP))
        if j is not None:                       # i is immediately above j
            pos.append(float(abs(E[i, BOTTOM] @ E[j, TOP])))
    pos = np.array(pos)

    # Negatives: tiles from different orthophotos, any edge pairing.
    orthos = np.array([o for o, _, _ in meta])
    n = len(meta)
    a = rng.integers(0, n, args.n_neg)
    b = rng.integers(0, n, args.n_neg)
    ok = orthos[a] != orthos[b]
    a, b = a[ok], b[ok]
    ea = rng.integers(0, 4, len(a))
    eb = rng.integers(0, 4, len(b))
    neg = np.abs(np.einsum("ij,ij->i", E[a, ea], E[b, eb]))

    print(f"  positives {len(pos)}, negatives {len(neg)}")
    print(f"  pos median {np.median(pos):.4f}, neg median {np.median(neg):.4f}")

    # Each test tile is compared against every training tile under four edge pairings, so a
    # threshold calibrated for a single comparison is applied thousands of times per tile and
    # false positives accumulate to near-certainty. The threshold is therefore set so that the
    # expected number of false matches per test tile, which is the number of comparisons times
    # the per-comparison false-positive rate, stays below `--max-false-per-tile`.
    def threshold_for(n_train: int) -> tuple[float, float, float]:
        n_comp = 4 * n_train
        target_p = args.max_false_per_tile / n_comp
        thr = float(np.quantile(neg, 1.0 - target_p))
        realised = float((neg >= thr).mean())
        return thr, realised * n_comp, float((pos >= thr).mean())

    # ---- Apply to both benchmarks: how many test tiles touch a training tile? ----
    def measure(name, train_paths, test_paths):
        Etr = np.stack([edges(p) for p in tqdm(train_paths, desc=f"{name} train")])
        Ete = np.stack([edges(p) for p in tqdm(test_paths, desc=f"{name} test")])
        # Only opposite edges can touch: test right vs train left, etc.
        pairs = [(RIGHT, LEFT), (LEFT, RIGHT), (BOTTOM, TOP), (TOP, BOTTOM)]
        best = np.zeros(len(test_paths), dtype=np.float32)
        for te_e, tr_e in pairs:
            M = np.abs(Ete[:, te_e] @ Etr[:, tr_e].T)
            best = np.maximum(best, M.max(axis=1))
        thr, exp_false, tpr = threshold_for(len(train_paths))
        hits = int((best >= thr).sum())
        print(f"  {name}: threshold {thr:.4f} (expected false matches per tile {exp_false:.3f}, "
              f"detector TPR {tpr:.3f})")
        return {"n_train": len(train_paths), "n_test": len(test_paths),
                "threshold": thr, "expected_false_matches_per_tile": exp_false,
                "detector_tpr_at_threshold": tpr,
                "n_test_touching_train": hits,
                "frac_test_touching_train": hits / len(test_paths),
                "mean_best_edge_corr": float(best.mean()),
                "median_best_edge_corr": float(np.median(best))}

    whu_tr = sorted(glob.glob(os.path.join(args.whu_root, "train", "Image", "*.png")))
    whu_te = sorted(glob.glob(os.path.join(args.whu_root, "test", "Image", "*.png")))
    pot_tr = sorted(glob.glob(os.path.join(args.potsdam_root, "img_dir", "train", "*.png")))
    pot_te = sorted(glob.glob(os.path.join(args.potsdam_root, "img_dir", "test", "*.png")))

    out = {
        "calibration": {
            "max_false_per_tile": args.max_false_per_tile,
            "n_positive_pairs": int(len(pos)), "n_negative_pairs": int(len(neg)),
            "pos_median": float(np.median(pos)), "neg_median": float(np.median(neg)),
            "pos_p05": float(np.percentile(pos, 5)),
            "neg_p99_99": float(np.percentile(neg, 99.99)),
        },
        "whu_building_ours_random_split": measure("whu", whu_tr, whu_te),
        "potsdam_official_orthophoto_split": measure("potsdam", pot_tr, pot_te),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    for k in ("whu_building_ours_random_split", "potsdam_official_orthophoto_split"):
        v = out[k]
        print(f"{k}: {v['n_test_touching_train']}/{v['n_test']} "
              f"({100*v['frac_test_touching_train']:.1f}%) test tiles touch a training tile")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
