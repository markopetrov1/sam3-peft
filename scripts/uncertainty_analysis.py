#!/usr/bin/env python3
"""
Image-level uncertainty and paired method comparisons from saved per-image confusion matrices.

The study trained one model per (strategy, benchmark) cell, so seed-to-seed training variance
cannot be estimated after the fact. What *can* be estimated without retraining is the uncertainty
that comes from evaluating on a finite test set, and whether two strategies differ consistently
across the individual test images.

This script therefore reports, per benchmark:

  * a nonparametric bootstrap confidence interval for each strategy's mIoU, obtained by resampling
    test images with replacement and recomputing mIoU from the summed confusion matrices, and
  * a paired comparison between two strategies: the bootstrap interval of the *difference*
    (using the same resampled image indices for both, so the pairing is preserved) and an exact-
    style paired permutation test in which the two strategies' per-image confusion matrices are
    swapped independently per image.

IMPORTANT INTERPRETATION. These intervals quantify test-set sampling uncertainty only. They do
NOT capture the variance that repeated training runs with different seeds would reveal, and they
must not be read as a substitute for it.

Usage
-----
    python scripts/uncertainty_analysis.py --reeval-dir analysis/reeval --out analysis/stats.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

METHODS = ["linear_probing", "lora", "adapter"]
DATASETS = ["massachusetts_buildings", "massachusetts_roads", "potsdam", "uavid",
            "vaihingen", "whu_building", "loveda"]


def miou_from(conf: np.ndarray, ignore: int = 0) -> float:
    """mIoU over classes present in the ground truth, excluding the ignore channel."""
    n = conf.shape[0]
    tp = np.diag(conf).astype(np.float64)
    fp = conf.sum(axis=0) - tp
    fn = conf.sum(axis=1) - tp
    denom = tp + fp + fn
    iou = np.divide(tp, denom, out=np.zeros(n), where=denom > 0)
    active = conf.sum(axis=1) > 0
    active[ignore] = False
    return float(iou[active].mean()) if active.any() else float("nan")


def load(reeval_dir: str, method: str, ds: str):
    p = os.path.join(reeval_dir, f"sam3_{method}_{ds}", "confusions.npz")
    if not os.path.isfile(p):
        return None
    z = np.load(p, allow_pickle=True)
    return z["per_image"].astype(np.int64)


def bootstrap_miou(per_image: np.ndarray, boot_idx: np.ndarray) -> np.ndarray:
    """mIoU for each bootstrap replicate, given precomputed resampling indices."""
    return np.array([miou_from(per_image[idx].sum(axis=0)) for idx in boot_idx])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reeval-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    results = {}

    for ds in DATASETS:
        data = {m: load(args.reeval_dir, m, ds) for m in METHODS}
        data = {m: v for m, v in data.items() if v is not None}
        if not data:
            print(f"-- {ds}: no re-evaluation output, skipped")
            continue
        n_img = next(iter(data.values())).shape[0]
        if any(v.shape[0] != n_img for v in data.values()):
            print(f"!! {ds}: methods disagree on image count, skipped")
            continue

        # One shared set of resampling indices keeps the method comparison paired.
        boot_idx = rng.integers(0, n_img, size=(args.n_boot, n_img))
        entry = {"n_images": int(n_img), "point": {}, "ci95": {}}
        boots = {}
        for m, per_image in data.items():
            entry["point"][m] = miou_from(per_image.sum(axis=0))
            b = bootstrap_miou(per_image, boot_idx)
            boots[m] = b
            entry["ci95"][m] = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]

        entry["paired"] = {}
        pairs = [("lora", "adapter"), ("lora", "linear_probing"), ("adapter", "linear_probing")]
        for a, b in pairs:
            if a not in data or b not in data:
                continue
            diff_boot = boots[a] - boots[b]
            obs = entry["point"][a] - entry["point"][b]

            # Paired permutation: independently swap the two methods' per-image confusion
            # matrices, recompute the difference, and count how often |perm| >= |observed|.
            pa, pb = data[a], data[b]
            count = 0
            for _ in range(args.n_perm // 100):
                flips = rng.random((100, n_img)) < 0.5
                for f in flips:
                    ca = np.where(f[:, None, None], pb, pa).sum(axis=0)
                    cb = np.where(f[:, None, None], pa, pb).sum(axis=0)
                    if abs(miou_from(ca) - miou_from(cb)) >= abs(obs) - 1e-12:
                        count += 1
            n_perm = (args.n_perm // 100) * 100
            entry["paired"][f"{a}_minus_{b}"] = {
                "observed": float(obs),
                "ci95": [float(np.percentile(diff_boot, 2.5)),
                         float(np.percentile(diff_boot, 97.5))],
                "frac_bootstrap_favouring_first": float((diff_boot > 0).mean()),
                "permutation_p": float((count + 1) / (n_perm + 1)),
            }
        results[ds] = entry
        pt = "  ".join(f"{m}={entry['point'][m]:.4f}" for m in entry["point"])
        print(f"-- {ds} (n={n_img}): {pt}")
        for k, v in entry["paired"].items():
            print(f"     {k}: {v['observed']:+.4f} "
                  f"[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]  p={v['permutation_p']:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"note": "Bootstrap over test images; captures test-set sampling uncertainty "
                           "only, not seed-to-seed training variance.",
                   "n_boot": args.n_boot, "n_perm": args.n_perm,
                   "results": results}, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
