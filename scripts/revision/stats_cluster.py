#!/usr/bin/env python3
"""
Uncertainty for mean IoU when evaluation images are not independent.

Tiles cut from one orthophoto or source image, and frames from one UAV sequence, share scene
content, so they are resampled together. Every statistic is a mean IoU computed from summed
confusion matrices, never an average of per-image scores.

  cluster bootstrap   clusters are drawn with replacement (Field and Welsh 2007); the same draw is
                      used for both methods of a pair, so differences stay paired.
  sign-flip test      under the null that the two trained models are exchangeable, the model labels
                      of all images in a cluster are swapped together. All 2^G assignments are
                      enumerated when G <= 20, otherwise a Monte Carlo sample is drawn; the p-value
                      counts the observed assignment, so its floor is 2/2^G for an exact test.

Both statements are conditional on the two trained models; they describe test-set sampling, not
the variability of training.
"""
from __future__ import annotations

import itertools
import re

import numpy as np

GROUPS = {
    "potsdam": lambda n: "_".join(n.split("_")[:2]),            # 2_13_x0_y0_x1_y1.png -> 2_13
    "vaihingen": lambda n: n.split("_")[0],                     # area10_... -> area10
    "massachusetts_buildings": lambda n: "_".join(n.split("_")[:2]),   # 22828930_15_... -> 22828930_15
    "massachusetts_roads": lambda n: "_".join(n.split("_")[:2]),
    "uavid": lambda n: n.split("_")[0],                         # seq21_000000.png -> seq21
    "loveda": lambda n: n[:-4],                                 # no spatial grouping is released
    "whu_building": lambda n: n[:-4],                           # our random repartition, no grouping
}


def cluster_index(dataset: str, names) -> tuple[np.ndarray, list]:
    keys = [GROUPS[dataset](str(n)) for n in names]
    uniq = sorted(set(keys), key=lambda k: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", k)])
    pos = {k: i for i, k in enumerate(uniq)}
    return np.array([pos[k] for k in keys]), uniq


def miou(conf: np.ndarray, classes) -> np.ndarray:
    """Mean IoU over `classes` for one confusion matrix [K,K] or a stack [B,K,K].

    A class absent from the reference of a given matrix is left out of that matrix's mean, as in
    the evaluation code.
    """
    c = conf if conf.ndim == 3 else conf[None]
    tp = np.einsum("bii->bi", c).astype(np.float64)
    gt = c.sum(axis=2).astype(np.float64)
    pr = c.sum(axis=1).astype(np.float64)
    denom = gt + pr - tp
    iou = np.divide(tp, denom, out=np.zeros_like(tp), where=denom > 0)
    sel = np.zeros(c.shape[1], bool)
    sel[list(classes)] = True
    present = (gt > 0) & sel[None, :]
    out = (iou * present).sum(1) / np.maximum(present.sum(1), 1)
    return out if conf.ndim == 3 else out[0]


def per_cluster(conf_img: np.ndarray, cidx: np.ndarray, G: int) -> np.ndarray:
    out = np.zeros((G,) + conf_img.shape[1:], dtype=np.float64)
    np.add.at(out, cidx, conf_img)
    return out


def bootstrap(conf_a, conf_b, cidx, classes, n_boot=10000, seed=0):
    """Percentile intervals for mIoU(a), mIoU(b) and the paired difference a - b."""
    G = int(cidx.max()) + 1
    A = per_cluster(conf_a, cidx, G).reshape(G, -1)
    B = None if conf_b is None else per_cluster(conf_b, cidx, G).reshape(G, -1)
    rng = np.random.default_rng(seed)
    K = conf_a.shape[1]
    stats_a, stats_b = np.empty(n_boot), np.empty(n_boot)
    chunk = 2000
    for s in range(0, n_boot, chunk):
        m = min(chunk, n_boot - s)
        draw = rng.integers(0, G, size=(m, G))
        W = np.zeros((m, G))
        np.add.at(W, (np.repeat(np.arange(m), G), draw.ravel()), 1)
        stats_a[s:s + m] = miou((W @ A).reshape(m, K, K), classes)
        if B is not None:
            stats_b[s:s + m] = miou((W @ B).reshape(m, K, K), classes)
    out = {"G": G, "ci_a": np.percentile(stats_a, [2.5, 97.5]).tolist()}
    if B is not None:
        d = stats_a - stats_b
        out["ci_b"] = np.percentile(stats_b, [2.5, 97.5]).tolist()
        out["ci_diff"] = np.percentile(d, [2.5, 97.5]).tolist()
        out["sd_diff"] = float(d.std(ddof=1))
    return out


def sign_flip(conf_a, conf_b, cidx, classes, mc=200000, seed=0):
    """Two-sided paired randomisation test at cluster level for mIoU(a) - mIoU(b)."""
    G = int(cidx.max()) + 1
    A = per_cluster(conf_a, cidx, G)
    B = per_cluster(conf_b, cidx, G)
    K = A.shape[1]
    obs = miou(A.sum(0), classes) - miou(B.sum(0), classes)
    Af, Bf = A.reshape(G, -1), B.reshape(G, -1)
    if G <= 20:
        S = np.array(list(itertools.product([0, 1], repeat=G)), dtype=np.float64)
        exact = True
    else:
        rng = np.random.default_rng(seed)
        S = rng.integers(0, 2, size=(mc, G)).astype(np.float64)
        S[0] = 0  # include the observed assignment
        exact = False
    count = 0
    for s in range(0, len(S), 16384):
        sl = S[s:s + 16384]
        ca = (1 - sl) @ Af + sl @ Bf
        cb = sl @ Af + (1 - sl) @ Bf
        d = miou(ca.reshape(-1, K, K), classes) - miou(cb.reshape(-1, K, K), classes)
        count += int((np.abs(d) >= abs(obs) - 1e-12).sum())
    return {"G": G, "diff": float(obs), "p": count / len(S), "exact": exact, "n_assign": len(S),
            "p_floor": (2 if exact else 1) / len(S)}
