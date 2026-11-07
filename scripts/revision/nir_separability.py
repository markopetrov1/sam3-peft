#!/usr/bin/env python3
"""
Descriptive statistics behind the different sensitivity to the near-infrared channel on the two ISPRS
benchmarks (Section 4.10 of the paper). No model is involved.

For the test tiles of Potsdam (near-infrared, green, blue) and Vaihingen (near-infrared, red, green)
it reports, over the pixels that enter the primary score:

  class shares       the fraction of evaluated pixels in each class;
  channel AUC        how well each input channel on its own separates vegetation (low vegetation and
                     tree) from the other evaluated classes, as the area under the ROC curve of the raw
                     channel value, folded so that 0.5 means no separation and 1 perfect separation;
  correlation        the Pearson correlation between the near-infrared channel and the channel that
                     replaces it in the channel-replacement analysis (red on Vaihingen, green on Potsdam);
  index AUC          the same AUC for the normalised difference between the near-infrared channel and
                     that neighbouring visible channel, a vegetation index computable on both benchmarks;
  linear AUC         the held-out AUC of a logistic regression on the two visible channels alone and on
                     all three channels, fitted on the pixels of half the tiles and scored on the other
                     half, which measures how much the near-infrared channel adds to the visible ones.

Pixels are subsampled uniformly within every tile with a fixed seed.

Usage: python scripts/revision/nir_separability.py --data-root /path/to/datasets [--per-tile 4000]
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image as PILImage
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from datasets import create_dataset  # noqa: E402
from utils.config import load_config  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SETS = {"potsdam": ("potsdam_mmseg", ["near-infrared", "green", "blue"], 1),
        "vaihingen": ("vaihingen_mmseg", ["near-infrared", "red", "green"], 1)}
VEGETATION = {"low vegetation", "tree"}


def auc(score, label):
    """Area under the ROC curve of `score` for the positive class, folded to [0.5, 1]."""
    r = rankdata(score)
    n_pos, n_neg = int(label.sum()), int((~label).sum())
    a = (r[label].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(max(a, 1 - a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("SAM3_DATA_ROOT", os.path.expanduser("~/datasets")))
    ap.add_argument("--per-tile", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    res = {}
    for name, (folder, channels, replacement) in SETS.items():
        cfg = load_config(os.path.join(REPO, "configs", f"sam3_lora_{name}.yaml")).dataset
        ds = create_dataset(cfg.type, root=os.path.join(a.data_root, folder), split="test",
                            image_size=cfg.image_size, augment=False,
                            exclude_classes=cfg.exclude_classes or None)
        classes = list(ds.active_classes.values())
        pix, lab, tile = [], [], []
        for t, (img_path, ann_path) in enumerate(ds.pairs):
            img = np.array(PILImage.open(img_path).convert("RGB"))
            raw = np.array(PILImage.open(ann_path))
            raw = raw[:, :, 0] if raw.ndim == 3 else raw
            gt = np.full(raw.shape, ds.IGNORE_INDEX, dtype=np.int64)
            for old, new in ds.class_id_remap.items():
                gt[raw == old] = new
            valid = np.flatnonzero(gt.ravel() != ds.IGNORE_INDEX)
            if valid.size == 0:
                continue
            take = rng.choice(valid, size=min(a.per_tile, valid.size), replace=False)
            pix.append(img.reshape(-1, 3)[take])
            lab.append(gt.ravel()[take])
            tile.append(np.full(take.size, t))
        pix, lab, tile = np.concatenate(pix).astype(np.float64), np.concatenate(lab), np.concatenate(tile)
        # class ids start at 1 in the remapped labels (0 is the ignore channel)
        shares = {c: float(np.mean(lab == i + 1)) for i, c in enumerate(classes)}
        veg = np.isin(lab, [i + 1 for i, c in enumerate(classes) if c in VEGETATION])
        res[name] = {
            "tiles": len(ds.pairs), "pixels_sampled": int(lab.size), "class_shares": shares,
            "vegetation_share": float(veg.mean()),
            "channel_auc_vegetation": {ch: auc(pix[:, k], veg) for k, ch in enumerate(channels)},
            "corr_nir_with_replacement": float(np.corrcoef(pix[:, 0], pix[:, replacement])[0, 1]),
            "replacement_channel": channels[replacement],
        }
        nd = (pix[:, 0] - pix[:, replacement]) / (pix[:, 0] + pix[:, replacement] + 1e-6)
        res[name]["index_auc_vegetation"] = auc(nd, veg)
        fit_tiles = rng.permutation(len(ds.pairs))[: len(ds.pairs) // 2]
        tr = np.isin(tile, fit_tiles)
        x = pix / 255.0
        for key, cols in [("linear_auc_visible_only", [1, 2]), ("linear_auc_all_three", [0, 1, 2])]:
            clf = LogisticRegression(max_iter=1000).fit(x[tr][:, cols], veg[tr])
            res[name][key] = auc(clf.decision_function(x[~tr][:, cols]), veg[~tr])
        print(name, json.dumps(res[name], indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
