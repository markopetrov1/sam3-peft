#!/usr/bin/env python3
"""
Plot pixel distribution per class for Potsdam, Vaihingen, LoveDA, UAVid, Massachusetts Buildings, Massachusetts Roads, or WHU Building.

Usage:
    python plot_pixel_distribution.py potsdam   # -> pixel_distribution_potsdam.png
    python plot_pixel_distribution.py vaihingen
    python plot_pixel_distribution.py loveda
    python plot_pixel_distribution.py uavid
    python plot_pixel_distribution.py massachusetts_buildings
    python plot_pixel_distribution.py massachusetts_roads
    python plot_pixel_distribution.py whu_building
"""

import argparse
import os
import glob

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Dataset location. Set SAM3_DATA_ROOT to the directory holding the prepared datasets;
# the paths below are resolved underneath it.
_DATA_ROOT = os.environ.get("SAM3_DATA_ROOT", "datasets")

# Dataset roots, display names, class names (index = class ID).
# exclude_from_plot: class IDs not shown as bars (e.g. unlabeled, background).
DATASET_CONFIG = {
    "potsdam": {
        "root": os.path.join(_DATA_ROOT, "potsdam_mmseg"),
        "display_name": "Potsdam",
        "ann_dirs": ["ann_dir/train", "ann_dir/val"],
        "exclude_from_plot": [0],
        "class_names": {
            0: "unlabeled",
            1: "impervious surface",
            2: "building",
            3: "low vegetation",
            4: "tree",
            5: "car",
            6: "clutter",
        },
    },
    "vaihingen": {
        "root": os.path.join(_DATA_ROOT, "vaihingen_mmseg"),
        "display_name": "Vaihingen",
        "ann_dirs": ["ann_dir/train", "ann_dir/val"],
        "exclude_from_plot": [0],
        "class_names": {
            0: "unlabeled",
            1: "impervious surface",
            2: "building",
            3: "low vegetation",
            4: "tree",
            5: "car",
            6: "clutter",
        },
    },
    "loveda": {
        "root": os.path.join(_DATA_ROOT, "loveDA_mmseg"),
        "display_name": "LoveDA",
        "ann_dirs": ["ann_dir/train", "ann_dir/val"],
        "exclude_from_plot": [0, 1],  # no-data and background — plot only semantic classes 2–7
        "class_names": {
            0: "no-data",
            1: "background",
            2: "building",
            3: "road",
            4: "water",
            5: "barren",
            6: "forest",
            7: "agriculture",
        },
    },
    "uavid": {
        "root": os.path.join(_DATA_ROOT, "uavid"),
        "display_name": "UAVid",
        "ann_dirs": ["train/masks", "val/masks", "test/masks"],
        "exclude_from_plot": [0],
        "class_names": {
            0: "unlabeled",
            1: "building",
            2: "road",
            3: "static car",
            4: "tree",
            5: "low vegetation",
            6: "human",
            7: "moving car",
            8: "background clutter",
        },
    },
    "massachusetts_buildings": {
        "root": os.path.join(_DATA_ROOT, "massachusetts_buildings_mmseg"),
        "display_name": "Massachusetts Buildings",
        "ann_dirs": ["ann_dir/train", "ann_dir/val", "ann_dir/test"],
        "exclude_from_plot": list(range(1, 255)),  # masks use 0 and 255 only
        "class_names": {
            0: "background",
            255: "building",
        },
    },
    "massachusetts_roads": {
        "root": os.path.join(_DATA_ROOT, "massachusetts_roads_mmseg"),
        "display_name": "Massachusetts Roads",
        "ann_dirs": ["ann_dir/train", "ann_dir/val", "ann_dir/test"],
        "exclude_from_plot": list(range(1, 255)),  # masks use 0 and 255 only
        "class_names": {
            0: "background",
            255: "road",
        },
    },
    "whu_building": {
        "root": os.path.join(_DATA_ROOT, "WHU"),
        "display_name": "WHU Building",
        "ann_dirs": ["train/Mask", "val/Mask", "test/Mask"],
        "exclude_from_plot": list(range(1, 255)),  # masks use 0 and 255 only
        "class_names": {
            0: "background",
            255: "building",
        },
    },
}


def collect_pixel_counts(ann_dirs_full, max_classes=10):
    """Count pixels per class across all annotation PNGs."""
    counts = {c: 0 for c in range(max_classes)}
    exts = ("*.png", "*.PNG")
    for ann_dir in ann_dirs_full:
        if not os.path.isdir(ann_dir):
            continue
        for ext in exts:
            for path in glob.glob(os.path.join(ann_dir, ext)):
                arr = np.array(Image.open(path))
                if arr.ndim == 3:
                    arr = arr[:, :, 0]
                for c in range(max_classes):
                    counts[c] += (arr == c).sum()
    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Plot pixel distribution per class for a dataset."
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=["potsdam", "vaihingen", "loveda", "uavid", "massachusetts_buildings", "massachusetts_roads", "whu_building"],
        help="Dataset name",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output path (default: pixel_distribution_<dataset>.png)",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Override dataset root",
    )
    args = parser.parse_args()

    cfg = DATASET_CONFIG[args.dataset]
    root = args.root or cfg["root"]
    ann_dirs_rel = cfg["ann_dirs"]
    class_names = cfg["class_names"]
    max_classes = max(class_names.keys()) + 1

    ann_dirs_full = [os.path.join(root, d) for d in ann_dirs_rel]
    present = [d for d in ann_dirs_full if os.path.isdir(d)]
    if not present:
        raise FileNotFoundError(f"No annotation dirs found under {root}: {ann_dirs_full}")

    counts = collect_pixel_counts(ann_dirs_full, max_classes=max_classes)
    total = sum(counts.values())
    if total == 0:
        raise RuntimeError("No pixels found in annotation directories.")

    exclude = set(cfg.get("exclude_from_plot", [0]))
    class_ids = [c for c in range(max_classes) if c not in exclude]
    labels = [class_names.get(c, f"class_{c}") for c in class_ids]
    values = [counts[c] for c in class_ids]
    n_bars = len(class_ids)
    colors = plt.cm.Set3(np.linspace(0, 1, max(n_bars, 1)))

    display_name = cfg.get("display_name", args.dataset.capitalize())
    fig, ax = plt.subplots(figsize=(max(10, n_bars * 1.2), 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="gray", linewidth=0.5)
    ax.set_ylabel("Pixel count", fontsize=12)
    ax.set_xlabel("Class", fontsize=12)
    ax.set_title(f"Pixel distribution by class — {display_name}", fontsize=14)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()

    for bar, v in zip(bars, values):
        if v > 0:
            pct = 100.0 * v / total
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{v:,}\n({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

    out_path = args.output or f"pixel_distribution_{args.dataset}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(f"Total pixels: {total:,} (excluded from plot: {sorted(exclude)})")
    for c in range(max_classes):
        if counts[c] > 0:
            pct = 100.0 * counts[c] / total
            name = class_names.get(c, str(c))
            tag = " [excluded from plot]" if c in exclude else ""
            print(f"  {name:25s}: {counts[c]:>12,} ({pct:5.2f}%){tag}")


if __name__ == "__main__":
    main()
