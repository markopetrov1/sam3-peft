#!/usr/bin/env python3
"""
Export image + segmentation overlay samples for LoveDA, UAVid, Massachusetts
Buildings, Massachusetts Roads, or WHU Building (for paper / report figures).

Pixel-perfect: overlay keeps original image where mask is 0 (boundary/unlabeled);
only class pixels are painted with palette colors. Side-by-side = image | overlay,
saved with PIL (no interpolation, no black lines).

Usage:
    python scripts/export_thesis_figures.py loveda -o loveda_figures --count 3
    python scripts/export_thesis_figures.py uavid -o uavid_figures --count 3
    python scripts/export_thesis_figures.py massachusetts_buildings -o ma_buildings_figures --count 3
    python scripts/export_thesis_figures.py massachusetts_roads -o ma_roads_figures --count 3
    python scripts/export_thesis_figures.py whu_building -o whu_building_figures --count 3
"""

import argparse
import os
import random

import numpy as np
from PIL import Image

# Dataset location. Set SAM3_DATA_ROOT to the directory holding the prepared datasets;
# the paths below are resolved underneath it.
_DATA_ROOT = os.environ.get("SAM3_DATA_ROOT", "datasets")

# LoveDA: 0=background, 1=background, 2=building, 3=road, 4=water, 5=barren, 6=forest, 7=agriculture
LOVEDA_PALETTE = np.array([
    [0, 0, 0],
    [128, 128, 128],
    [255, 0, 0],
    [255, 165, 0],
    [0, 0, 255],
    [139, 69, 19],
    [0, 128, 0],
    [255, 255, 0],
    [255, 0, 255],
], dtype=np.uint8)

# UAVid: 0=unlabeled, 1=building, 2=road, 3=static car, 4=tree, 5=low veg, 6=human, 7=moving car, 8=clutter
UAVID_PALETTE = np.array([
    [0, 0, 0],
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [0, 255, 255],
    [255, 0, 255],
    [128, 128, 0],
    [128, 0, 128],
], dtype=np.uint8)

# Massachusetts Buildings / Roads: masks use 0=background, 255=class; we remap to 0,1 for palette index
MASS_BUILDINGS_PALETTE = np.array([[0, 0, 0], [255, 0, 0]], dtype=np.uint8)   # background, building
MASS_ROADS_PALETTE = np.array([[0, 0, 0], [0, 0, 255]], dtype=np.uint8)         # background, road

# WHU Building: same as Massachusetts Buildings (0=background, 255=building)
WHU_BUILDING_PALETTE = np.array([[0, 0, 0], [255, 0, 0]], dtype=np.uint8)

DATASET_ROOTS = {
    "loveda": os.path.join(_DATA_ROOT, "loveDA_mmseg"),
    "uavid": os.path.join(_DATA_ROOT, "uavid"),
    "massachusetts_buildings": os.path.join(_DATA_ROOT, "massachusetts_buildings_mmseg"),
    "massachusetts_roads": os.path.join(_DATA_ROOT, "massachusetts_roads_mmseg"),
    "whu_building": os.path.join(_DATA_ROOT, "WHU"),
}


def get_paired_tiles_mmseg(root: str, split: str):
    """MMSeg layout: img_dir/split, ann_dir/split (LoveDA, Massachusetts Buildings, Massachusetts Roads)."""
    img_dir = os.path.join(root, "img_dir", split)
    ann_dir = os.path.join(root, "ann_dir", split)
    if not os.path.isdir(img_dir) or not os.path.isdir(ann_dir):
        return []
    pairs = []
    for name in sorted(os.listdir(img_dir)):
        if not name.lower().endswith((".png", ".jpg")):
            continue
        ann_path = os.path.join(ann_dir, name)
        if os.path.isfile(ann_path):
            pairs.append((os.path.join(img_dir, name), ann_path))
    return pairs


def get_paired_tiles_whu(root: str, split: str):
    """WHU Building: root/split/Image, root/split/Mask."""
    img_dir = os.path.join(root, split, "Image")
    ann_dir = os.path.join(root, split, "Mask")
    if not os.path.isdir(img_dir) or not os.path.isdir(ann_dir):
        return []
    pairs = []
    for name in sorted(os.listdir(img_dir)):
        if not name.lower().endswith((".png", ".jpg")):
            continue
        ann_path = os.path.join(ann_dir, name)
        if os.path.isfile(ann_path):
            pairs.append((os.path.join(img_dir, name), ann_path))
    return pairs


def get_paired_tiles_uavid(root: str, split: str):
    """UAVid: root/split/images, root/split/masks."""
    img_dir = os.path.join(root, split, "images")
    ann_dir = os.path.join(root, split, "masks")
    if not os.path.isdir(img_dir) or not os.path.isdir(ann_dir):
        return []
    pairs = []
    for name in sorted(os.listdir(img_dir)):
        if not name.lower().endswith((".png", ".jpg")):
            continue
        ann_path = os.path.join(ann_dir, name)
        if os.path.isfile(ann_path):
            pairs.append((os.path.join(img_dir, name), ann_path))
    return pairs


def overlay_segmentation_on_image(image: np.ndarray, mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Original image with only class pixels painted; mask==0 keeps original color."""
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    out = np.array(image, dtype=np.uint8, copy=True)
    n = len(palette)
    for i in range(1, n):
        out[mask == i] = palette[i]
    return out


def colorize_mask_pixel_perfect(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Full mask RGB from palette (for standalone mask PNG)."""
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(len(palette)):
        out[mask == i] = palette[i]
    return out


def main():
    parser = argparse.ArgumentParser(description="Export thesis figures for LoveDA, UAVid, Massachusetts Buildings, or Massachusetts Roads.")
    parser.add_argument(
        "dataset",
        type=str,
        choices=["loveda", "uavid", "massachusetts_buildings", "massachusetts_roads", "whu_building"],
        help="Dataset name",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Output folder (default: <dataset>_thesis_figures)",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Override dataset root",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Number of samples (default: 3)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=("train", "val", "test"),
        default="train",
        help="Split to sample from (default: train)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="Start sample index (default: 1). E.g. --start-index 4 --count 2 → sample_4, sample_5",
    )
    args = parser.parse_args()

    root = args.root or DATASET_ROOTS[args.dataset]
    if args.dataset == "loveda":
        pairs = get_paired_tiles_mmseg(root, args.split)
        palette = LOVEDA_PALETTE
        mask_remap_255_to_1 = False
    elif args.dataset == "uavid":
        pairs = get_paired_tiles_uavid(root, args.split)
        palette = UAVID_PALETTE
        mask_remap_255_to_1 = False
    elif args.dataset == "massachusetts_buildings":
        pairs = get_paired_tiles_mmseg(root, args.split)
        palette = MASS_BUILDINGS_PALETTE
        mask_remap_255_to_1 = True
    elif args.dataset == "massachusetts_roads":
        pairs = get_paired_tiles_mmseg(root, args.split)
        palette = MASS_ROADS_PALETTE
        mask_remap_255_to_1 = True
    elif args.dataset == "whu_building":
        pairs = get_paired_tiles_whu(root, args.split)
        palette = WHU_BUILDING_PALETTE
        mask_remap_255_to_1 = True
    else:
        pairs = get_paired_tiles_uavid(root, args.split)
        palette = UAVID_PALETTE
        mask_remap_255_to_1 = False

    if len(pairs) < args.count:
        raise SystemExit(f"Only {len(pairs)} pairs found for {args.dataset}/{args.split}, need {args.count}")

    out_dir = args.output_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{args.dataset}_thesis_figures")
    os.makedirs(out_dir, exist_ok=True)

    random.seed(args.seed)
    chosen = random.sample(pairs, args.count)

    for i, (img_path, ann_path) in enumerate(chosen, start=args.start_index):
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(ann_path))
        if mask_remap_255_to_1:
            # Massachusetts masks: 0=background, 255=class → remap to 0,1 for palette index
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            mask = np.where(mask == 255, 1, 0).astype(np.uint8)

        overlay = overlay_segmentation_on_image(image, mask, palette)
        mask_rgb = colorize_mask_pixel_perfect(mask, palette)

        base = f"{args.dataset}_sample_{i}"
        Image.fromarray(image).save(os.path.join(out_dir, f"{base}_image.png"))
        Image.fromarray(mask_rgb).save(os.path.join(out_dir, f"{base}_mask.png"))
        Image.fromarray(overlay).save(os.path.join(out_dir, f"{base}_segmentation_overlay.png"))
        combined = np.concatenate([image, overlay], axis=1)
        Image.fromarray(combined).save(os.path.join(out_dir, f"{base}_image_and_mask.png"))
        print(f"Saved: {base}_image.png, {base}_mask.png, {base}_segmentation_overlay.png, {base}_image_and_mask.png")

    print(f"All figures written to: {out_dir}")


if __name__ == "__main__":
    main()
