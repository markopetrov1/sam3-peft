#!/usr/bin/env python3
"""
Create proper train / val / test splits to fix data leakage.

Current (wrong): validation during training and evaluation both use the same
'val' set. This script:
  1. Renames current val/ -> test/ (unseen holdout for final evaluation).
  2. From train/, moves 20% of samples into a new val/ (for validation during training).

Result: train/ (80% of original train), val/ (20% of original train), test/ (former val).
Run once per dataset. Supports --dry-run to show counts without moving files.

Usage:
  python scripts/split_train_val_test.py --dataset potsdam [--root /path/to/potsdam_mmseg] [--dry-run]
  python scripts/split_train_val_test.py --dataset vaihingen [--dry-run]
"""

import argparse
import glob
import os
import random
import shutil

# Dataset location. Set SAM3_DATA_ROOT to the directory holding the prepared datasets;
# the paths below are resolved underneath it.
_DATA_ROOT = os.environ.get("SAM3_DATA_ROOT", "datasets")

# MMSeg layout
IMG_SUBDIR = "img_dir"
ANN_SUBDIR = "ann_dir"
IMG_EXTS = ("*.png", "*.PNG", "*.tif", "*.TIF", "*.tiff", "*.TIFF", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG")
VAL_FRAC = 0.20
SEED = 1337

DEFAULT_ROOTS = {
    "potsdam": os.path.join(_DATA_ROOT, "potsdam_mmseg"),
    "vaihingen": os.path.join(_DATA_ROOT, "vaihingen_mmseg"),
}


def get_split_paths(root: str, split: str):
    img_dir = os.path.join(root, IMG_SUBDIR, split)
    ann_dir = os.path.join(root, ANN_SUBDIR, split)
    return img_dir, ann_dir


def list_images(img_dir: str) -> list[str]:
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(img_dir, ext)))
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(
        description="Split dataset: rename val->test, move 20%% of train into val."
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=["potsdam", "vaihingen"],
        help="Dataset name (potsdam or vaihingen)",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help=f"Dataset root (default: {DEFAULT_ROOTS['potsdam']} for potsdam, etc.)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print image counts and percentages; do not move files",
    )
    args = parser.parse_args()

    root = args.root or DEFAULT_ROOTS[args.dataset]
    if not os.path.isdir(root):
        print(f"Error: root not found: {root}")
        return 1

    train_img_dir, train_ann_dir = get_split_paths(root, "train")
    val_img_dir, val_ann_dir = get_split_paths(root, "val")
    test_img_dir = os.path.join(root, IMG_SUBDIR, "test")
    test_ann_dir = os.path.join(root, ANN_SUBDIR, "test")

    if os.path.isdir(test_img_dir) or os.path.isdir(test_ann_dir):
        print("test/ already exists; assuming split was already applied. Exiting.")
        return 0

    if not os.path.isdir(train_img_dir):
        print(f"Error: train image dir not found: {train_img_dir}")
        return 1
    if not os.path.isdir(train_ann_dir):
        print(f"Error: train annotation dir not found: {train_ann_dir}")
        return 1

    train_imgs = list_images(train_img_dir)
    n_train = len(train_imgs)
    n_val_from_train = max(1, int(n_train * VAL_FRAC))
    n_train_after = n_train - n_val_from_train

    # Current val (will become test)
    if os.path.isdir(val_img_dir):
        current_val_imgs = list_images(val_img_dir)
        n_test = len(current_val_imgs)
    else:
        current_val_imgs = []
        n_test = 0

    total = n_train_after + n_val_from_train + n_test
    pct_train = 100.0 * n_train_after / total if total else 0
    pct_val = 100.0 * n_val_from_train / total if total else 0
    pct_test = 100.0 * n_test / total if total else 0

    print(f"Dataset: {args.dataset}")
    print(f"Root:    {root}")
    print()
    print("Current state:")
    print(f"  train: {n_train} images (will become {n_train_after} after moving {n_val_from_train} to val)")
    print(f"  val:   {n_test} images (will be renamed to test)")
    print()
    print("After split:")
    print(f"  train: {n_train_after} images ({pct_train:.1f}%)")
    print(f"  val:   {n_val_from_train} images ({pct_val:.1f}%) — from train")
    print(f"  test:  {n_test} images ({pct_test:.1f}%) — former val (unseen)")
    print()

    if args.dry_run:
        print("Dry run. No files moved.")
        return 0

    # --- Perform moves ---
    random.seed(SEED)
    train_imgs_shuffled = train_imgs.copy()
    random.shuffle(train_imgs_shuffled)
    to_val = set(train_imgs_shuffled[:n_val_from_train])

    # 1) Rename val -> test
    if os.path.isdir(val_img_dir):
        if os.path.exists(test_img_dir):
            print(f"Error: {test_img_dir} already exists. Remove it or run without --dry-run only once.")
            return 1
        if os.path.exists(test_ann_dir):
            print(f"Error: {test_ann_dir} already exists.")
            return 1
        shutil.move(val_img_dir, test_img_dir)
        print(f"Renamed: {val_img_dir} -> {test_img_dir}")
    if os.path.isdir(val_ann_dir):
        shutil.move(val_ann_dir, test_ann_dir)
        print(f"Renamed: {val_ann_dir} -> {test_ann_dir}")

    # 2) Create val dirs and move 20% from train to val
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_ann_dir, exist_ok=True)
    moved = 0
    for img_path in to_val:
        base = os.path.basename(img_path)
        ann_path = os.path.join(train_ann_dir, base)
        if not os.path.exists(ann_path):
            print(f"Warning: no annotation for {base}, skipping")
            continue
        dest_img = os.path.join(val_img_dir, base)
        dest_ann = os.path.join(val_ann_dir, base)
        shutil.move(img_path, dest_img)
        shutil.move(ann_path, dest_ann)
        moved += 1
    print(f"Moved {moved} image+annotation pairs from train to val")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
