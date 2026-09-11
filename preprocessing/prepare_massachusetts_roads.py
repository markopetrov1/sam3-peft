# Massachusetts Roads Dataset — 1500x1500 -> 512x512 tiles (MMSeg format).
# Split is taken from metadata.csv (source of truth). Same tiling as Buildings:
# clip_size=512, stride_size=256 (overlapping).

import argparse
import csv
import math
import os
import os.path as osp
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

METADATA_CSV = "metadata.csv"
COL_SPLIT = "split"
COL_TIFF_IMAGE = "tiff_image_path"
COL_TIF_LABEL = "tif_label_path"


def mkdir_or_exist(path):
    os.makedirs(path, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Massachusetts Roads Dataset to MMSeg format (512x512 tiles) using metadata.csv"
    )
    parser.add_argument(
        "dataset_path",
        help="Path to Massachusetts-Roads-Dataset root (contains metadata.csv, tiff/)",
    )
    parser.add_argument(
        "-o", "--out_dir",
        default=None,
        help="Output root (default: dataset_path/../massachusetts_roads_mmseg)",
    )
    parser.add_argument(
        "--clip_size",
        type=int,
        default=512,
        help="Tile size (default: 512)",
    )
    parser.add_argument(
        "--stride_size",
        type=int,
        default=256,
        help="Stride for sliding window (default: 256, i.e. 50%% overlap)",
    )
    args = parser.parse_args()
    return args


def _imread_tiff(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        path_alt = path.replace(".tiff", ".tif") if path.lower().endswith(".tiff") else path.replace(".tif", ".tiff")
        img = cv2.imread(path_alt, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return img


def _read_image(path):
    if path.lower().endswith((".tif", ".tiff")):
        return _imread_tiff(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return img


def _imwrite(path, arr):
    ok = cv2.imwrite(path, arr)
    if not ok:
        raise IOError(f"Failed to write {path}")


def load_metadata(dataset_path):
    """Load metadata.csv and return dict split -> list of (tiff_image_path, tif_label_path)."""
    csv_path = osp.join(dataset_path, METADATA_CSV)
    if not osp.isfile(csv_path):
        raise FileNotFoundError(f"Missing {METADATA_CSV} at {csv_path}")
    by_split = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = (row.get(COL_SPLIT) or "").strip().lower()
            img_rel = (row.get(COL_TIFF_IMAGE) or "").strip()
            lab_rel = (row.get(COL_TIF_LABEL) or "").strip()
            if not split or not img_rel or not lab_rel:
                continue
            if split not in ("train", "val", "test"):
                continue
            by_split[split].append((img_rel, lab_rel))
    return dict(by_split)


def clip_big_image(image_path, label_path, clip_save_img_dir, clip_save_ann_dir, args):
    """Crop image and label with the same grid; write PNG tiles."""
    image = _read_image(image_path)
    label = _read_image(label_path)
    if label.ndim == 3:
        label = label[:, :, 0]
    # Normalize label to 0/255 if needed (some TIFFs may be 0/1)
    if label.max() == 1:
        label = label * 255
    label = label.astype(np.uint8)
    h, w = image.shape[:2]
    if label.shape[:2] != (h, w):
        raise ValueError(
            f"Shape mismatch: image {image_path} {image.shape[:2]} vs "
            f"label {label_path} {label.shape[:2]}"
        )

    cs = args.clip_size
    ss = args.stride_size
    num_rows = (
        math.ceil((h - cs) / ss)
        if math.ceil((h - cs) / ss) * ss + cs >= h
        else math.ceil((h - cs) / ss) + 1
    )
    num_cols = (
        math.ceil((w - cs) / ss)
        if math.ceil((w - cs) / ss) * ss + cs >= w
        else math.ceil((w - cs) / ss) + 1
    )
    x, y = np.meshgrid(np.arange(num_cols + 1), np.arange(num_rows + 1))
    xmin = x.ravel() * cs
    ymin = y.ravel() * cs
    xmin_offset = np.where(xmin + cs > w, w - xmin - cs, np.zeros_like(xmin))
    ymin_offset = np.where(ymin + cs > h, h - ymin - cs, np.zeros_like(ymin))
    boxes = np.stack(
        [
            xmin + xmin_offset,
            ymin + ymin_offset,
            np.minimum(xmin + cs, w),
            np.minimum(ymin + cs, h),
        ],
        axis=1,
    )

    base = osp.splitext(osp.basename(image_path))[0]
    for box in boxes:
        start_x, start_y, end_x, end_y = box.astype(int)
        img_tile = image[start_y:end_y, start_x:end_x]
        if img_tile.ndim == 2:
            img_tile = np.repeat(img_tile[..., None], 3, axis=2)
        ann_tile = label[start_y:end_y, start_x:end_x]
        out_name = f"{base}_{start_x}_{start_y}_{end_x}_{end_y}.png"
        _imwrite(osp.join(clip_save_img_dir, out_name), img_tile.astype(np.uint8))
        _imwrite(osp.join(clip_save_ann_dir, out_name), ann_tile.astype(np.uint8))


def main():
    args = parse_args()
    dataset_path = osp.abspath(args.dataset_path)
    if args.out_dir is None:
        out_dir = osp.join(osp.dirname(dataset_path), "massachusetts_roads_mmseg")
    else:
        out_dir = osp.abspath(args.out_dir)

    by_split = load_metadata(dataset_path)
    if not by_split:
        raise RuntimeError("No rows found in metadata.csv for train/val/test")

    for split in ("train", "val", "test"):
        mkdir_or_exist(osp.join(out_dir, "img_dir", split))
        mkdir_or_exist(osp.join(out_dir, "ann_dir", split))

    for split, pairs in by_split.items():
        clip_img_dir = osp.join(out_dir, "img_dir", split)
        clip_ann_dir = osp.join(out_dir, "ann_dir", split)
        for img_rel, lab_rel in tqdm(pairs, desc=f"Clip {split}"):
            img_path = osp.join(dataset_path, img_rel)
            lab_path = osp.join(dataset_path, lab_rel)
            if not osp.isfile(img_path):
                print(f"Missing image: {img_path}, skip")
                continue
            if not osp.isfile(lab_path):
                print(f"Missing label: {lab_path}, skip")
                continue
            try:
                clip_big_image(img_path, lab_path, clip_img_dir, clip_ann_dir, args)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                raise

    print("Done. Output:", out_dir)


if __name__ == "__main__":
    main()
