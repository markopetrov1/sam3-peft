# Copyright (c) OpenMMLab. All rights reserved.
# Minimal, mmcv/mmengine-free version for Vaihingen.

import argparse
import glob
import math
import os
import os.path as osp
import tempfile
import zipfile

import cv2
import numpy as np
from tqdm import tqdm


# --- tiny helpers to replace mmengine ---
def mkdir_or_exist(path):
    os.makedirs(path, exist_ok=True)


class ProgressBar:
    def __init__(self, total):
        self.pbar = tqdm(total=total, ncols=80, desc="Clipping")

    def update(self):
        self.pbar.update(1)


# --- arg parsing stays the same ---
def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert vaihingen dataset to mmsegmentation format')
    parser.add_argument('dataset_path', help='vaihingen folder path')
    parser.add_argument('--tmp_dir', help='path of the temporary directory')
    parser.add_argument('-o', '--out_dir', help='output path')
    parser.add_argument(
        '--clip_size',
        type=int,
        help='clipped size of image after preparation',
        default=512)
    parser.add_argument(
        '--stride_size',
        type=int,
        help='stride of clipping original images',
        default=256)
    args = parser.parse_args()
    return args


def _imread(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return img


def _imwrite(path, arr):
    ok = cv2.imwrite(path, arr)
    if not ok:
        raise IOError(f"Failed to write {path}")


def clip_big_image(image_path, clip_save_dir, to_label=False):
    # Same logic as original mmseg script, but using OpenCV/tqdm.
    image = _imread(image_path)

    if image.ndim == 2:
        h, w = image.shape
        c = 1
    else:
        h, w, c = image.shape

    cs = args.clip_size
    ss = args.stride_size

    num_rows = math.ceil((h - cs) / ss) if math.ceil(
        (h - cs) / ss) * ss + cs >= h else math.ceil((h - cs) / ss) + 1
    num_cols = math.ceil((w - cs) / ss) if math.ceil(
        (w - cs) / ss) * ss + cs >= w else math.ceil((w - cs) / ss) + 1

    x, y = np.meshgrid(np.arange(num_cols + 1), np.arange(num_rows + 1))
    xmin = x * cs
    ymin = y * cs

    xmin = xmin.ravel()
    ymin = ymin.ravel()
    xmin_offset = np.where(xmin + cs > w, w - xmin - cs, np.zeros_like(xmin))
    ymin_offset = np.where(ymin + cs > h, h - ymin - cs, np.zeros_like(ymin))
    boxes = np.stack([
        xmin + xmin_offset, ymin + ymin_offset,
        np.minimum(xmin + cs, w),
        np.minimum(ymin + cs, h)
    ], axis=1)

    if to_label:
        # Same color map & hashing trick as the mmseg script.
        color_map = np.array([[0, 0, 0], [255, 255, 255], [255, 0, 0],
                              [255, 255, 0], [0, 255, 0], [0, 255, 255],
                              [0, 0, 255]], dtype=np.uint8)
        if image.ndim == 3:
            flatten_v = (image.reshape(-1, c) @ np.array([2, 3, 4]).reshape(3, 1))
            out = np.zeros_like(flatten_v)
            for idx, class_color in enumerate(color_map):
                value_idx = (class_color @ np.array([2, 3, 4]).reshape(3, 1))
                out[flatten_v == value_idx] = idx
            image = out.reshape(h, w)
        # if ndim==2 we assume labels are already indexed

    for box in boxes:
        start_x, start_y, end_x, end_y = box.astype(int)

        if to_label:
            clipped_image = image[start_y:end_y, start_x:end_x]
        else:
            tile = image[start_y:end_y, start_x:end_x]
            # ensure 3-channel for imagery if cv2 loaded grayscale
            if tile.ndim == 2:
                tile = np.repeat(tile[..., None], 3, axis=2)
            clipped_image = tile

        area_idx = osp.basename(image_path).split('_')[3].strip('.tif')
        _imwrite(
            osp.join(
                clip_save_dir,
                f'{area_idx}_{start_x}_{start_y}_{end_x}_{end_y}.png'
            ),
            clipped_image.astype(np.uint8)
        )


def main():
    splits = {
        'train': [
            'area1', 'area11', 'area13', 'area15', 'area17', 'area21',
            'area23', 'area26', 'area28', 'area3', 'area30', 'area32',
            'area34', 'area37', 'area5', 'area7'
        ],
        'val': [
            'area6', 'area24', 'area35', 'area16', 'area14', 'area22',
            'area10', 'area4', 'area2', 'area20', 'area8', 'area31', 'area33',
            'area27', 'area38', 'area12', 'area29'
        ],
    }

    dataset_path = args.dataset_path
    out_dir = osp.join('data', 'vaihingen') if args.out_dir is None else args.out_dir

    print('Making directories...')
    mkdir_or_exist(osp.join(out_dir, 'img_dir', 'train'))
    mkdir_or_exist(osp.join(out_dir, 'img_dir', 'val'))
    mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'train'))
    mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'val'))

    zipp_list = glob.glob(os.path.join(dataset_path, '*.zip'))
    print('Find the data', zipp_list)

    # single temp dir, same as original structure
    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir:
        for zipp in zipp_list:
            zip_file = zipfile.ZipFile(zipp)
            zip_file.extractall(tmp_dir)

            src_path_list = glob.glob(os.path.join(tmp_dir, '*.tif'))
            if 'ISPRS_semantic_labeling_Vaihingen' in zipp:
                # tops reside in 'top/*.tif'
                src_path_list = glob.glob(os.path.join(tmp_dir, 'top', '*.tif'))
            if 'ISPRS_semantic_labeling_Vaihingen_ground_truth_eroded_COMPLETE' in zipp:
                src_path_list = glob.glob(os.path.join(tmp_dir, '*.tif'))
                # delete unused area9 ground truth
                src_path_list = [p for p in src_path_list if 'area9' not in p]

            prog_bar = ProgressBar(len(src_path_list))
            for src_path in src_path_list:
                area_idx = osp.basename(src_path).split('_')[3].strip('.tif')
                data_type = 'train' if area_idx in splits['train'] else 'val'
                if 'noBoundary' in src_path:
                    dst_dir = osp.join(out_dir, 'ann_dir', data_type)
                    clip_big_image(src_path, dst_dir, to_label=True)
                else:
                    dst_dir = osp.join(out_dir, 'img_dir', data_type)
                    clip_big_image(src_path, dst_dir, to_label=False)
                prog_bar.update()

        print('Removing the temporary files...')

    print('Done!')


if __name__ == '__main__':
    args = parse_args()
    main()
