#!/usr/bin/env python3
"""Cut the eroded ISPRS reference (the *_noBoundary rasters distributed with the benchmark) into
the same 512x512 test tiles the models are evaluated on.

Tile names carry their pixel box as <area>_<x0>_<y0>_<x1>_<y1>.png, so each eroded tile is the
same window of the eroded raster. Colours are mapped with the colour table used to build the
training data (preprocessing/prepare_potsdam.py, BGR order because cv2 reads BGR); black, which
marks the eroded boundary, becomes 0 and is ignored like every other unlabelled pixel.

A cross-check reports how many stored annotation tiles are identical to the eroded and to the
complete reference. For both ISPRS benchmarks every stored tile (training and test) turned out to
be the eroded reference, so the manuscript protocol already follows the benchmark's eroded-boundary
convention; --complete-out-dir writes the complete reference for the boundary analysis.
"""
import argparse, glob, os, re, sys, zipfile, tempfile
import cv2
import numpy as np

COLOR_MAP_BGR = np.array([[0, 0, 0], [255, 255, 255], [255, 0, 0], [255, 255, 0],
                          [0, 255, 0], [0, 255, 255], [0, 0, 255]], dtype=np.int64)


def to_index(img_bgr):
    h, w, _ = img_bgr.shape
    key = img_bgr.reshape(-1, 3).astype(np.int64) @ np.array([2, 3, 4])
    out = np.zeros(h * w, dtype=np.uint8)
    for idx, col in enumerate(COLOR_MAP_BGR):
        out[key == col @ np.array([2, 3, 4])] = idx
    return out.reshape(h, w)


def area_of(name, dataset):
    stem = os.path.basename(name)[:-4]
    parts = stem.split("_")
    if dataset == "potsdam":            # 2_13_x0_y0_x1_y1
        return f"{parts[0]}_{parts[1]}", tuple(map(int, parts[2:6]))
    return parts[0], tuple(map(int, parts[1:5]))   # area10_x0_y0_x1_y1


def raster_name(area, dataset, eroded):
    if dataset == "potsdam":
        return f"top_potsdam_{area}_label{'_noBoundary' if eroded else ''}.tif"
    return f"top_mosaic_09cm_{area}{'_noBoundary' if eroded else ''}.tif"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["potsdam", "vaihingen"], required=True)
    ap.add_argument("--ann-dir", required=True, help="stored test annotation tiles")
    ap.add_argument("--eroded-dir", required=True, help="folder of *_noBoundary rasters")
    ap.add_argument("--complete-zip", required=True, help="zip of the complete reference")
    ap.add_argument("--out-dir", required=True, help="eroded tiles are written here")
    ap.add_argument("--complete-out-dir", default=None,
                    help="optionally also write tiles of the complete reference here")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    if a.complete_out_dir:
        os.makedirs(a.complete_out_dir, exist_ok=True)

    tiles = sorted(glob.glob(os.path.join(a.ann_dir, "*.png")))
    by_area = {}
    for t in tiles:
        area, box = area_of(t, a.dataset)
        by_area.setdefault(area, []).append((t, box))

    zf = zipfile.ZipFile(a.complete_zip)
    members = {os.path.basename(m): m for m in zf.namelist()}
    n_mismatch = n_checked = n_eq_eroded = 0
    eroded_px = valid_px = 0
    for area, items in sorted(by_area.items()):
        # cross-check against the complete reference
        with tempfile.TemporaryDirectory() as td:
            m = members[raster_name(area, a.dataset, eroded=False)]
            zf.extract(m, td)
            full = to_index(cv2.imread(os.path.join(td, m), cv2.IMREAD_UNCHANGED))
        ero = to_index(cv2.imread(os.path.join(a.eroded_dir, raster_name(area, a.dataset, True)),
                                  cv2.IMREAD_UNCHANGED))
        assert full.shape == ero.shape, (area, full.shape, ero.shape)
        for t, (x0, y0, x1, y1) in items:
            stored = cv2.imread(t, cv2.IMREAD_UNCHANGED)
            if stored.ndim == 3:
                stored = stored[:, :, 0]
            n_checked += 1
            if not np.array_equal(stored, full[y0:y1, x0:x1]):
                n_mismatch += 1
            e = ero[y0:y1, x0:x1]
            n_eq_eroded += int(np.array_equal(stored, e))
            if a.complete_out_dir:
                cv2.imwrite(os.path.join(a.complete_out_dir, os.path.basename(t)),
                            full[y0:y1, x0:x1])
            eroded_px += int(((e == 0) & (stored != 0)).sum())
            valid_px += int((stored != 0).sum())
            cv2.imwrite(os.path.join(a.out_dir, os.path.basename(t)), e)
        print(f"{area}: {len(items)} tiles, raster {full.shape}")
    print(f"cross-check: {n_mismatch} of {n_checked} stored tiles differ from the complete "
          f"reference; {n_eq_eroded} of {n_checked} are identical to the eroded reference")
    print(f"eroded-away share of labelled stored-tile pixels: {eroded_px / max(valid_px, 1):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
