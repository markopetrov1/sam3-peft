#!/usr/bin/env python3
"""
Write the exact file list of every train, validation and test split to splits/<dataset>/<split>.txt.

Why this exists. Five of the seven benchmarks are used with their official splits, which anyone can
reconstruct from the dataset providers plus the tiling scripts in this directory. Two are not:

  * WHU Building is a 70/15/15 repartition of the 8188 official tiles that we made ourselves. The
    shuffling seed was not recorded and the tiles were renamed in the process, so the partition is
    not reproducible from a description. The manifest is the only complete record of it.
  * Potsdam and Vaihingen keep the official test tiles, but their train/validation boundary comes
    from make_internal_val_split.py with seed 1337, applied to the official training areas.

Listing every split makes all of this checkable rather than merely described, and lets a reader
reproduce our numbers on the same partition even where it departs from the published protocol.

Usage
-----
    SAM3_DATA_ROOT=/path/to/datasets python preprocessing/export_split_manifests.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Where each dataset keeps its images, per split, relative to the dataset root.
LAYOUTS = {
    "potsdam":                 ("potsdam_mmseg",                 "img_dir/{split}"),
    "vaihingen":               ("vaihingen_mmseg",               "img_dir/{split}"),
    "loveda":                  ("loveDA_mmseg",                  "img_dir/{split}"),
    "massachusetts_buildings": ("massachusetts_buildings_mmseg", "img_dir/{split}"),
    "massachusetts_roads":     ("massachusetts_roads_mmseg",     "img_dir/{split}"),
    "whu_building":            ("WHU",                           "{split}/Image"),
    "uavid":                   ("uavid",                         "{split}/images"),
}

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=os.environ.get("SAM3_DATA_ROOT", "datasets"))
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "splits"))
    args = ap.parse_args()

    summary = {}
    for name, (subdir, pattern) in sorted(LAYOUTS.items()):
        entry = {}
        for split in ("train", "val", "test"):
            d = os.path.join(args.data_root, subdir, pattern.format(split=split))
            if not os.path.isdir(d):
                continue
            files = sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXTS))
            if not files:
                continue
            os.makedirs(os.path.join(args.out, name), exist_ok=True)
            path = os.path.join(args.out, name, f"{split}.txt")
            with open(path, "w") as fh:
                fh.write("\n".join(files) + "\n")
            # A digest of the listing lets anyone confirm they hold the same partition without
            # diffing thousands of lines.
            digest = hashlib.sha256("\n".join(files).encode()).hexdigest()[:16]
            entry[split] = {"n": len(files), "sha256_16": digest}
            print(f"  {name:24s} {split:5s} {len(files):6d} files  sha256:{digest}")
        if entry:
            summary[name] = entry

    with open(os.path.join(args.out, "manifest_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote manifests under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
