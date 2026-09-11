# Split manifests

One file per benchmark and split, listing every image used, one filename per line. Regenerate with:

```bash
SAM3_DATA_ROOT=/path/to/datasets python preprocessing/export_split_manifests.py
```

`manifest_summary.json` records the file count and a short SHA-256 digest of each listing, so you can
confirm you hold the same partition without diffing thousands of lines.

## Why these files are here

Five benchmarks use their official splits, which can be rebuilt from the providers' data plus the
tiling scripts in [`preprocessing/`](../preprocessing). Two cannot:

- **WHU Building** is our own 70/15/15 repartition of the 8188 official tiles, not the official
  4736/1036/2416 spatial split. The shuffling seed was never recorded and the tiles were renamed
  when the partition was made, so these manifests are the only complete record of it. Results on
  this split are not comparable with published WHU results; see the paper for why.
- **ISPRS Potsdam and Vaihingen** keep the official test areas, but the boundary between train and
  validation is a random 20 percent draw over the official *training* areas, made by
  [`preprocessing/make_internal_val_split.py`](../preprocessing/make_internal_val_split.py) with
  seed 1337. That split selects checkpoints only; no validation number is reported for them.

## Counts

| Dataset | Train | Val | Test | Split origin |
|---|---:|---:|---:|---|
| massachusetts_buildings | 1,233 | 36 | 90 | official, by source scene |
| massachusetts_roads | 9,972 | 126 | 441 | official, by source scene |
| potsdam | 2,765 | 691 | 2,016 | official test areas; internal train/val draw |
| vaihingen | 276 | 68 | 398 | official test areas; internal train/val draw |
| whu_building | 5,732 | 1,228 | 1,228 | **ours**, random over tiles |
| uavid | 8,000 | 2,800 | 150 | official, by sequence |
| loveda | 2,522 | 1,669 | 1,796 | official, by scene |

Train and validation counts are tiles. The UAVid test column counts native 3840x2160 frames; its
train and validation frames are cut into 40 tiles each. The LoveDA test split is listed for
completeness but is unlabelled, so the paper reports LoveDA on validation.
