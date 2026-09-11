# SAM3-PEFT

**Parameter-efficient fine-tuning of SAM 3 for remote-sensing semantic segmentation.**

Three PEFT methods (LoRA, frozen-backbone Linear Probing, FFT-prompt Adapter) on top of a vendored Meta SAM 3 image backbone, evaluated on seven public remote-sensing datasets (aerial / UAV / satellite). All training and evaluation runs from YAML configs.

![Potsdam test sample — SAM3 + LoRA (mIoU 0.88)](assets/example_potsdam_lora.png)

*Inference on a Potsdam test image. Best run in this repo: SAM3 + LoRA on Potsdam, mIoU **0.8806**.*

---

## Methods

| Method                | Module                            | Trainable params *(MA Buildings run)* | % of total |
|-----------------------|-----------------------------------|--------------------------------------:|----------:|
| `sam3_linear_probing` | `peft/sam3_linear_probing.py`     |                             2,299,652 |    0.50 % |
| `sam3_lora`           | `peft/sam3_lora.py` + `peft/lora.py` |                          4,539,139 |    0.98 % |
| `sam3_adapter`        | `peft/sam3_adapter.py` + `peft/adapter.py` |                       337,891 |    0.07 % |

A common factory `peft.build_peft_model(method=...)` returns a wrapper with a unified interface (`forward(images, multimask_output, image_size) → {"masks": ...}`, `save_parameters`, `load_parameters`).

## Datasets

All inputs are resized to **1008×1008** for both training and evaluation.

| Dataset                  | Type                | Train | Val   | Test  |
|--------------------------|---------------------|------:|------:|------:|
| Potsdam (ISPRS)          | aerial 5cm GSD      | 2,765 |   691 | 2,016 |
| Vaihingen (ISPRS)        | aerial 9cm GSD      |   276 |    68 |   398 |
| UAVid                    | UAV oblique 4K      | 8,000 | 2,800 |   150 |
| LoveDA                   | satellite (urban+rural) | 2,522 | 1,669 |   —   |
| Massachusetts Buildings  | aerial 1m GSD       | 1,233 |    36 |    90 |
| Massachusetts Roads      | aerial 1m GSD       | 9,972 |   126 |   441 |
| WHU Building             | aerial 0.3m GSD     | 5,732 | 1,228 | 1,228 |

Datasets follow the MMSeg-style layout `root/img_dir/{split}/` + `root/ann_dir/{split}/`, except UAVid and Massachusetts which use `root/{split}/{images,masks}/`. Loaders live in [`datasets.py`](datasets.py).

## Results

Best validation checkpoint, evaluated on the held-out test split (LoveDA reports val numbers since the public test set is unlabeled):

| Dataset                  | Linear Probing |  LoRA  | Adapter |
|--------------------------|:--------------:|:------:|:-------:|
| Potsdam                  |     0.7186     | 0.8806 | 0.8735  |
| Vaihingen                |     0.6917     | 0.8326 | 0.8152  |
| UAVid                    |     0.4554     | 0.6106 | 0.6619  |
| LoveDA                   |     0.4237     | 0.5663 | 0.5518  |
| Massachusetts Buildings  |     0.7381     | 0.8278 | 0.8087  |
| Massachusetts Roads      |     0.7048     | 0.8128 | 0.8077  |
| WHU Building             |     0.8877     | 0.9492 | 0.9403  |

Numbers are mIoU. Full per-class IoU and overall accuracy: see [`evaluation_results.md`](evaluation_results.md).

---

## Pre-trained weights

Best-checkpoint weights for all 21 runs (3 methods × 7 datasets) are available on Google Drive:

**[Download weights (Google Drive)](https://drive.google.com/drive/folders/1CII_0gT69e2aqylGJUG9XZxPQ9ot6Pfa?usp=sharing)**

The folder is organised as:

```
sam3_peft_weights/
├── linear_probing/          # ~8.8 MB each
│   ├── potsdam.pth
│   ├── vaihingen.pth
│   ├── uavid.pth
│   ├── loveda.pth
│   ├── massachusetts_buildings.pth
│   ├── massachusetts_roads.pth
│   └── whu_building.pth
├── lora/                    # ~27 MB each
│   └── ...
└── adapter/                 # ~1.4 MB each
    └── ...
```

Each file is the `best.pth` for that run (best validation mIoU checkpoint). To evaluate with a downloaded weight:

```bash
python test.py --config configs/sam3_lora_potsdam.yaml \
  --checkpoint /path/to/sam3_peft_weights/lora/potsdam.pth
```

---

## Setup

```bash
git clone <this-repo> sam3-peft && cd sam3-peft
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# Install PyTorch matching your CUDA, e.g.:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

The vendored Meta SAM 3 model and BPE assets live inside the repo (`sam3/` + `sam3/assets/`); no external SAM-3 install is required.

Set `dataset.root` in each config to the local path of your dataset before running, or leave the
configs alone and export `SAM3_DATA_ROOT`, which resolves each config's dataset directory name
underneath it:

```bash
export SAM3_DATA_ROOT=/path/to/datasets   # expects potsdam_mmseg/, WHU/, uavid/, ... underneath
```

Optional: set `model.sam3_checkpoint` and `model.bpe_path` to override the bundled defaults. The
SAM 3 backbone weights are downloaded from the `facebook/sam3` HuggingFace repository on first run.

## Train

```bash
python train.py --config configs/sam3_lora_potsdam.yaml
# Override GPU from CLI:
python train.py --config configs/sam3_lora_potsdam.yaml --gpu 1
```

Each run writes to `experiments/<experiment_name>_<timestamp>/`:

- `train.log` — full stdout / stderr
- `best.pth`, `last.pth`, `epoch_*.pth` — checkpoints
- `tb_logs/` — TensorBoard scalars
- `training_summary.txt` — final mIoU, OA, peak GPU memory, wall-clock time

## Evaluate

```bash
python test.py --config configs/sam3_lora_potsdam.yaml \
  --checkpoint experiments/<run>/best.pth
```

Saves an `eval_<timestamp>/eval.log` with per-class IoU, mIoU, OA. Pass `--save_preds` to write per-image prediction PNGs. Ten random side-by-side `image | gt | pred` panels are always saved under `eval_<timestamp>/visualizations/`.

## Evaluate with TTA

D4 test-time augmentation (8 views: identity + 90°/180°/270° rotations + h-flip combinations, merged by mean of softmax logits):

```bash
python test_tta.py --config configs/sam3_lora_potsdam.yaml \
  --checkpoint experiments/<run>/best.pth
```

## One-command train + evaluate

```bash
./train.sh configs/sam3_lora_potsdam.yaml
```

---

## Analysis scripts

These reproduce the evaluation and the protocol checks reported in the paper. None of them trains
anything; each loads an existing `best.pth` and runs forward passes only.

| Script | What it does |
|--------|--------------|
| `scripts/reevaluate.py` | Re-scores a checkpoint at **native ground-truth resolution** as well as at the 1008×1008 working resolution, and writes per-image confusion matrices plus a boundary-band / interior decomposition. `--score-ignore-channel` scores channel 0 as a class, which yields UAVid's 8-class (clutter-included) mIoU. `--band-swap` runs the input-channel ablations. |
| `scripts/uncertainty_analysis.py` | Image-level bootstrap confidence intervals and paired permutation tests between strategies, computed from the per-image confusion matrices. Quantifies test-set sampling uncertainty only, **not** seed-to-seed training variance. |
| `scripts/qualitative_figure.py` | Builds `image / ground truth / linear probing / LoRA / adapter` comparison figures, selecting tiles by method disagreement, worst-case IoU, or at random. |
| `scripts/duplicate_audit.py` | Perceptual-hash near-duplicate audit between train and test splits. |
| `scripts/calibrate_adjacency.py` | Calibrates an image-edge adjacency detector on Potsdam, where true tile adjacency is known from the filenames, and reports its power. Documents a **negative result**: the detector cannot resolve per-tile adjacency once multiple-comparison effects are controlled. |
| `scripts/split_train_val_test.py` | The script that produced the internal Potsdam/Vaihingen train/validation partition (seed 1337). |
| `scripts/compute_confusion_matrix.py` | Confusion matrix and per-class precision/recall plots. |

Note that `scripts/` was previously excluded by `.gitignore`, so earlier clones of this repository
did not contain any of it. That is fixed.

## Preprocessing

`preprocessing/` holds the scripts that turn each provider's download into the tiled MMSeg-style
layout the loaders expect, plus the script that made our internal validation split.

| Script | Purpose |
|--------|---------|
| `prepare_potsdam.py` / `prepare_vaihingen.py` | Cut the ISPRS orthophotos into 512x512 tiles |
| `prepare_massachusetts_buildings.py` / `prepare_massachusetts_roads.py` | Cut the 1500x1500 Mnih scenes into 512x512 tiles, taking the split from `metadata.csv` |
| `prepare_loveda.py` | Unpack LoveDA into the MMSeg layout |
| `make_internal_val_split.py` | Move a random 20% of the official ISPRS training tiles into a validation split (seed 1337). Test tiles are untouched |
| `export_split_manifests.py` | Write the exact file list of every split to `splits/` |

## Split manifests

`splits/<dataset>/{train,val,test}.txt` lists every image used, one filename per line, with counts
and digests in `splits/manifest_summary.json`. See [`splits/README.md`](splits/README.md).

These matter most for WHU Building, whose 70/15/15 partition is **ours** rather than the official
4736/1036/2416 spatial split, was made with an unrecorded seed, and renamed the tiles in the
process. The manifest is the only complete record of it, and WHU results are therefore not
comparable with published numbers on the official split.

## Dataset splits

| Dataset | Split used | Provenance |
|---------|-----------|------------|
| ISPRS Potsdam | official test (14 orthophotos, 2016 tiles) | Official ISPRS split; train/val is an 80/20 random tile partition (seed 1337) of the 24 official training orthophotos, used only for checkpoint selection |
| ISPRS Vaihingen | official test (17 orthophotos, 398 tiles) | as above, from the 16 official training orthophotos |
| Massachusetts Buildings/Roads | official test (10 / 49 source images) | Official Mnih split at source-image level; each 1500×1500 image cut into a 3×3 grid of 512×512 tiles |
| UAVid | official test (150 frames, seq21–30 & 38–42) | Official sequence split. Test ground truth has been publicly released by the dataset authors since 2023 |
| WHU Building | **our own** 70/15/15 repartition (5732/1228/1228) | **Not** the official 4736/1036/2416 spatial split. Numbers are therefore not comparable with published WHU results |
| LoveDA | official validation (1669 images) | Public test split is unlabelled; validation also served for checkpoint selection, so these values are optimistically biased |

## Project layout

```
.
├── train.py / test.py / test_tta.py / train.sh    # Entry points
├── datasets.py                                    # All dataset loaders + create_dataset(...)
├── peft/                                          # PEFT wrappers
│   ├── lora.py             # generic LoRA layers
│   ├── sam3_lora.py
│   ├── sam3_linear_probing.py
│   ├── adapter.py          # generic adapter / FFT prompt generator
│   └── sam3_adapter.py
├── sam3/                                          # Vendored Meta SAM 3 (model + BPE assets)
├── configs/                                       # 21 YAML configs (3 methods × 7 datasets)
├── utils/
│   ├── config.py           # YAML loader with attribute access
│   ├── losses.py           # Dice loss
│   ├── tta.py              # D4 test-time augmentation
│   └── run_log.py          # Timestamped run dirs + tee logging
├── experiments/                                   # Created by train.py / test.py (gitignored)
└── pre_weight/                                    # Pretrained SAM weights (gitignored)
```

## Configs

There are 21 configs in `configs/` (one per `{method × dataset}`). All follow the same shape:

```yaml
experiment:
  name: sam3_lora_potsdam
  output_dir: experiments
  seed: 1337
  gpu: "0"

dataset:
  type: potsdam            # potsdam | vaihingen | uavid | loveda
                           # massachusetts_buildings | massachusetts_roads | whu_building
  root: /path/to/dataset
  image_size: 1008
  num_classes: 6
  augment: true
  exclude_classes: []

model:
  method: sam3_lora        # sam3_lora | sam3_linear_probing | sam3_adapter
  sam3_checkpoint: null    # null = use bundled
  bpe_path: null
  rank: 8                  # LoRA-only
  alpha: 16                # LoRA-only
  dropout: 0.0             # LoRA-only
  # sam3_adapter-only:
  scale_factor: 32
  input_type: fft
  freq_nums: 0.25
  prompt_type: highpass
  tuning_stage: "1234"
  handcrafted_tune: true
  embedding_tune: true
  adaptor: adaptor

training:
  epochs: 50
  batch_size: 24
  num_workers: 4
  lr: 3.0e-4
  weight_decay: 0.01
  warmup_epochs: 2
  grad_clip_norm: 1.0
  amp: true
  save_every: 5
  val_every: 1
  log_every: 20
  early_stopping_patience: 10

resume:
  checkpoint: null
  epoch: 0
```

The only field that *must* be set per machine is `dataset.root`.

---

## Augmentations

Joint image+mask augmentations (training only, [`datasets.py`](datasets.py) → `_augment`):

- horizontal flip (p=0.5)
- vertical flip (p=0.5) — disabled for UAVid (oblique imagery)
- 90° / 180° / 270° rotation (uniform from {0,1,2,3})
- brightness, contrast, saturation jitter ±15 % (each p=0.5)

## Loss

Cross-entropy + Dice (1:1), with `ignore_index` honoured. Implementation: `utils/losses.py`.
