This repository is now focused on **SAM3** training for remote sensing segmentation.

## Current status

- **Methods:** `sam3_lora`, `sam3_linear_probing` (frozen backbone + linear head, see `15_SAM3_linearn_probing.ipynb`)
- **Datasets:** Potsdam and Vaihingen (MMSeg layout)
- **Entry points:** `train.py`, `test.py`, `train.sh`

## Configs

- `configs/sam3_lora_potsdam.yaml`, `configs/sam3_lora_vaihingen.yaml`
- `configs/sam3_linear_probing_potsdam.yaml`, `configs/sam3_linear_probing_vaihingen.yaml`

Set `dataset.root` in the chosen config before running.

## Setup

1. Install this repo dependencies (`requirements.txt`)
2. SAM3 and PEFT code live **inside the repo**: `sam3/` (model + assets) and `peft/` (LoRA, linear probing, and wrappers). No external path is required.

## Train

```bash
python train.py --config configs/sam3_lora_potsdam.yaml
```

## Evaluate

```bash
python test.py --config configs/sam3_lora_potsdam.yaml \
  --checkpoint experiments/sam3_lora_potsdam/best.pth
```

## One-command train + eval

```bash
./train.sh configs/sam3_lora_potsdam.yaml
```

## Project layout

- `sam3/` — SAM3 model package (model_builder, `sam3/assets` for BPE). No external path.
- `peft/` — All PEFT methods and SAM3 wrappers:
  - `peft/lora.py` — Generic LoRA layers and `apply_lora_to_model` (model-agnostic).
  - `peft/sam3_lora.py` — SAM3 + LoRA + segmentation head (uses `peft.lora`).
  - `peft/sam3_linear_probing.py` — Frozen SAM3 + linear head.
  - (future) `peft/adapter.py`, `peft/sam3_adapter.py` for adapter method.
- `configs/` — Set `dataset.root` and optional `model.sam3_checkpoint` / `model.bpe_path` (null = in-repo defaults).
