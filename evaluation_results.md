# Evaluation results (mIoU, OA & per-class IoU)

Parsed from `eval.log` files (and loveda `train.log`, which has no eval run) under `experiments/`. Re-run `scripts/extract_evaluation_tables.py` to refresh.

---

## Dataset setup (resolution & split sizes)

All images are resized to **1008×1008** (config `dataset.image_size`) for training and evaluation.

| Dataset | Resolution (crop) | Train | Val | Test |
|---------|-------------------|-------|-----|------|
| massachusetts_buildings | 1008×1008 | 1,233 | 36 | 90 |
| massachusetts_roads | 1008×1008 | 9,972 | 126 | 441 |
| potsdam | 1008×1008 | 2,765 | 691 | 2,016 |
| uavid | 1008×1008 | 8,000 | 2,800 | 150 |
| vaihingen | 1008×1008 | 276 | 68 | 398 |
| loveda | 1008×1008 | 2,522 | 1,669 | — |
| whu_building | 1008×1008 | 5,732 | 1,228 | 1,228 |

---

## Trainable parameters (by method)

| Method | Trainable params | Total params | Trainable % |
|--------|------------------|--------------|-------------|
| Linear Probing | 2,299,652 | 456,339,460 | 0.50% |
| LoRA | 4,539,139 | 460,877,828 | 0.98% |
| Adapter | 337,891 | 456,676,580 | 0.07% |

*(From Massachusetts Buildings runs; slight variation with num_classes.)*

---

## Summary table (all runs)

| Dataset | Method | mIoU | OA |
|---------|--------|------|-----|
| loveda | Linear Probing | 0.4237 | 0.5838 |
| loveda | LoRA | 0.5663 | 0.7411 |
| loveda | Adapter | 0.5518 | 0.7288 |
| massachusetts_buildings | Linear Probing | 0.7381 | 0.9080 |
| massachusetts_buildings | LoRA | 0.8278 | 0.9415 |
| massachusetts_buildings | Adapter | 0.8087 | 0.9335 |
| massachusetts_roads | Linear Probing | 0.7048 | 0.9638 |
| massachusetts_roads | LoRA | 0.8128 | 0.9798 |
| massachusetts_roads | Adapter | 0.8077 | 0.9790 |
| potsdam | Linear Probing | 0.7186 | 0.8170 |
| potsdam | LoRA | 0.8806 | 0.9306 |
| potsdam | Adapter | 0.8735 | 0.9293 |
| uavid | Linear Probing | 0.4554 | 0.7440 |
| uavid | LoRA | 0.6106 | 0.8697 |
| uavid | Adapter | 0.6619 | 0.8932 |
| vaihingen | Linear Probing | 0.6917 | 0.8186 |
| vaihingen | LoRA | 0.8326 | 0.9101 |
| vaihingen | Adapter | 0.8152 | 0.9048 |
| whu_building | Linear Probing | 0.8877 | 0.9682 |
| whu_building | LoRA | 0.9492 | 0.9862 |
| whu_building | Adapter | 0.9403 | 0.9837 |

---

## Comparison by dataset (methods as columns)

| Dataset | Linear Probing (mIoU) | Linear Probing (OA) | LoRA (mIoU) | LoRA (OA) | Adapter (mIoU) | Adapter (OA) |
|---------|----------|----------|----------|----------|----------|----------|
| loveda | 0.4237 | 0.5838 | 0.5663 | 0.7411 | 0.5518 | 0.7288 |
| massachusetts_buildings | 0.7381 | 0.9080 | 0.8278 | 0.9415 | 0.8087 | 0.9335 |
| massachusetts_roads | 0.7048 | 0.9638 | 0.8128 | 0.9798 | 0.8077 | 0.9790 |
| potsdam | 0.7186 | 0.8170 | 0.8806 | 0.9306 | 0.8735 | 0.9293 |
| uavid | 0.4554 | 0.7440 | 0.6106 | 0.8697 | 0.6619 | 0.8932 |
| vaihingen | 0.6917 | 0.8186 | 0.8326 | 0.9101 | 0.8152 | 0.9048 |
| whu_building | 0.8877 | 0.9682 | 0.9492 | 0.9862 | 0.9403 | 0.9837 |

---

## Per-class IoU by dataset

### loveda

| Class | Linear Probing | LoRA | Adapter |
|-------|----------|----------|----------|
| background | 0.3271 | 0.5533 | 0.5473 |
| building | 0.5934 | 0.6925 | 0.6617 |
| road | 0.5107 | 0.5874 | 0.5869 |
| water | 0.5649 | 0.7087 | 0.7054 |
| barren | 0.1676 | 0.3656 | 0.3472 |
| forest | 0.3282 | 0.4126 | 0.3984 |
| agriculture | 0.4737 | 0.6441 | 0.6158 |
| **mIoU** | **0.4237** | **0.5663** | **0.5518** |
| OA | 0.5838 | 0.7411 | 0.7288 |

### massachusetts_buildings

| Class | Linear Probing | LoRA | Adapter |
|-------|----------|----------|----------|
| background | 0.8946 | 0.9308 | 0.9216 |
| building | 0.5817 | 0.7249 | 0.6958 |
| **mIoU** | **0.7381** | **0.8278** | **0.8087** |
| OA | 0.9080 | 0.9415 | 0.9335 |

### massachusetts_roads

| Class | Linear Probing | LoRA | Adapter |
|-------|----------|----------|----------|
| background | 0.9627 | 0.9790 | 0.9782 |
| road | 0.4469 | 0.6465 | 0.6372 |
| **mIoU** | **0.7048** | **0.8128** | **0.8077** |
| OA | 0.9638 | 0.9798 | 0.9790 |

### potsdam

| Class | Linear Probing | LoRA | Adapter |
|-------|----------|----------|----------|
| impervious surface | 0.7268 | 0.9070 | 0.9051 |
| building | 0.7778 | 0.9612 | 0.9567 |
| low vegetation | 0.6002 | 0.8078 | 0.8056 |
| tree | 0.6542 | 0.8089 | 0.7985 |
| car | 0.8338 | 0.9182 | 0.9016 |
| **mIoU** | **0.7186** | **0.8806** | **0.8735** |
| OA | 0.8170 | 0.9306 | 0.9293 |

### uavid

| Class | Linear Probing | LoRA | Adapter |
|-------|----------|----------|----------|
| building | 0.6686 | 0.8962 | 0.9191 |
| road | 0.6596 | 0.8176 | 0.8391 |
| tree | 0.5989 | 0.7652 | 0.7915 |
| low vegetation | 0.4111 | 0.6231 | 0.6858 |
| moving car | 0.3233 | 0.5047 | 0.5230 |
| static car | 0.4027 | 0.5395 | 0.7027 |
| human | 0.1232 | 0.1276 | 0.1722 |
| **mIoU** | **0.4554** | **0.6106** | **0.6619** |
| OA | 0.7440 | 0.8697 | 0.8932 |

### vaihingen

| Class | Linear Probing | LoRA | Adapter |
|-------|----------|----------|----------|
| impervious surface | 0.7215 | 0.8784 | 0.8721 |
| building | 0.8128 | 0.9322 | 0.9143 |
| low vegetation | 0.4987 | 0.7427 | 0.7323 |
| tree | 0.7262 | 0.8192 | 0.8157 |
| car | 0.6991 | 0.7905 | 0.7415 |
| **mIoU** | **0.6917** | **0.8326** | **0.8152** |
| OA | 0.8186 | 0.9101 | 0.9048 |

### whu_building

| Class | Linear Probing | LoRA | Adapter |
|-------|----------|----------|----------|
| background | 0.9632 | 0.9838 | 0.9809 |
| building | 0.8123 | 0.9146 | 0.8997 |
| **mIoU** | **0.8877** | **0.9492** | **0.9403** |
| OA | 0.9682 | 0.9862 | 0.9837 |

---

## Parsed log files

- `experiments/sam3_adapter_loveda/train.log` — loveda / Adapter: mIoU=0.5518, OA=0.7288
- `experiments/sam3_linear_probing_loveda/train.log` — loveda / Linear Probing: mIoU=0.4237, OA=0.5838
- `experiments/sam3_lora_loveda/train.log` — loveda / LoRA: mIoU=0.5663, OA=0.7411
- `experiments/sam3_adapter_massachusetts_buildings/sam3_adapter_massachusetts_buildings_eval_2026-03-24_10-40-26/eval.log` — massachusetts_buildings / Adapter: mIoU=0.8087, OA=0.9335
- `experiments/sam3_linear_probing_massachusetts_buildings/sam3_linear_probing_massachusetts_buildings_eval_2026-03-09_06-55-50/eval.log` — massachusetts_buildings / Linear Probing: mIoU=0.7381, OA=0.9080
- `experiments/sam3_lora_massachusetts_buildings/sam3_lora_massachusetts_buildings_eval_2026-03-07_21-54-09/eval.log` — massachusetts_buildings / LoRA: mIoU=0.8278, OA=0.9415
- `experiments/sam3_adapter_massachusetts_roads/sam3_adapter_massachusetts_roads_eval_2026-04-08_19-54-55/eval.log` — massachusetts_roads / Adapter: mIoU=0.8077, OA=0.9790
- `experiments/sam3_linear_probing_massachusetts_roads/sam3_linear_probing_massachusetts_roads_eval_2026-03-08_21-48-20/eval.log` — massachusetts_roads / Linear Probing: mIoU=0.7048, OA=0.9638
- `experiments/sam3_lora_massachusetts_roads/sam3_lora_massachusetts_roads_eval_2026-03-07_22-03-41/eval.log` — massachusetts_roads / LoRA: mIoU=0.8128, OA=0.9798
- `experiments/sam3_adapter_potsdam/sam3_adapter_potsdam_eval_2026-03-16_23-38-43/eval.log` — potsdam / Adapter: mIoU=0.8735, OA=0.9293
- `experiments/sam3_linear_probing_potsdam/sam3_linear_probing_potsdam_eval_2026-02-25_22-40-38/eval.log` — potsdam / Linear Probing: mIoU=0.7186, OA=0.8170
- `experiments/sam3_lora_potsdam/sam3_lora_potsdam_eval_2026-02-25_15-46-31/eval.log` — potsdam / LoRA: mIoU=0.8806, OA=0.9306
- `experiments/sam3_adapter_uavid/eval_2026-03-24_10-55-55/eval.log` — uavid / Adapter: mIoU=0.6619, OA=0.8932
- `experiments/sam3_linear_probing_uavid/sam3_linear_probing_eval_2026-03-15_17-02-00/eval.log` — uavid / Linear Probing: mIoU=0.4554, OA=0.7440
- `experiments/sam3_lora_uavid/sam3_lora_uavid_eval_2026-03-14_23-00-44/eval.log` — uavid / LoRA: mIoU=0.6106, OA=0.8697
- `experiments/sam3_adapter_vaihingen/sam3_adapter_vaihingen_eval_2026-03-16_23-34-44/eval.log` — vaihingen / Adapter: mIoU=0.8152, OA=0.9048
- `experiments/sam3_linear_probing_vaihingen/sam3_linear_probing_vaihingen_eval_2026-02-26_13-57-59/eval.log` — vaihingen / Linear Probing: mIoU=0.6917, OA=0.8186
- `experiments/sam3_lora_vaihingen/sam3_lora_vaihingen_eval_2026-02-26_13-47-26/eval.log` — vaihingen / LoRA: mIoU=0.8326, OA=0.9101
- `experiments/sam3_adapter_whu_building/eval_2026-03-24_11-07-39/eval.log` — whu_building / Adapter: mIoU=0.9403, OA=0.9837
- `experiments/sam3_linear_probing_whu_building/eval_2026-03-12_19-16-13/eval.log` — whu_building / Linear Probing: mIoU=0.8877, OA=0.9682
- `experiments/sam3_lora_whu_building/sam_lora_whu_building_eval_2026-03-11_23-36-25/eval.log` — whu_building / LoRA: mIoU=0.9492, OA=0.9862
