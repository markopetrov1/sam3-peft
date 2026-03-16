# Evaluation results (mIoU & OA)

Parsed from all `eval.log` files under `experiments/`. Re-run the script to refresh.

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
| whu_building | 1008×1008 | 5,732 | 1,228 | 1,228 |

*(LoveDA: train 2,522, val 1,669 — no test evaluation in this report.)*

---

## Trainable parameters (by method)

| Method | Trainable params | Total params | Trainable % |
|--------|------------------|--------------|-------------|
| sam3_linear_probing | 2,299,652 | 456,339,460 | 0.50% |
| sam3_lora | 4,539,139 | 460,877,828 | 0.98% |

*(From Massachusetts Buildings runs; slight variation with num_classes.)*

---

## Summary table (all runs)

| Dataset | Method | mIoU | OA |
|---------|--------|------|-----|
| massachusetts_buildings | sam3_linear_probing | 0.7381 | 0.9080 |
| massachusetts_buildings | sam3_lora | 0.8278 | 0.9415 |
| massachusetts_roads | sam3_linear_probing | 0.7048 | 0.9638 |
| massachusetts_roads | sam3_lora | 0.8128 | 0.9798 |
| potsdam | sam3_linear_probing | 0.7186 | 0.8170 |
| potsdam | sam3_lora | 0.8806 | 0.9306 |
| uavid | sam3_linear_probing | 0.4533 | 0.7445 |
| uavid | sam3_lora | 0.6174 | 0.8663 |
| vaihingen | sam3_linear_probing | 0.6917 | 0.8186 |
| vaihingen | sam3_lora | 0.8326 | 0.9101 |
| whu_building | sam3_lora | 0.9492 | 0.9862 |

---

## Comparison by dataset (methods as columns)

| Dataset | sam3_linear_probing (mIoU) | sam3_linear_probing (OA) | sam3_lora (mIoU) | sam3_lora (OA) |
|---------|---------------------------|--------------------------|------------------|----------------|
| massachusetts_buildings | 0.7381 | 0.9080 | 0.8278 | 0.9415 |
| massachusetts_roads | 0.7048 | 0.9638 | 0.8128 | 0.9798 |
| potsdam | 0.7186 | 0.8170 | 0.8806 | 0.9306 |
| uavid | 0.4533 | 0.7445 | 0.6174 | 0.8663 |
| vaihingen | 0.6917 | 0.8186 | 0.8326 | 0.9101 |
| whu_building | — | — | 0.9492 | 0.9862 |

---

## Per-dataset tables

### massachusetts_buildings

| Method | mIoU | OA |
|--------|------|-----|
| sam3_linear_probing | 0.7381 | 0.9080 |
| sam3_lora | 0.8278 | 0.9415 |

### massachusetts_roads

| Method | mIoU | OA |
|--------|------|-----|
| sam3_linear_probing | 0.7048 | 0.9638 |
| sam3_lora | 0.8128 | 0.9798 |

### potsdam

| Method | mIoU | OA |
|--------|------|-----|
| sam3_linear_probing | 0.7186 | 0.8170 |
| sam3_lora | 0.8806 | 0.9306 |

### uavid

| Method | mIoU | OA |
|--------|------|-----|
| sam3_linear_probing | 0.4533 | 0.7445 |
| sam3_lora | 0.6174 | 0.8663 |

### vaihingen

| Method | mIoU | OA |
|--------|------|-----|
| sam3_linear_probing | 0.6917 | 0.8186 |
| sam3_lora | 0.8326 | 0.9101 |

### whu_building

| Method | mIoU | OA |
|--------|------|-----|
| sam3_lora | 0.9492 | 0.9862 |

---

## Parsed log files

- `experiments/sam3_linear_probing_massachusetts_buildings/sam3_linear_probing_massachusetts_buildings_eval_2026-03-09_06-55-50/eval.log` — massachusetts_buildings / sam3_linear_probing: mIoU=0.7381, OA=0.9080
- `experiments/sam3_lora_massachusetts_buildings/sam3_lora_massachusetts_buildings_eval_2026-03-07_21-54-09/eval.log` — massachusetts_buildings / sam3_lora: mIoU=0.8278, OA=0.9415
- `experiments/sam3_linear_probing_massachusetts_roads/sam3_linear_probing_massachusetts_roads_eval_2026-03-08_21-48-20/eval.log` — massachusetts_roads / sam3_linear_probing: mIoU=0.7048, OA=0.9638
- `experiments/sam3_lora_massachusetts_roads/sam3_lora_massachusetts_roads_eval_2026-03-07_22-03-41/eval.log` — massachusetts_roads / sam3_lora: mIoU=0.8128, OA=0.9798
- `experiments/sam3_linear_probing_potsdam/sam3_linear_probing_potsdam_eval_2026-02-25_22-40-38/eval.log` — potsdam / sam3_linear_probing: mIoU=0.7186, OA=0.8170
- `experiments/sam3_lora_potsdam/sam3_lora_potsdam_eval_2026-02-25_15-46-31/eval.log` — potsdam / sam3_lora: mIoU=0.8806, OA=0.9306
- `experiments/sam3_linear_probing_uavid/sam3_linear_probing_uavid_eval_2026-03-02_17-12-22/eval.log` — uavid / sam3_linear_probing: mIoU=0.4533, OA=0.7445
- `experiments/sam3_lora_uavid/sam3_lora_uavid_eval_2026-02-28_19-16-21/eval.log` — uavid / sam3_lora: mIoU=0.6174, OA=0.8663
- `experiments/sam3_linear_probing_vaihingen/sam3_linear_probing_vaihingen_eval_2026-02-26_13-57-59/eval.log` — vaihingen / sam3_linear_probing: mIoU=0.6917, OA=0.8186
- `experiments/sam3_lora_vaihingen/sam3_lora_vaihingen_eval_2026-02-26_13-47-26/eval.log` — vaihingen / sam3_lora: mIoU=0.8326, OA=0.9101
- `experiments/sam3_lora_whu_building/sam_lora_whu_building_eval_2026-03-11_23-36-25/eval.log` — whu_building / sam3_lora: mIoU=0.9492, OA=0.9862
