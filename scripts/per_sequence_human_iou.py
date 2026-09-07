"""
Compute per-sequence human IoU on UAVid test with the trained Adapter model.

Hypothesis: high overall human IoU is pixel-weighted and dominated by crowd scenes
(seq27, seq28). Isolated-pedestrian sequences should show much lower IoU.
"""

import os, sys
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from peft import build_peft_model
from datasets import create_dataset
from utils.config import load_config

CFG = "configs/sam3_adapter_uavid.yaml"
CKPT = "experiments/sam3_adapter_uavid/sam3_adapter_uavid_2026-03-17_00-13-44/best.pth"

cfg = load_config(CFG)
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.experiment.gpu)

ds_cfg = cfg.dataset
ds = create_dataset(ds_cfg.type, root=ds_cfg.root, split="test",
                    image_size=ds_cfg.image_size, augment=False,
                    exclude_classes=ds_cfg.exclude_classes or None)
loader = DataLoader(ds, batch_size=cfg.training.batch_size, shuffle=False,
                    num_workers=cfg.training.num_workers, pin_memory=True)
ignore_index = ds.IGNORE_INDEX
num_classes = len(ds.active_classes)
class_names = ["ignore"] + list(ds.active_classes.values())
HUMAN = class_names.index("human")

m_cfg = cfg.model
model = build_peft_model(
    sam_model=None, method=m_cfg.method, num_classes=num_classes,
    image_size=ds_cfg.image_size,
    sam3_checkpoint=getattr(m_cfg, "sam3_checkpoint", None),
    bpe_path=getattr(m_cfg, "bpe_path", None),
    scale_factor=getattr(m_cfg, "scale_factor", 32),
    input_type=getattr(m_cfg, "input_type", "fft"),
    freq_nums=getattr(m_cfg, "freq_nums", 0.25),
    prompt_type=getattr(m_cfg, "prompt_type", "highpass"),
    tuning_stage=getattr(m_cfg, "tuning_stage", "1234"),
    handcrafted_tune=getattr(m_cfg, "handcrafted_tune", True),
    embedding_tune=getattr(m_cfg, "embedding_tune", True),
    adaptor=getattr(m_cfg, "adaptor", "adaptor"),
).cuda()
model.load_parameters(CKPT)
model.eval()
multimask_output = num_classes > 2
use_amp = getattr(cfg.training, "amp", True)

# Need the filename per index — pairs lives on the dataset
pairs = ds.pairs

per_seq = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

idx = 0
with torch.no_grad():
    for batch in tqdm(loader, desc="Evaluating"):
        images = batch["image"].cuda()
        labels = batch["label"].cuda()
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(images, multimask_output, ds_cfg.image_size)
        preds = outputs["masks"].argmax(dim=1)
        preds_np = preds.cpu().numpy()
        labels_np = labels.cpu().numpy()
        for b in range(preds_np.shape[0]):
            img_path, _ = pairs[idx]
            seq = os.path.basename(img_path).split("_")[0]
            p = preds_np[b]; g = labels_np[b]
            valid = g != ignore_index
            gp = g[valid]; pp = p[valid]
            tp = int(((gp == HUMAN) & (pp == HUMAN)).sum())
            fn = int(((gp == HUMAN) & (pp != HUMAN)).sum())
            fp = int(((gp != HUMAN) & (pp == HUMAN)).sum())
            per_seq[seq]["tp"] += tp
            per_seq[seq]["fp"] += fp
            per_seq[seq]["fn"] += fn
            idx += 1

print()
print(f"{'seq':<8}{'human GT px':>14}{'TP':>12}{'FN':>12}{'FP':>12}{'IoU':>8}{'recall':>8}")
print("-" * 80)
tot_tp = tot_fp = tot_fn = 0
for seq in sorted(per_seq.keys(), key=lambda s: int(s.replace('seq',''))):
    d = per_seq[seq]
    gt = d["tp"] + d["fn"]
    iou = d["tp"] / max(d["tp"] + d["fn"] + d["fp"], 1)
    rec = d["tp"] / max(gt, 1)
    tot_tp += d["tp"]; tot_fp += d["fp"]; tot_fn += d["fn"]
    print(f"{seq:<8}{gt:>14,}{d['tp']:>12,}{d['fn']:>12,}{d['fp']:>12,}{iou:>8.3f}{rec:>8.3f}")
print("-" * 80)
overall_iou = tot_tp / max(tot_tp + tot_fn + tot_fp, 1)
overall_rec = tot_tp / max(tot_tp + tot_fn, 1)
print(f"{'TOTAL':<8}{tot_tp+tot_fn:>14,}{tot_tp:>12,}{tot_fn:>12,}{tot_fp:>12,}{overall_iou:>8.3f}{overall_rec:>8.3f}")
