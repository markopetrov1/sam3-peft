#!/usr/bin/env python3
"""
Inference throughput with repetitions, for linear probing, low-rank adaptation with its branches kept
separate (as trained), low-rank adaptation with the factors merged into the frozen weights, and the
adapter. Run on an otherwise idle GPU.

The test tiles are decoded and resized once, before timing, and kept on the GPU as 8-bit tensors,
so the measurement covers the network alone: each batch is converted to float outside the timed
region, and the forward call is bracketed by CUDA synchronisation. Every repetition runs the whole
image set; the mean and standard deviation over repetitions are reported, together with the median
per-batch latency. For the merged model, agreement with the unmerged one is checked on the same
images (largest absolute logit difference and the share of pixels whose argmax changes).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
from PIL import Image as PILImage

from peft import build_peft_model  # noqa: E402
from peft.lora import LoRALinear  # noqa: E402
from utils.config import load_config  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build(method, dataset):
    cfg = load_config(f"{REPO}/configs/sam3_{method}_{dataset}.yaml")
    m = cfg.model
    model = build_peft_model(
        sam_model=None, method=m.method, num_classes=cfg.dataset.num_classes, image_size=1008,
        sam3_checkpoint=None, bpe_path=None, rank=getattr(m, "rank", 8), alpha=getattr(m, "alpha", 16),
        dropout=getattr(m, "dropout", 0.0), scale_factor=getattr(m, "scale_factor", 32),
        input_type=getattr(m, "input_type", "fft"), freq_nums=getattr(m, "freq_nums", 0.25),
        prompt_type=getattr(m, "prompt_type", "highpass"), tuning_stage=getattr(m, "tuning_stage", "1234"),
        handcrafted_tune=getattr(m, "handcrafted_tune", True), embedding_tune=getattr(m, "embedding_tune", True),
        adaptor=getattr(m, "adaptor", "adaptor"))
    exp = f"{REPO}/experiments/sam3_{method}_{dataset}"
    ck = (glob.glob(f"{exp}/best.pth") or glob.glob(f"{exp}/*/best.pth"))[0]
    model.load_parameters(ck)
    return model.cuda().eval(), cfg


def merge_lora(model):
    """Fold every LoRA branch into its frozen linear layer and remove the branch."""
    n = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                base = child.original_layer
                with torch.no_grad():
                    delta = (child.lora.lora_A @ child.lora.lora_B).T * child.lora.scaling
                    base.weight += delta.to(base.weight.dtype)
                setattr(parent, name, base)
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="potsdam")
    ap.add_argument("--root", default="/home/users/marko.petrov/datasets/potsdam_mmseg")
    ap.add_argument("--n-images", type=int, default=400)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--methods", default="linear_probing,lora,lora_merged,adapter")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", default="0")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    files = sorted(glob.glob(f"{a.root}/img_dir/test/*.png"))[: a.n_images]
    imgs = torch.stack([torch.from_numpy(np.array(PILImage.open(f).convert("RGB").resize(
        (1008, 1008), PILImage.BILINEAR))).permute(2, 0, 1) for f in files]).cuda()
    meths = a.methods.split(",")
    models, mm = {}, None
    for meth in meths:
        base_method = "lora" if meth == "lora_merged" else meth
        model, cfg = build(base_method, a.dataset)
        merged = merge_lora(model) if meth == "lora_merged" else 0
        models[meth] = (model, merged)
        mm = cfg.dataset.num_classes > 2
    res = {"n_images": len(files), "batch": a.batch, "reps": a.reps, "device": torch.cuda.get_device_name(0),
           "torch": torch.__version__, "design": "all models resident; networks take turns on every batch (batch-level round robin)"}
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
        ref = models["lora"][0](imgs[: a.batch].float(), mm, 1008)["low_res_logits"].float() if "lora" in models else None
        for meth, (model, merged) in models.items():
            for _ in range(3):  # warm-up
                model(imgs[: a.batch].float(), mm, 1008)
            out = model(imgs[: a.batch].float(), mm, 1008)["low_res_logits"].float()
            res[meth] = {"merged_layers": merged, "img_per_s_reps": [], "batch_ms": []}
            if meth == "lora_merged" and ref is not None:
                res[meth]["max_abs_logit_diff_vs_unmerged"] = float((out - ref).abs().max())
                res[meth]["argmax_agreement_72x72"] = float((out.argmax(1) == ref.argmax(1)).float().mean())
        torch.cuda.synchronize()
        for r in range(a.reps):
            tot = {meth: 0.0 for meth in models}
            # Batch-level round robin: the networks process the same batch one after another, so that
            # load from other jobs on a shared device affects all of them alike and the per-batch time
            # ratios are paired.
            for s0 in range(0, len(files), a.batch):
                x = imgs[s0 : s0 + a.batch].float()
                for meth, (model, _) in models.items():
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    model(x, mm, 1008)
                    torch.cuda.synchronize()
                    dt = time.perf_counter() - t0
                    tot[meth] += dt
                    res[meth]["batch_ms"].append(1000 * dt)
            for meth in models:
                res[meth]["img_per_s_reps"].append(len(files) / tot[meth])
            print(f"rep {r + 1}: " + "  ".join(f"{m} {res[m]['img_per_s_reps'][-1]:.3f}" for m in models), flush=True)
    base_ms = np.array(res[meths[0]]["batch_ms"])
    for meth in meths:
        q = np.array(res[meth]["batch_ms"]) / base_ms   # paired per-batch time ratio to the first network
        res[meth]["batch_ms_raw"] = list(res[meth]["batch_ms"])
        res[meth]["paired_time_ratio_median"] = float(np.median(q))
        res[meth]["paired_time_ratio_iqr"] = np.percentile(q, [25, 75]).tolist()
        bs = np.random.default_rng(0).choice(q, size=(10000, len(q)))
        res[meth]["paired_time_ratio_median_ci"] = np.percentile(np.median(bs, 1), [2.5, 97.5]).tolist()
    base = res[meths[0]]["img_per_s_reps"]
    for meth in meths:
        thr = res[meth]["img_per_s_reps"]
        ratio = [t / b for t, b in zip(thr, base)]
        res[meth].update({"img_per_s_mean": statistics.mean(thr), "img_per_s_sd": statistics.stdev(thr),
                          "ratio_to_first_mean": statistics.mean(ratio), "ratio_to_first_sd": statistics.stdev(ratio),
                          "time_ratio_to_first_mean": statistics.mean(1 / q for q in ratio),
                          "time_ratio_to_first_sd": statistics.stdev(1 / q for q in ratio),
                          "batch_ms_median": statistics.median(res[meth]["batch_ms"])})
        del res[meth]["batch_ms"]
        print(meth, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[meth].items() if k != "img_per_s_reps"}, flush=True)
    for meth, (model, _) in models.items():
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
            model(imgs[: a.batch].float(), mm, 1008)
        torch.cuda.synchronize()
        res[meth]["peak_mib_batch_above_resident"] = (torch.cuda.max_memory_allocated() - torch.cuda.memory_allocated()) / 2**20
        res[meth]["weights_mib"] = sum(t.numel() * t.element_size() for t in
                                       list(model.parameters()) + list(model.buffers())) / 2**20
    json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
