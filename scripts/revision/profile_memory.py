#!/usr/bin/env python3
"""
Where does the adapter's extra training memory come from? A measurement, not a training run.

For each configuration the script builds the model exactly as train.py does, switches it to
training mode (so the activation checkpointing used in training is active), and runs ONE forward
and backward pass of the training loss on a synthetic batch under the same mixed precision. No
optimiser is created and no parameter is updated. The peak allocated memory above the resting
level is recorded for several batch sizes, and a straight line through those points is
extrapolated to the training batch of 24 and compared with the peaks logged during training.

Configurations
  lora                as trained
  adapter             as trained
  adapter_no_fft      adapter with the handcrafted FFT pyramid removed (handcrafted_tune=False)
  adapter_no_embed    adapter with the per-stage embedding projection removed (embedding_tune=False)
  adapter_in_ckpt     adapter as trained, but with the prompt computed inside each block's
                      activation checkpoint instead of before it (same arithmetic, different
                      placement relative to the checkpoint boundary)
  linear_probing      as trained
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt_utils

from peft import build_peft_model  # noqa: E402
from utils.losses import DiceLoss  # noqa: E402


def build(cfg_name: str, num_classes: int, size: int):
    method = {"lora": "sam3_lora", "linear_probing": "sam3_linear_probing"}.get(cfg_name, "sam3_adapter")
    kw = dict(scale_factor=32, input_type="fft", freq_nums=0.25, prompt_type="highpass",
              tuning_stage="1234", handcrafted_tune=cfg_name != "adapter_no_fft",
              embedding_tune=cfg_name != "adapter_no_embed", adaptor="adaptor")
    model = build_peft_model(sam_model=None, method=method, num_classes=num_classes, image_size=size,
                             sam3_checkpoint=None, bpe_path=None, rank=8, alpha=16, dropout=0.0, **kw)
    if cfg_name == "adapter_in_ckpt":
        move_prompt_inside_checkpoint(model)
    if cfg_name == "adapter_no_fft":
        # With handcrafted_tune=False the FFT pyramid is never built, but init_handcrafted is still
        # called by the injected forward; skip it so only the embedding prompt remains.
        model.prompt_generator.init_handcrafted = lambda x: (None, None, None, None)
    return model


def move_prompt_inside_checkpoint(model):
    """Re-patch the ViT so that prompt injection and block run inside one checkpoint."""
    from sam3.model.vitdet import get_abs_pos
    vit = model.sam3.backbone.vision_backbone.trunk
    pg = model.prompt_generator
    from peft.adapter import resize_handcrafted
    dps = len(vit.blocks) // 4

    def forward(x):
        inp = x
        x = vit.patch_embed(x)
        h, w = x.shape[1], x.shape[2]
        s = 0
        if vit.retain_cls_token:
            x = torch.cat([vit.class_embedding, x.flatten(1, 2)], dim=1)
            s = 1
        if vit.pos_embed is not None:
            x = x + get_abs_pos(vit.pos_embed, vit.pretrain_use_cls_token, (h, w),
                                vit.retain_cls_token, tiling=vit.tile_abs_pos)
        x = vit.ln_pre(x)
        hc_list = pg.init_handcrafted(inp)
        outputs = []
        for i, blk in enumerate(vit.blocks):
            stage, rel = i // dps + 1, i % dps
            hc = resize_handcrafted(hc_list[stage - 1], h, w)

            def step(x_, hc_, blk=blk, stage=stage, rel=rel):
                x_ = pg.get_prompt(x_, pg.init_prompt(x_, hc_, stage), stage, rel)
                return blk(x_)

            x = (ckpt_utils.checkpoint(step, x, hc, use_reentrant=False)
                 if (vit.use_act_checkpoint and vit.training) else step(x, hc))
            if (i == vit.full_attn_ids[-1]) or (vit.return_interm_layers and i in vit.full_attn_ids):
                if i == vit.full_attn_ids[-1]:
                    x = vit.ln_post(x)
                feats = x[:, s:]
                feats = feats.permute(0, 3, 1, 2) if feats.ndim == 4 else \
                    feats.reshape(feats.shape[0], h, w, feats.shape[-1]).permute(0, 3, 1, 2)
                outputs.append(feats)
        return outputs

    vit.forward = forward


def one_step(model, batch: int, size: int, n_out: int):
    x = torch.rand(batch, 3, size, size, device="cuda") * 255
    y = torch.randint(0, n_out, (batch, size, size), device="cuda")
    ce = torch.nn.CrossEntropyLoss(ignore_index=0)
    dice = DiceLoss(n_out)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with torch.amp.autocast("cuda", enabled=True):
        masks = model(x, n_out > 3, size)["masks"]
        loss = 0.5 * (ce(masks, y) + dice(F.softmax(masks, dim=1), y.unsqueeze(1)))
    loss.backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    for p in model.parameters():
        p.grad = None
    del x, y, masks, loss
    torch.cuda.empty_cache()
    return (peak - base) / 2**30, base / 2**30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="lora,adapter,adapter_no_fft,adapter_no_embed,adapter_in_ckpt,linear_probing")
    ap.add_argument("--batches", default="1,2,4")
    ap.add_argument("--num-classes", type=int, default=7, help="UAVid uses 8 (nine output channels)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", default="0")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    size = 1008
    res = json.load(open(a.out)) if os.path.isfile(a.out) else {}
    for name in a.configs.split(","):
        torch.manual_seed(0)
        model = build(name, a.num_classes, size).cuda()
        model.train()
        n_out = a.num_classes + 1
        res.setdefault(name, {})
        for b in map(int, a.batches.split(",")):
            peak, base = one_step(model, b, size, n_out)
            res[name][str(b)] = {"peak_above_resting_gib": peak, "resting_gib": base}
            print(f"{name:18s} batch {b:2d}: peak above resting {peak:6.2f} GiB (resting {base:5.2f} GiB)", flush=True)
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        res[name]["trainable_flagged"] = n_tr
        json.dump(res, open(a.out, "w"), indent=2)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
