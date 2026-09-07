#!/usr/bin/env python3
"""
Evaluate UAVid on the official labelled validation split, at native 4K resolution.

Why this script exists
----------------------
UAVid withholds its test-set annotations; the official test protocol runs through the benchmark's
own evaluation server. The labelled split that is unambiguously public is the validation split
(sequences 16-20, 36 and 37). This script scores the trained checkpoints on that split.

The local copy stores the validation split as 512x512 tiles cut from the native 3840x2160 frames
on a grid whose last row and column are clipped, so tiles overlap. Scoring the tiles independently
would therefore count roughly a quarter of the pixels twice. Instead this script performs
sliding-window inference: each tile is run through the network at the trained 1008x1008 working
resolution, its logits are resampled back to the tile's native 512x512 footprint and accumulated
into a full-frame logit canvas, overlapping contributions are averaged, and the argmax is taken
once per frame at the native 3840x2160 resolution. Metrics are then computed at native resolution,
which is the resolution the UAVid benchmark itself uses.

IMPORTANT. The training runs used this same validation split for checkpoint selection and early
stopping. The numbers this script produces are therefore optimistically biased as absolute
performance estimates. They remain useful for comparing the three adaptation strategies, which
were all selected under an identical procedure, but they must not be presented as held-out results.

Usage
-----
    python scripts/uavid_val_native.py --method sam3_lora --root /path/to/uavid \
        --out-dir analysis/uavid_val/lora
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import create_dataset          # noqa: E402
from peft import build_peft_model            # noqa: E402
from utils.config import load_config         # noqa: E402

FRAME_H, FRAME_W = 2160, 3840
# seq16_000000_<top>_<left>_<bottom>_<right>.png
TILE_RE = re.compile(r"^(?P<frame>.+?)_(?P<t>\d+)_(?P<l>\d+)_(?P<b>\d+)_(?P<r>\d+)\.png$")


def metrics_from_confusion(conf: np.ndarray, ignore: int) -> dict:
    n = conf.shape[0]
    tp = np.diag(conf).astype(np.float64)
    fp = conf.sum(axis=0) - tp
    fn = conf.sum(axis=1) - tp
    denom = tp + fp + fn
    iou = np.divide(tp, denom, out=np.zeros(n), where=denom > 0)
    f1d = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, f1d, out=np.zeros(n), where=f1d > 0)
    active = conf.sum(axis=1) > 0
    active[ignore] = False
    total = conf.sum()
    return {
        "miou": float(iou[active].mean()) if active.any() else 0.0,
        "oa": float(np.diag(conf).sum() / total) if total else 0.0,
        "mean_f1": float(f1[active].mean()) if active.any() else 0.0,
        "per_class_iou": {int(c): float(iou[c]) for c in range(n) if active[c]},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True,
                    choices=["sam3_linear_probing", "sam3_lora", "sam3_adapter"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit-frames", type=int, default=0)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = load_config(os.path.join(args.repo, "configs", f"{args.method}_uavid.yaml"))
    ds_cfg = cfg.dataset
    dataset = create_dataset("uavid", root=args.root, split="val",
                             image_size=ds_cfg.image_size, augment=False,
                             exclude_classes=ds_cfg.exclude_classes or None)
    ignore = dataset.IGNORE_INDEX
    n_cls = len(dataset.active_classes)
    n_ch = n_cls + 1
    class_names = ["ignore"] + list(dataset.active_classes.values())

    # Group the tiles by the native frame they were cut from.
    frames: dict[str, list] = defaultdict(list)
    for img_path, ann_path in dataset.pairs:
        m = TILE_RE.match(os.path.basename(img_path))
        if not m:
            raise RuntimeError(f"unexpected tile name: {img_path}")
        frames[m.group("frame")].append(
            (img_path, ann_path, int(m.group("t")), int(m.group("l")),
             int(m.group("b")), int(m.group("r"))))
    frame_ids = sorted(frames)
    if args.limit_frames:
        frame_ids = frame_ids[: args.limit_frames]
    print(f"UAVid val: {len(frame_ids)} native frames, "
          f"{sum(len(frames[f]) for f in frame_ids)} tiles")

    exp = f"{args.method}_uavid"
    ck = os.path.join(args.repo, "experiments", exp, "best.pth")
    if not os.path.isfile(ck):
        import glob
        hits = glob.glob(os.path.join(args.repo, "experiments", exp, "*", "best.pth"))
        ck = hits[0]
    mcfg = cfg.model
    model = build_peft_model(
        sam_model=None, method=args.method, num_classes=n_cls, image_size=ds_cfg.image_size,
        sam3_checkpoint=getattr(mcfg, "sam3_checkpoint", None),
        bpe_path=getattr(mcfg, "bpe_path", None),
        rank=getattr(mcfg, "rank", 8), alpha=getattr(mcfg, "alpha", 16),
        dropout=getattr(mcfg, "dropout", 0.0),
        scale_factor=getattr(mcfg, "scale_factor", 32),
        input_type=getattr(mcfg, "input_type", "fft"),
        freq_nums=getattr(mcfg, "freq_nums", 0.25),
        prompt_type=getattr(mcfg, "prompt_type", "highpass"),
        tuning_stage=getattr(mcfg, "tuning_stage", "1234"),
        handcrafted_tune=getattr(mcfg, "handcrafted_tune", True),
        embedding_tune=getattr(mcfg, "embedding_tune", True),
        adaptor=getattr(mcfg, "adaptor", "adaptor"),
    ).cuda()
    model.load_parameters(ck)
    model.eval()
    print(f"loaded {ck}")

    conf_total = np.zeros((n_ch, n_ch), dtype=np.int64)
    per_frame = np.zeros((len(frame_ids), n_ch, n_ch), dtype=np.int64)

    with torch.no_grad():
        for fi, fid in enumerate(tqdm(frame_ids, desc=f"{args.method} uavid val")):
            tiles = frames[fid]
            logit_sum = torch.zeros((n_ch, FRAME_H, FRAME_W), dtype=torch.float32, device="cuda")
            weight = torch.zeros((1, FRAME_H, FRAME_W), dtype=torch.float32, device="cuda")
            gt_frame = np.full((FRAME_H, FRAME_W), ignore, dtype=np.uint8)

            for s in range(0, len(tiles), args.batch_size):
                chunk = tiles[s : s + args.batch_size]
                xs = []
                for img_path, ann_path, t, l, b, r in chunk:
                    pil = PILImage.open(img_path).convert("RGB")
                    arr = np.array(pil.resize((ds_cfg.image_size, ds_cfg.image_size),
                                              PILImage.BILINEAR), dtype=np.float32)
                    xs.append(torch.from_numpy(arr).permute(2, 0, 1))
                    raw = np.array(PILImage.open(ann_path))
                    if raw.ndim == 3:
                        raw = raw[:, :, 0]
                    g = np.full_like(raw, ignore, dtype=np.uint8)
                    for old, new in dataset.class_id_remap.items():
                        g[raw == old] = new
                    gt_frame[t:b, l:r] = g

                low = model(torch.stack(xs).cuda(), n_cls > 2,
                            ds_cfg.image_size)["low_res_logits"].float()
                for k, (_, _, t, l, b, r) in enumerate(chunk):
                    up = F.interpolate(low[k : k + 1], size=(b - t, r - l),
                                       mode="bilinear", align_corners=False)[0]
                    logit_sum[:, t:b, l:r] += up
                    weight[:, t:b, l:r] += 1.0

            weight = weight.clamp(min=1.0)
            pred = (logit_sum / weight).argmax(dim=0).cpu().numpy().astype(np.uint8)
            valid = gt_frame != ignore
            np.add.at(per_frame[fi], (gt_frame[valid], pred[valid]), 1)
            conf_total += per_frame[fi]
            del logit_sum, weight
            torch.cuda.empty_cache()

    summary = {
        "method": args.method, "dataset": "uavid", "split": "val",
        "resolution": f"{FRAME_H}x{FRAME_W} (native, sliding-window tile inference)",
        "n_frames": len(frame_ids),
        "class_names": class_names,
        "caveat": "Checkpoints were selected and early-stopped on this split; absolute values "
                  "are optimistically biased and are not held-out estimates.",
        "native": metrics_from_confusion(conf_total, ignore),
    }
    np.savez_compressed(os.path.join(args.out_dir, "confusions.npz"),
                        per_image=per_frame, names=np.array(frame_ids),
                        conf_native=conf_total, class_names=np.array(class_names))
    with open(os.path.join(args.out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary["native"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
