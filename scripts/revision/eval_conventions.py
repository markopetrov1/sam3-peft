#!/usr/bin/env python3
"""
Inference-only re-scoring of one trained SAM3-PEFT checkpoint under the conventions requested in
the first-round review. No parameter is updated and no optimiser is built.

For every evaluation image the script stores a confusion matrix at native ground-truth resolution
under each scoring convention below, so that every derived quantity (mean IoU, foreground IoU,
F1, cluster bootstrap, permutation tests) can be recomputed offline from one file.

Scoring conventions (keys of confusions.npz)
  full        argmax over all C+1 output channels, ground-truth ignore pixels discarded. This is
              the manuscript protocol: a valid pixel assigned to the ignore channel counts as an
              error for its true class.
  restrict    argmax over the evaluated class channels only. The ignore channel, and the unused
              ninth UAVid channel, can never be predicted. This is the label a model trained with
              a consistently masked objective would emit at inference.
  clutter8    UAVid only. Channel 0 is scored as the background-clutter class and no pixel is
              discarded, which is the official eight-class UAVid protocol.
  clutter6    ISPRS only. Clutter reference pixels are kept as a sixth class and channel 0 is read
              as a clutter prediction, so a clutter pixel labelled car is a false positive for car.
              Mean IoU and F1 are then taken over the five foreground classes. This is how the
              GeoSeg code behind UNetFormer scores Potsdam and Vaihingen.
  band{r}     `full` restricted to reference pixels within r pixels of a label change, and
  int{r}      to the remaining interior pixels; rband{r} and rint{r} are the same split scored with
              the restricted argmax.
  dedup       `full` with each pixel weighted by 1/(number of test tiles covering it), so that
              the overlapping last row and column of a tile grid are counted once (float matrix).
  c_*         ISPRS only, with --complete-dir: the same conventions against the complete
              reference instead of the eroded reference the stored tiles hold.

Input handling (--input-mode)
  stretch     the whole image is resized to 1008x1008 (manuscript protocol), batch of images.
  tiled       the image is cut into the 512x512 tile grid used to build the training split
              (stride 512, last row and column shifted inward), each tile is resized to 1008x1008,
              its 72x72 logits are resampled to the tile footprint, overlapping logits are
              averaged, and the argmax is taken once per native pixel.
  letterbox   the long side is resized to 1008 with the aspect ratio kept, the short side is
              zero-padded to 1008, and the padding is cropped from the logits before resampling.

Input-channel perturbations (--band-op), applied to the 8-bit raster before normalisation
  none, c0_from_c1, c1_from_c2, c2_from_c1, c0_mean (channel 0 set to its per-image mean),
  c0_shuffle (channel 0 taken from another test image, a fixed derangement of the image order).

Also recorded per image: the summed softmax probability of every channel over valid pixels, so the
mass the ignore channel and the unused UAVid channel take at inference can be reported.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from datasets import create_dataset  # noqa: E402
from peft import build_peft_model  # noqa: E402
from utils.config import load_config  # noqa: E402

TILE = 512
BAND_RADII = (1, 3, 7)
BAND_OPS = ("none", "c0_from_c1", "c1_from_c2", "c2_from_c1", "c0_mean", "c0_shuffle")
ISPRS_CLUTTER_RAW = 6


def tile_grid(h: int, w: int, tile: int = TILE):
    """Boxes (x0, y0, x1, y1) of the stride-`tile` grid with the last row/column shifted inward.

    Reproduces the grid of the stored UAVid training tiles (x0 in 0,512,...,3072 and W-512;
    y0 in 0,512,1024,1536,1648) and of the ISPRS and Massachusetts test tiles.
    """
    def starts(n):
        s = list(range(0, n - tile + 1, tile))
        if s[-1] + tile < n:
            s.append(n - tile)
        return s
    return [(x, y, x + tile, y + tile) for y in starts(h) for x in starts(w)]


def boundary_band(label: torch.Tensor, radius: int) -> torch.Tensor:
    """Pixels within `radius` of a change in label value, on the GPU.

    Same definition as reevaluate.py (3x3 dilation/erosion to find label changes, then a square
    dilation of side 2r+1). Max pooling pads with -inf, which matches OpenCV's default border
    for both dilation and erosion, so the two implementations agree pixel for pixel.
    """
    x = label.float()[None, None]
    mx = F.max_pool2d(x, 3, 1, 1)
    mn = -F.max_pool2d(-x, 3, 1, 1)
    edges = ((mx != x) | (mn != x)).float()
    if radius > 1:
        edges = F.max_pool2d(edges, 2 * radius + 1, 1, radius)
    return edges[0, 0] > 0


def confusion(gt: torch.Tensor, pred: torch.Tensor, n: int, drop: int | None,
              weight: torch.Tensor | None = None) -> np.ndarray:
    """n x n confusion of (gt, pred) on the GPU, discarding gt == drop (None keeps every pixel)."""
    g = gt.reshape(-1).long()
    p = pred.reshape(-1).long()
    w = None if weight is None else weight.reshape(-1).double()
    if drop is not None:
        keep = g != drop
        g, p = g[keep], p[keep]
        w = None if w is None else w[keep]
    if g.numel() == 0:
        return np.zeros((n, n), dtype=np.float64 if weight is not None else np.int64)
    if int(g.max()) >= n or int(p.max()) >= n:
        raise ValueError(f"index out of range for {n} classes")
    flat = torch.bincount(g * n + p, weights=w, minlength=n * n).reshape(n, n)
    return flat.cpu().numpy() if weight is not None else flat.cpu().numpy().astype(np.int64)


def confusion_b(gt: torch.Tensor, pred: torch.Tensor, n: int, drop: int | None,
                weight: torch.Tensor | None = None) -> np.ndarray:
    """[B, n, n] confusions of a batch of equally sized (gt, pred) maps in one bincount.

    Identical to calling confusion() per image; it exists because on a GPU shared by many processes
    every small kernel waits for a time slice, so a few large kernels are far faster than many small.
    """
    B = gt.shape[0]
    idx = (torch.arange(B, device=gt.device).view(B, 1, 1) * (n * n) + gt.long() * n + pred.long())
    w = None if weight is None else weight.double()
    if drop is not None:
        keep = gt != drop
        idx = idx[keep]
        w = None if w is None else w[keep]
    else:
        idx = idx.reshape(-1)
        w = None if w is None else w.reshape(-1)
    if idx.numel() and (int(gt.max()) >= n or int(pred.max()) >= n):
        raise ValueError(f"index out of range for {n} classes")
    flat = torch.bincount(idx, weights=w, minlength=B * n * n).reshape(B, n, n)
    return flat.cpu().numpy() if weight is not None else flat.cpu().numpy().astype(np.int64)


def boundary_band_b(label: torch.Tensor, radius: int) -> torch.Tensor:
    """boundary_band() for a [B, H, W] batch."""
    x = label.float()[:, None]
    mx = F.max_pool2d(x, 3, 1, 1)
    mn = -F.max_pool2d(-x, 3, 1, 1)
    edges = ((mx != x) | (mn != x)).float()
    if radius > 1:
        edges = F.max_pool2d(edges, 2 * radius + 1, 1, radius)
    return edges[:, 0] > 0


def apply_band_op(arr: np.ndarray, op: str, donor: np.ndarray | None) -> np.ndarray:
    a = arr.copy()
    if op == "none":
        return a
    if op == "c0_from_c1":
        a[:, :, 0] = a[:, :, 1]
    elif op == "c1_from_c2":
        a[:, :, 1] = a[:, :, 2]
    elif op == "c2_from_c1":
        a[:, :, 2] = a[:, :, 1]
    elif op == "c0_mean":
        a[:, :, 0] = np.round(a[:, :, 0].astype(np.float64).mean())
    elif op == "c0_shuffle":
        if donor is None or donor.shape != a.shape:
            raise ValueError("c0_shuffle needs a donor image of the same size")
        a[:, :, 0] = donor[:, :, 0]
    else:
        raise ValueError(op)
    return a


def read_label(path: str) -> np.ndarray:
    raw = np.array(PILImage.open(path))
    return raw[:, :, 0] if raw.ndim == 3 else raw


class Items(Dataset):
    """Raw 8-bit image (after the channel perturbation), raw and remapped native labels."""

    def __init__(self, pairs, remap, ignore, band_op, complete_dir):
        self.pairs, self.remap, self.ignore = pairs, remap, ignore
        self.band_op, self.complete_dir = band_op, complete_dir
        n = len(pairs)
        # Fixed derangement for c0_shuffle: image i takes channel 0 of image (i + n//2) mod n.
        self.donor = [(i + n // 2) % n for i in range(n)]

    def __len__(self):
        return len(self.pairs)

    def remapped(self, raw):
        gt = np.full(raw.shape, self.ignore, dtype=np.uint8)
        for old, new in self.remap.items():
            gt[raw == old] = new
        return gt

    def __getitem__(self, i):
        img_path, ann_path = self.pairs[i]
        img = np.array(PILImage.open(img_path).convert("RGB"))
        donor = None
        if self.band_op == "c0_shuffle":
            donor = np.array(PILImage.open(self.pairs[self.donor[i]][0]).convert("RGB"))
        img = apply_band_op(img, self.band_op, donor)
        raw = read_label(ann_path)
        out = {"img": img, "raw": raw, "gt": self.remapped(raw),
               "name": os.path.basename(img_path)}
        if self.complete_dir:
            craw = read_label(os.path.join(self.complete_dir, os.path.basename(ann_path)))
            out["craw"], out["cgt"] = craw, self.remapped(craw)
        return out


def collate(batch):
    return batch


def to_tensor(arr_u8: np.ndarray, size: int) -> torch.Tensor:
    """PIL bilinear resize to size x size, exactly as the training loader does."""
    pil = PILImage.fromarray(arr_u8).resize((size, size), PILImage.BILINEAR)
    return torch.from_numpy(np.array(pil, dtype=np.float32)).permute(2, 0, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--input-mode", default="stretch", choices=["stretch", "tiled", "letterbox"])
    ap.add_argument("--band-op", default="none", choices=BAND_OPS)
    ap.add_argument("--complete-dir", default=None,
                    help="ISPRS: tiles of the complete (non-eroded) reference, same file names")
    ap.add_argument("--conventions", default="all",
                    help="comma list from full,restrict,clutter8,clutter6,bands,dedup or 'all'")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.makedirs(args.out_dir, exist_ok=True)
    if os.path.isfile(os.path.join(args.out_dir, "summary.json")):
        print(f"skip: {args.out_dir} already complete")
        return 0

    cfg = load_config(args.config)
    ds_cfg = cfg.dataset
    size = int(ds_cfg.image_size)
    dataset = create_dataset(ds_cfg.type, root=args.root, split=args.split, image_size=size,
                             augment=False, exclude_classes=ds_cfg.exclude_classes or None)
    ignore = dataset.IGNORE_INDEX
    names_active = ["ignore"] + list(dataset.active_classes.values())
    n_ch = len(names_active)
    # Evaluated channels: every active class except the unused UAVid placeholder.
    eval_ch = [c for c in range(1, n_ch) if names_active[c] != "unused"]
    is_uavid = ds_cfg.type == "uavid"
    is_isprs = ds_cfg.type in ("potsdam", "vaihingen")
    has_boxes = ds_cfg.type in ("potsdam", "vaihingen", "massachusetts_buildings",
                                "massachusetts_roads")

    want = set(["full", "restrict", "clutter8", "clutter6", "bands", "dedup"]
               if args.conventions == "all" else args.conventions.split(","))
    if not is_uavid:
        want.discard("clutter8")
    if not is_isprs:
        want.discard("clutter6")
    if not has_boxes or args.input_mode != "stretch":
        want.discard("dedup")
    complete = bool(args.complete_dir) and is_isprs

    m = cfg.model
    method = getattr(m, "method", "sam3_lora")
    model = build_peft_model(
        sam_model=None, method=method, num_classes=len(dataset.active_classes), image_size=size,
        sam3_checkpoint=getattr(m, "sam3_checkpoint", None), bpe_path=getattr(m, "bpe_path", None),
        rank=getattr(m, "rank", 8), alpha=getattr(m, "alpha", 16),
        dropout=getattr(m, "dropout", 0.0), scale_factor=getattr(m, "scale_factor", 32),
        input_type=getattr(m, "input_type", "fft"), freq_nums=getattr(m, "freq_nums", 0.25),
        prompt_type=getattr(m, "prompt_type", "highpass"),
        tuning_stage=getattr(m, "tuning_stage", "1234"),
        handcrafted_tune=getattr(m, "handcrafted_tune", True),
        embedding_tune=getattr(m, "embedding_tune", True),
        adaptor=getattr(m, "adaptor", "adaptor"),
    ).cuda()
    model.load_parameters(args.checkpoint)
    model.eval()
    use_amp = bool(getattr(cfg.training, "amp", True))
    multimask = len(dataset.active_classes) > 2

    pairs = dataset.pairs[: args.limit] if args.limit else dataset.pairs
    per_load = args.batch_size if args.input_mode == "stretch" else 1
    loader = DataLoader(Items(pairs, dataset.class_id_remap, ignore, args.band_op,
                              args.complete_dir if complete else None),
                        batch_size=per_load, shuffle=False, num_workers=args.workers,
                        collate_fn=collate, prefetch_factor=4 if args.workers else None)

    # Tile coverage of each source image, for the de-duplicated convention.
    coverage = {}
    if "dedup" in want:
        boxes_by_src = {}
        for p, _ in pairs:
            stem = os.path.basename(p)[:-4].split("_")
            src, box = "_".join(stem[:-4]), tuple(map(int, stem[-4:]))
            boxes_by_src.setdefault(src, []).append(box)
        for src, boxes in boxes_by_src.items():
            W = max(b[2] for b in boxes)
            H = max(b[3] for b in boxes)
            cov = np.zeros((H, W), np.float32)
            for x0, y0, x1, y1 in boxes:
                cov[y0:y1, x0:x1] += 1
            coverage[src] = cov

    def forward(batch_imgs):
        x = torch.stack(batch_imgs).cuda(non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(x, multimask, size)
        return out["low_res_logits"].float()  # [B, C+1, 72, 72]

    def native_logits_batch(imgs):
        """List of [C+1, H, W] float logits at the native resolution of each image."""
        if args.input_mode == "stretch":
            low = forward([to_tensor(im, size) for im in imgs])
            return [F.interpolate(low[k : k + 1], size=im.shape[:2], mode="bilinear",
                                  align_corners=False)[0] for k, im in enumerate(imgs)]
        assert len(imgs) == 1
        img = imgs[0]
        h, w = img.shape[:2]
        if args.input_mode == "letterbox":
            s = size / max(h, w)
            nh, nw = int(round(h * s)), int(round(w * s))
            pil = PILImage.fromarray(img).resize((nw, nh), PILImage.BILINEAR)
            canvas = np.zeros((size, size, 3), np.float32)
            canvas[:nh, :nw] = np.array(pil, dtype=np.float32)
            low = forward([torch.from_numpy(canvas).permute(2, 0, 1)])
            up = F.interpolate(low, size=(size, size), mode="bilinear", align_corners=False)
            up = up[:, :, :nh, :nw]
            return [F.interpolate(up, size=(h, w), mode="bilinear", align_corners=False)[0]]
        boxes = tile_grid(h, w)
        acc = torch.zeros((n_ch, h, w), dtype=torch.float32, device="cuda")
        cnt = torch.zeros((1, h, w), dtype=torch.float32, device="cuda")
        for s0 in range(0, len(boxes), args.batch_size):
            chunk = boxes[s0 : s0 + args.batch_size]
            low = forward([to_tensor(np.ascontiguousarray(img[y0:y1, x0:x1]), size)
                           for (x0, y0, x1, y1) in chunk])
            up = F.interpolate(low, size=(TILE, TILE), mode="bilinear", align_corners=False)
            for k, (x0, y0, x1, y1) in enumerate(chunk):
                acc[:, y0:y1, x0:x1] += up[k]
                cnt[:, y0:y1, x0:x1] += 1
        return [acc / cnt]

    eval_idx = torch.tensor(eval_ch, device="cuda")
    store = {}
    prob_mass = []
    names = []
    t0 = time.time()

    def put(key, i, mat):
        if key not in store:
            store[key] = np.zeros((len(pairs),) + mat.shape, dtype=mat.dtype)
        store[key][i] = mat

    def score(prefix, i, gt_np, raw_np, pred_full, pred_r):
        gt_t = torch.from_numpy(gt_np).cuda()
        if "full" in want:
            put(prefix + "full", i, confusion(gt_t, pred_full, n_ch, ignore))
        if "restrict" in want:
            put(prefix + "restrict", i, confusion(gt_t, pred_r, n_ch, ignore))
        if "clutter8" in want:
            put(prefix + "clutter8", i, confusion(gt_t, pred_full, n_ch, None))
        if "clutter6" in want:
            # raw ISPRS labels: 1-5 foreground classes, 6 clutter, 0 eroded boundary (dropped).
            raw_t = torch.from_numpy(raw_np.astype(np.int64)).cuda()
            p6 = torch.where(pred_full == 0, torch.full_like(pred_full, ISPRS_CLUTTER_RAW),
                             pred_full)
            put(prefix + "clutter6", i, confusion(raw_t, p6, 7, 0))
        if "bands" in want:
            for r in BAND_RADII:
                band = boundary_band(gt_t, r)
                gb, gi = torch.where(band, gt_t, ignore), torch.where(~band, gt_t, ignore)
                put(f"{prefix}band{r}", i, confusion(gb, pred_full, n_ch, ignore))
                put(f"{prefix}int{r}", i, confusion(gi, pred_full, n_ch, ignore))
                put(f"{prefix}rband{r}", i, confusion(gb, pred_r, n_ch, ignore))
                put(f"{prefix}rint{r}", i, confusion(gi, pred_r, n_ch, ignore))
        return gt_t

    def put_b(key, i0, mats):
        if key not in store:
            store[key] = np.zeros((len(pairs),) + mats.shape[1:], dtype=mats.dtype)
        store[key][i0 : i0 + len(mats)] = mats

    def score_batched(prefix, i0, gt_np, raw_np, pf, pr):
        gt_t = torch.from_numpy(np.stack(gt_np)).cuda()
        if "full" in want:
            put_b(prefix + "full", i0, confusion_b(gt_t, pf, n_ch, ignore))
        if "restrict" in want:
            put_b(prefix + "restrict", i0, confusion_b(gt_t, pr, n_ch, ignore))
        if "clutter8" in want:
            put_b(prefix + "clutter8", i0, confusion_b(gt_t, pf, n_ch, None))
        if "clutter6" in want:
            raw_t = torch.from_numpy(np.stack(raw_np).astype(np.int64)).cuda()
            p6 = torch.where(pf == 0, torch.full_like(pf, ISPRS_CLUTTER_RAW), pf)
            put_b(prefix + "clutter6", i0, confusion_b(raw_t, p6, 7, 0))
        if "bands" in want:
            for r in BAND_RADII:
                band = boundary_band_b(gt_t, r)
                gb, gi = torch.where(band, gt_t, ignore), torch.where(~band, gt_t, ignore)
                put_b(f"{prefix}band{r}", i0, confusion_b(gb, pf, n_ch, ignore))
                put_b(f"{prefix}int{r}", i0, confusion_b(gi, pf, n_ch, ignore))
                # the same split with the argmax restricted to the evaluated classes
                put_b(f"{prefix}rband{r}", i0, confusion_b(gb, pr, n_ch, ignore))
                put_b(f"{prefix}rint{r}", i0, confusion_b(gi, pr, n_ch, ignore))
        return gt_t

    with torch.no_grad():
        idx = 0
        for batch in tqdm(loader, desc=os.path.basename(args.out_dir), mininterval=30):
            logits = native_logits_batch([it["img"] for it in batch])
            same = (args.input_mode == "stretch" and len(batch) > 1
                    and len({lg.shape for lg in logits}) == 1)
            if same:
                i0 = idx
                idx += len(batch)
                L = torch.stack(logits)
                pf = L.argmax(1)
                pr = eval_idx[L[:, eval_ch].argmax(1)]
                names.extend(it["name"] for it in batch)
                gt_t = score_batched("", i0, [it["gt"] for it in batch], [it["raw"] for it in batch], pf, pr)
                valid = (gt_t != ignore)[:, None]
                prob = torch.softmax(L, 1)
                prob_mass.extend((prob * valid).sum(dim=(2, 3)).double().cpu().numpy())
                del prob
                if "dedup" in want:
                    ws = []
                    for it in batch:
                        stem = it["name"][:-4].split("_")
                        src, (x0, y0, x1, y1) = "_".join(stem[:-4]), tuple(map(int, stem[-4:]))
                        ws.append(1.0 / coverage[src][y0:y1, x0:x1])
                    put_b("dedup", i0, confusion_b(gt_t, pf, n_ch, ignore,
                                                   weight=torch.from_numpy(np.stack(ws)).cuda()))
                if complete:
                    score_batched("c_", i0, [it["cgt"] for it in batch], [it["craw"] for it in batch], pf, pr)
                del logits, L
                continue
            for item, logit in zip(batch, logits):
                i = idx
                idx += 1
                names.append(item["name"])
                pred_full = logit.argmax(0)
                pred_r = eval_idx[logit[eval_ch].argmax(0)]
                gt_t = score("", i, item["gt"], item["raw"], pred_full, pred_r)
                valid_t = gt_t != ignore
                prob = torch.softmax(logit, 0)
                prob_mass.append((prob * valid_t).sum(dim=(1, 2)).double().cpu().numpy())
                del prob
                if "dedup" in want:
                    stem = item["name"][:-4].split("_")
                    src, (x0, y0, x1, y1) = "_".join(stem[:-4]), tuple(map(int, stem[-4:]))
                    wgt = torch.from_numpy(1.0 / coverage[src][y0:y1, x0:x1]).cuda()
                    put("dedup", i, confusion(gt_t, pred_full, n_ch, ignore, weight=wgt))
                if complete:
                    score("c_", i, item["cgt"], item["craw"], pred_full, pred_r)
            del logits

    elapsed = time.time() - t0
    np.savez_compressed(os.path.join(args.out_dir, "confusions.npz"),
                        names=np.array(names), class_names=np.array(names_active),
                        prob_mass=np.array(prob_mass), **store)
    summary = {
        "config": args.config, "checkpoint": args.checkpoint, "method": method,
        "dataset": ds_cfg.type, "split": args.split, "input_mode": args.input_mode,
        "band_op": args.band_op, "n_images": len(pairs), "class_names": names_active,
        "eval_channels": eval_ch, "conventions": sorted(store), "seconds": elapsed,
        "batch_size": args.batch_size, "torch": torch.__version__,
    }
    for key, arr in store.items():
        conf = arr.sum(0)
        k = conf.shape[0]
        tp = np.diag(conf).astype(np.float64)
        fp = conf.sum(0) - tp
        fn = conf.sum(1) - tp
        iou = np.divide(tp, tp + fp + fn, out=np.zeros(k), where=(tp + fp + fn) > 0)
        if key.endswith("clutter8"):
            cls = list(range(k))
        elif key.endswith("clutter6"):
            cls = [1, 2, 3, 4, 5]
        else:
            cls = eval_ch
        cls = [c for c in cls if conf[c].sum() > 0]
        summary[key] = {"miou": float(iou[cls].mean())}
    with open(os.path.join(args.out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({k: round(summary[k]["miou"], 4) for k in store}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
