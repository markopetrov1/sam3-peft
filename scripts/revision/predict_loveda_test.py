#!/usr/bin/env python3
"""Write LoveDA test-set predictions in the format of the official evaluation server
(codalab.lisn.upsaclay.fr/competitions/421): one 1024x1024 PNG per test image, named as the image,
with class indices 0 (background) to 6 (agriculture). The server needs an account, so this script
only prepares the zip; submitting it is left to the authors.

The argmax is restricted to the seven evaluated channels, since the server has no ignore class, and
the label index is the model's channel minus one (channel 0 is the no-data ignore channel).
"""
import argparse
import glob
import io
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage

from peft import build_peft_model  # noqa: E402
from utils.config import load_config  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["linear_probing", "lora", "adapter"])
    ap.add_argument("--images", default="/home/users/marko.petrov/datasets/loveDA_mmseg/img_dir/test")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gpu", default="0")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    cfg = load_config(f"{REPO}/configs/sam3_{a.method}_loveda.yaml")
    m = cfg.model
    model = build_peft_model(
        sam_model=None, method=m.method, num_classes=7, image_size=1008, sam3_checkpoint=None, bpe_path=None,
        rank=getattr(m, "rank", 8), alpha=getattr(m, "alpha", 16), dropout=getattr(m, "dropout", 0.0),
        scale_factor=getattr(m, "scale_factor", 32), input_type=getattr(m, "input_type", "fft"),
        freq_nums=getattr(m, "freq_nums", 0.25), prompt_type=getattr(m, "prompt_type", "highpass"),
        tuning_stage=getattr(m, "tuning_stage", "1234"), handcrafted_tune=getattr(m, "handcrafted_tune", True),
        embedding_tune=getattr(m, "embedding_tune", True), adaptor=getattr(m, "adaptor", "adaptor")).cuda()
    exp = f"{REPO}/experiments/sam3_{a.method}_loveda"
    model.load_parameters((glob.glob(f"{exp}/best.pth") or glob.glob(f"{exp}/*/best.pth"))[0])
    model.eval()
    files = sorted(glob.glob(f"{a.images}/*.png"), key=lambda p: int(os.path.basename(p)[:-4]))
    os.makedirs(a.out_dir, exist_ok=True)
    out_zip = os.path.join(a.out_dir, f"loveda_test_{a.method}.zip")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf, torch.no_grad():
        for s in range(0, len(files), 8):
            chunk = files[s : s + 8]
            pils = [PILImage.open(f).convert("RGB") for f in chunk]
            x = torch.stack([torch.from_numpy(np.array(p.resize((1008, 1008), PILImage.BILINEAR),
                                                       dtype=np.float32)).permute(2, 0, 1) for p in pils]).cuda()
            with torch.amp.autocast("cuda", enabled=True):
                low = model(x, True, 1008)["low_res_logits"].float()
            for f, p, lg in zip(chunk, pils, low):
                up = F.interpolate(lg[None], size=(p.height, p.width), mode="bilinear", align_corners=False)[0]
                lab = (up[1:8].argmax(0)).to(torch.uint8).cpu().numpy()   # channels 1..7 -> labels 0..6
                buf = io.BytesIO()
                PILImage.fromarray(lab).save(buf, format="PNG")
                zf.writestr(os.path.basename(f), buf.getvalue())
    print("wrote", out_zip, len(files), "masks")


if __name__ == "__main__":
    main()
