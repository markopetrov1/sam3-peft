"""
Re-render an existing eval visualization (image | gt | pred) into a publication-
quality figure with panel titles and a class-colour legend, for the README.

Reads a tri-panel PNG produced by test.py and writes a polished version with
the dataset's class palette baked into the legend.
"""

import os
import argparse
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# Potsdam (5 active + ignore). Order/colors match test.py CLASS_PALETTE.
POTSDAM_CLASSES = [
    ("impervious surface", (255, 255, 255)),
    ("building",           (  0,   0, 255)),
    ("low vegetation",     (  0, 255,   0)),
    ("tree",               (  0, 128,   0)),
    ("car",                (255, 255,   0)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="Path to the tri-panel PNG saved by test.py")
    ap.add_argument("--output", required=True,
                    help="Where to write the polished figure")
    ap.add_argument("--title", default="Potsdam — input | ground truth | prediction")
    cli = ap.parse_args()

    img = np.array(Image.open(cli.input).convert("RGB"))
    h, w, _ = img.shape
    assert w == 3 * h, f"Expected 3:1 aspect (image | gt | pred), got {h}x{w}"
    panel = w // 3
    parts = [img[:, i * panel:(i + 1) * panel] for i in range(3)]
    titles = ["Input image", "Ground truth", "Prediction"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    for ax, p, t in zip(axes, parts, titles):
        ax.imshow(p)
        ax.set_title(t, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    handles = [Patch(facecolor=np.array(c) / 255.0, edgecolor="black", label=name)
               for name, c in POTSDAM_CLASSES]
    fig.legend(handles=handles, loc="lower center", ncol=len(POTSDAM_CLASSES),
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(cli.title, fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(cli.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {cli.output}")


if __name__ == "__main__":
    main()
