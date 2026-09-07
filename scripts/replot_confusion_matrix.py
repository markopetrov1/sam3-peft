"""
Re-plot a confusion matrix from saved confusion.npy + classes.txt.

Useful when you want to change the plot style without re-running eval.

Usage:
  python scripts/replot_confusion_matrix.py \
      --input_dir experiments/confusion_matrices/uavid_adapter \
      --title "UAVid / Adapter"
"""

import os
import sys
import argparse

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.compute_confusion_matrix import plot_confusion_and_iou


def per_class_iou_from_cm(cm):
    n = cm.shape[0]
    iou = np.zeros(n)
    for c in range(n):
        tp = cm[c, c]
        denom = cm[:, c].sum() + cm[c, :].sum() - tp
        iou[c] = tp / denom if denom > 0 else 0.0
    return iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True,
                    help="Folder containing confusion.npy and classes.txt")
    ap.add_argument("--title", default=None)
    ap.add_argument("--ignore_index", type=int, default=0)
    ap.add_argument("--output", default=None,
                    help="Output file (default: <input_dir>/confusion_matrix.png)")
    cli = ap.parse_args()

    cm = np.load(os.path.join(cli.input_dir, "confusion.npy"))
    with open(os.path.join(cli.input_dir, "classes.txt")) as f:
        class_names = [line.rstrip("\n") for line in f]

    per_class_iou = per_class_iou_from_cm(cm)

    keep = [c for c in range(cm.shape[0])
            if c != cli.ignore_index and cm[c, :].sum() > 0]
    cm_plot = cm[np.ix_(keep, keep)]
    names_plot = [class_names[c] for c in keep]
    iou_plot = per_class_iou[keep]
    miou_plot = iou_plot.mean() if len(iou_plot) else 0.0

    title = cli.title or os.path.basename(cli.input_dir.rstrip("/"))
    outfile = cli.output or os.path.join(cli.input_dir, "confusion_matrix.png")
    plot_confusion_and_iou(cm_plot, names_plot, iou_plot, miou_plot,
                           title=title, outfile=outfile)
    print(f"Saved {outfile}  (mIoU over {len(keep)} active classes = {miou_plot:.4f})")


if __name__ == "__main__":
    main()
