#!/usr/bin/env python3
"""Precision against recall per UAVid class for the three strategies, from the confusion matrices of
eval_conventions.py (tiled inference, the revised primary UAVid protocol). Same visual design as
the figure in the submitted manuscript: one colour per class, one marker per strategy, faint lines
joining a class across strategies, and dashed iso-lines of constant intersection over union."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = [("Linear Probing", "linear_probing", "o"), ("LoRA", "lora", "^"), ("Adapter", "adapter", "s")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="/home/users/marko.petrov/sam3-peft-revision/eval/main_tiled")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    plt.rcParams.update({"font.size": 6, "font.family": "DejaVu Sans", "axes.linewidth": 0.5})
    conf, names = {}, None
    for _, key, _ in METHODS:
        z = np.load(os.path.join(a.eval_dir, f"{key}_uavid", "confusions.npz"))
        conf[key] = z["full"].sum(0).astype(np.float64)
        names = [str(c) for c in z["class_names"]]
    classes = [c for c in range(1, len(names)) if names[c] != "unused" and conf["lora"][c].sum() > 0]
    cmap = plt.get_cmap("tab10")
    colour = {c: cmap(i % 10) for i, c in enumerate(classes)}

    def pr(cm, c):
        tp = cm[c, c]
        return tp / max(cm[c].sum(), 1), tp / max(cm[:, c].sum(), 1)

    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    rr, pp = np.meshgrid(np.linspace(1e-3, 1, 400), np.linspace(1e-3, 1, 400))
    iou = (pp * rr) / (pp + rr - pp * rr)
    cs = ax.contour(rr, pp, iou, levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    colors="0.82", linewidths=0.5, linestyles="--")
    ax.clabel(cs, fmt="IoU=%.1f", fontsize=4, inline=True)
    for c in classes:
        pts = [pr(conf[key], c) for _, key, _ in METHODS]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=colour[c], alpha=0.35, lw=0.8, zorder=2)
        for (_, key, mk), (r, p) in zip(METHODS, pts):
            ax.scatter(r, p, s=22, marker=mk, color=colour[c], edgecolor="black", linewidth=0.4, zorder=3)
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02); ax.set_aspect("equal")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.grid(alpha=0.2, lw=0.4)
    mh = [plt.Line2D([], [], marker=mk, ls="", markerfacecolor="0.85", markeredgecolor="black",
                     markersize=4, label=lab) for lab, _, mk in METHODS]
    ax.legend(handles=mh, title="Method", loc="upper left", fontsize=5, title_fontsize=5.5, framealpha=0.95)
    ch = [plt.Line2D([], [], marker="o", ls="", markerfacecolor=colour[c], markeredgecolor="black",
                     markersize=4, label=names[c]) for c in classes]
    fig.legend(handles=ch, loc="lower center", ncol=4, fontsize=5, frameon=False, bbox_to_anchor=(0.55, 0.0))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.98, bottom=0.27)
    fig.savefig(a.out)
    for c in classes:
        print(names[c], " ".join(f"{key}: R={pr(conf[key], c)[0]:.3f} P={pr(conf[key], c)[1]:.3f}" for _, key, _ in METHODS))


if __name__ == "__main__":
    main()
