"""
Precision-Recall scatter plots for UAVid.

Produces:
  - One combined PNG with all three methods overlaid (color = class,
    marker = method; same-class points connected by a faint line).
  - Per-method PNGs: one plot per method.

Light grey IoU iso-curves in the background show why two points with the
same recall can correspond to very different IoU.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

METHODS = [
    ("Linear Probing", "uavid_linear_probing", "o"),
    ("LoRA",           "uavid_lora",           "^"),
    ("Adapter",        "uavid_adapter",        "s"),
]
IGNORE_INDEX = 0
BASE = os.path.join(REPO_ROOT, "experiments", "confusion_matrices")


def load_matrix(name):
    cm = np.load(os.path.join(BASE, name, "confusion.npy"))
    classes = open(os.path.join(BASE, name, "classes.txt")).read().splitlines()
    return cm, classes


def pr_iou(cm, c):
    tp = cm[c, c]
    fn = cm[c, :].sum() - tp
    fp = cm[:, c].sum() - tp
    rec = tp / max(tp + fn, 1)
    pre = tp / max(tp + fp, 1)
    iou = tp / max(tp + fn + fp, 1)
    return rec, pre, iou


# Work out the classes once (all methods share the same class list)
_, class_names = load_matrix(METHODS[0][1])
active = []
for c, name in enumerate(class_names):
    if c == IGNORE_INDEX:
        continue
    total_gt = sum(load_matrix(m[1])[0][c, :].sum() for m in METHODS)
    if total_gt > 0:
        active.append((c, name))

cmap = plt.get_cmap("tab10")
class_colors = {c: cmap(i % 10) for i, (c, _) in enumerate(active)}

# Pre-compute IoU iso-curve grid (same for every plot)
rr, pp = np.meshgrid(np.linspace(1e-3, 1, 400), np.linspace(1e-3, 1, 400))
iou_grid = (pp * rr) / (pp + rr - pp * rr)
levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _draw_iso(ax):
    cs = ax.contour(rr, pp, iou_grid, levels=levels,
                    colors="lightgrey", linewidths=0.8, linestyles="--")
    ax.clabel(cs, fmt="IoU=%.1f", fontsize=8, inline=True)


def _format_pr_axes(ax, title):
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.set_aspect("equal")


# ---------------- Combined plot (all methods on one figure) ----------------
matrices = {label: load_matrix(folder)[0] for label, folder, _ in METHODS}
points = {(c, label): pr_iou(matrices[label], c)
          for c, _ in active for label, _, _ in METHODS}

fig, ax = plt.subplots(figsize=(9, 9))
_draw_iso(ax)

# Faint connecting lines per class (LP -> LoRA -> Adapter trajectory)
for c, _ in active:
    xs = [points[(c, label)][0] for label, _, _ in METHODS]
    ys = [points[(c, label)][1] for label, _, _ in METHODS]
    ax.plot(xs, ys, "-", color=class_colors[c],
            alpha=0.35, linewidth=1.0, zorder=2)

# Scatter points
for c, name in active:
    for label, _, marker in METHODS:
        rec, pre, _ = points[(c, label)]
        ax.scatter(rec, pre, s=130, marker=marker,
                   color=class_colors[c], edgecolor="black",
                   linewidth=0.8, zorder=3)

# Two legends: class colours and method markers
class_handles = [plt.Line2D([], [], marker="o", linestyle="",
                            markerfacecolor=class_colors[c],
                            markeredgecolor="black", markersize=10, label=name)
                 for c, name in active]
method_handles = [plt.Line2D([], [], marker=marker, linestyle="",
                             markerfacecolor="lightgrey",
                             markeredgecolor="black", markersize=10, label=label)
                  for label, _, marker in METHODS]
leg1 = ax.legend(handles=class_handles, title="Class", loc="lower left",
                 fontsize=9, title_fontsize=10, framealpha=0.95)
ax.add_artist(leg1)
ax.legend(handles=method_handles, title="Method", loc="upper left",
          fontsize=9, title_fontsize=10, framealpha=0.95)

_format_pr_axes(ax, "UAVid — precision vs recall per class, all methods\n"
                    "(dashed curves = constant-IoU iso-lines; "
                    "faint lines connect same class across methods)")

combined_out = os.path.join(BASE, "uavid_precision_recall.png")
fig.tight_layout()
fig.savefig(combined_out, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved {combined_out}")


# ---------------- Per-method plots ----------------
for label, folder, _ in METHODS:
    cm = matrices[label]
    fig, ax = plt.subplots(figsize=(8, 8))
    _draw_iso(ax)

    for c, name in active:
        rec, pre, iou = pr_iou(cm, c)
        ax.scatter(rec, pre, s=160, color=class_colors[c],
                   edgecolor="black", linewidth=0.8, zorder=3, label=name)
        ax.annotate(f"{name}\n(IoU={iou:.2f})",
                    (rec, pre), textcoords="offset points",
                    xytext=(8, 6), fontsize=8)

    ax.legend(title="Class", loc="lower left", fontsize=9,
              title_fontsize=10, framealpha=0.95)
    _format_pr_axes(ax, f"UAVid / {label} — precision vs recall per class\n"
                        "(dashed curves = constant-IoU iso-lines)")

    out = os.path.join(BASE, f"{folder}_precision_recall.png")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
