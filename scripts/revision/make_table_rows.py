#!/usr/bin/env python3
"""Print LaTeX table rows for the revision tables straight from the evaluation outputs, so that no
number is typed by hand. Each block is printed under a header naming the table it belongs to."""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_cluster import bootstrap, cluster_index, miou  # noqa: E402

EV = "/home/users/marko.petrov/sam3-peft-revision/eval"
NICE = {"massachusetts_buildings": "Massachusetts Buildings", "massachusetts_roads": "Massachusetts Roads",
        "potsdam": "ISPRS Potsdam", "vaihingen": "ISPRS Vaihingen", "uavid": "UAVid",
        "whu_building": "WHU Building", "loveda": "LoveDA"}
M = ["linear_probing", "lora", "adapter"]


def load(tag, m, ds):
    p = f"{EV}/{tag}/{m}_{ds}/confusions.npz"
    return np.load(p) if os.path.isfile(p) else None


def cls_of(z):
    cn = [str(c) for c in z["class_names"]]
    return [i for i, c in enumerate(cn) if i > 0 and c != "unused"]


def f4(x):
    return f"{x:.4f}"


def sgn(x):
    return f"${x:+.4f}$"


def main_tag(ds):
    return "main_tiled" if ds == "uavid" else "main_stretch"


def ignore_rows():
    print("% --- tab:ignore: share of valid pixels assigned to the ignore channel, and restricted - primary")
    for ds in ["potsdam", "vaihingen", "uavid", "loveda"]:
        cells = []
        for m in M:
            z = load(main_tag(ds), m, ds)
            if z is None:
                cells.append("\\TBD{x} & \\TBD{x}")
                continue
            c = cls_of(z)
            full = z["full"].sum(0).astype(float)
            share = full[c, 0].sum() / full[c].sum()
            d = round(miou(z["restrict"].sum(0), c), 4) - round(miou(z["full"].sum(0), c), 4)
            cells.append(f"{100 * share:.2f} & {sgn(d)}")
        print(f"{NICE[ds]:24s} & " + " & ".join(cells) + " \\\\")


def boundary_rows():
    print("% --- tab:boundary: band3 / int3 for models (ISPRS: complete reference) and the oracle")
    orc = {k: json.load(open(f)) for k, f in ((os.path.basename(f)[:-5], f) for f in glob.glob(f"{EV}/oracle/*.json"))}
    for ds in ["massachusetts_buildings", "potsdam", "vaihingen", "uavid", "whu_building", "loveda"]:
        cells = []
        for m in M:
            z = load(main_tag(ds), m, ds)
            if z is None:
                cells.append("\\TBD{x} & \\TBD{x}")
                continue
            c = cls_of(z)
            if ds in ("potsdam", "vaihingen"):
                zr = z if "c_rband3" in z.files else load("isprs_rbands", m, ds)
                if zr is None:
                    cells.append("\\TBD{x} & \\TBD{x}")
                    continue
                b, it = zr["c_rband3"], zr["c_rint3"]
            else:
                b, it = z["band3"], z["int3"]
            cells.append(f"{f4(miou(b.sum(0), c))} & {f4(miou(it.sum(0), c))}")
        key = {"potsdam": "potsdam_complete", "vaihingen": "vaihingen_complete", "uavid": "uavid_tiled"}.get(ds, ds)
        o = orc.get(key)
        cells.append(f"{f4(o['band3'])} & {f4(o['int3'])}" if o else "\\TBD{o} & \\TBD{o}")
        print(f"{NICE[ds]:24s} & " + " & ".join(cells) + " \\\\")


def band_rows():
    print("% --- tab:spectral: drops (intact - perturbed) per operation")
    ops = {"vaihingen": ["c0_from_c1", "c1_from_c2", "c2_from_c1", "c0_mean", "c0_shuffle"],
           "potsdam": ["c0_from_c1", "c1_from_c2", "c2_from_c1", "c0_shuffle"]}
    for ds, oplist in ops.items():
        for m in M:
            base = load("main_stretch", m, ds)
            if base is None:
                print(f"% {ds} {m}: intact missing")
                continue
            c = cls_of(base)
            intact = round(miou(base["full"].sum(0), c), 4)
            cells = [f4(intact)]
            for op in oplist:
                z = load(f"band_{op}", m, ds)
                cells.append(sgn(round(miou(z["full"].sum(0), c), 4) - intact) if z is not None else "\\TBD{x}")
            print(f"{ds:10s} {m:15s} & " + " & ".join(cells) + " \\\\")


def band_ci():
    print("% --- cluster intervals of the channel-0 drops")
    for ds in ["vaihingen", "potsdam"]:
        for m in M:
            base = load("main_stretch", m, ds)
            z = load("band_c0_from_c1", m, ds)
            if base is None or z is None:
                continue
            names = [str(n) for n in base["names"]]
            cidx, g = cluster_index(ds, names)
            bs = bootstrap(base["full"], z["full"], cidx, cls_of(base), n_boot=10000, seed=5)
            print(f"{ds} {m}: drop CI [{bs['ci_diff'][0]:.4f}, {bs['ci_diff'][1]:.4f}] G={bs['G']}")


def matched_rows():
    print("% --- tab:matched")
    pairs = [("uavid", "matched_e25_tiled", 25)]
    common = {"loveda": 10, "potsdam": 35, "vaihingen": 30, "massachusetts_buildings": 40,
              "massachusetts_roads": 40, "whu_building": 45}
    for ds, e in common.items():
        pairs.append((ds, "matched_common", e))
    for ds, tag, e in pairs:
        zl, za = load(tag, "lora", ds), load(tag, "adapter", ds)
        bl, ba = load(main_tag(ds), "lora", ds), load(main_tag(ds), "adapter", ds)
        if zl is None or za is None or bl is None or ba is None:
            print(f"% {ds}: pending")
            continue
        c = cls_of(zl)
        ml, ma = round(miou(zl["full"].sum(0), c), 4), round(miou(za["full"].sum(0), c), 4)
        sl, sa = round(miou(bl["full"].sum(0), c), 4), round(miou(ba["full"].sum(0), c), 4)
        print(f"{NICE[ds]:24s} & {e} & {f4(ml)} & {f4(ma)} & {sgn(ml - ma)} & {sgn(sl - sa)} \\\\")


if __name__ == "__main__":
    for fn in sys.argv[1:] or ["ignore", "boundary", "band", "band_ci", "matched"]:
        globals()[f"{fn}_rows" if not fn.endswith("ci") else fn]()
