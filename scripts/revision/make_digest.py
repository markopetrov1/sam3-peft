#!/usr/bin/env python3
"""
Collects every quantity the revised manuscript reports into one JSON digest (and a readable
Markdown copy). The manuscript's numbers are copied from this digest only, so that text, tables
and the response letter cannot drift apart.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_cluster import bootstrap, cluster_index, miou, sign_flip  # noqa: E402

EV = "/home/users/marko.petrov/sam3-peft-revision/eval"
OUT = "/home/users/marko.petrov/sam3-peft-revision/digest"
DS = ["massachusetts_buildings", "massachusetts_roads", "potsdam", "vaihingen", "uavid",
      "whu_building", "loveda"]
HELD_OUT = ["massachusetts_buildings", "massachusetts_roads", "potsdam", "vaihingen", "uavid"]
METHODS = ["linear_probing", "lora", "adapter"]
NB = int(os.environ.get("N_BOOT", "10000"))


def load(tag, method, ds):
    p = f"{EV}/{tag}/{method}_{ds}/confusions.npz"
    if not os.path.isfile(p):
        return None
    z = np.load(p, allow_pickle=False)
    return {k: z[k] for k in z.files}


def eval_classes(class_names):
    return [i for i, c in enumerate(class_names) if i > 0 and str(c) != "unused"]


def metrics(conf, classes, k_all=None):
    conf = conf.astype(np.float64)
    tp = np.diag(conf)
    gt = conf.sum(1)
    pr = conf.sum(0)
    den = gt + pr - tp
    iou = np.divide(tp, den, out=np.zeros_like(tp), where=den > 0)
    f1d = gt + pr
    f1 = np.divide(2 * tp, f1d, out=np.zeros_like(tp), where=f1d > 0)
    prec = np.divide(tp, pr, out=np.zeros_like(tp), where=pr > 0)
    rec = np.divide(tp, gt, out=np.zeros_like(tp), where=gt > 0)
    cls = [c for c in classes if gt[c] > 0]
    total = conf.sum()
    return {"miou": float(iou[cls].mean()), "oa": float(tp.sum() / total) if total else 0.0,
            "mf1": float(f1[cls].mean()),
            "iou": {int(c): float(iou[c]) for c in cls}, "f1": {int(c): float(f1[c]) for c in cls},
            "precision": {int(c): float(prec[c]) for c in cls},
            "recall": {int(c): float(rec[c]) for c in cls},
            "pixels": {int(c): int(gt[c]) for c in cls}}



def training_logs():
    """Epochs run, peak memory and wall-clock time of every training run, from its summary file."""
    import re as _re
    out = {}
    for f in glob.glob("/home/users/marko.petrov/sam3-peft/experiments/**/training_summary.txt", recursive=True):
        t = open(f).read()
        exp = _re.search(r"Experiment:\s+(\S+)", t).group(1)
        m, ds = None, None
        for meth in ("linear_probing", "lora", "adapter"):
            if exp.startswith(f"sam3_{meth}_"):
                m, ds = meth, exp[len(f"sam3_{meth}_"):]
        ep = int(_re.search(r"Epochs:\s+(\d+)/", t).group(1))
        sec = float(_re.search(r"Total time:.*\(([\d.]+) s\)", t).group(1))
        peak = float(_re.search(r"Peak GPU memory:\s+([\d.]+) MiB", t).group(1))
        tr = int(_re.search(r"Trainable params:\s+([\d,]+)", t).group(1).replace(",", ""))
        tot = int(_re.search(r"Total params:\s+([\d,]+)", t).group(1).replace(",", ""))
        out.setdefault(ds, {})[m] = {"epochs": ep, "seconds": sec, "s_per_epoch": sec / ep,
                                     "peak_mib": peak, "trainable_flagged": tr, "total_params": tot}
    return out



def val_curves():
    """Validation mean IoU after every epoch, parsed from each run's train.log."""
    import re as _re
    out = {}
    for f in glob.glob("/home/users/marko.petrov/sam3-peft/experiments/**/train.log", recursive=True):
        t = open(f, errors="ignore").read()
        mexp = _re.search(r"Experiment:\s+(\S+)", t) or _re.search(r"name:\s+(sam3_\S+)", t)
        exp = mexp.group(1)
        for meth in ("linear_probing", "lora", "adapter"):
            if exp.startswith(f"sam3_{meth}_"):
                m, ds = meth, exp[len(f"sam3_{meth}_"):]
        curve = {int(e): float(v) for e, v in _re.findall(r"\[Val epoch (\d+)\] mIoU: ([\d.]+)", t)}
        out.setdefault(ds, {})[m] = curve
    return out


def main_protocol_tag(ds):
    return "main_tiled" if ds == "uavid" else "main_stretch"


def main():
    os.makedirs(OUT, exist_ok=True)
    D = {"accuracy": {}, "stats": {}, "ignore_channel": {}, "conventions": {}, "boundary": {},
         "uavid_protocols": {}, "matched": {}, "band_ops": {}, "loveda_split_half": {},
         "oracle": {}, "notes": {}}
    # ---------------- accuracy under the main protocol, with every convention ----------------
    for ds in DS:
        D["accuracy"][ds] = {}
        for m in METHODS:
            z = load(main_protocol_tag(ds), m, ds)
            if z is None:
                continue
            cn = [str(c) for c in z["class_names"]]
            cls = eval_classes(cn)
            rec = {"class_names": cn, "n_images": int(len(z["names"])),
                   "full": metrics(z["full"].sum(0), cls)}
            if "restrict" in z:
                rec["restrict"] = metrics(z["restrict"].sum(0), cls)
            if "clutter8" in z:
                rec["clutter8"] = metrics(z["clutter8"].sum(0), list(range(len(cn))) [:8])
            if "clutter6" in z:
                rec["clutter6"] = metrics(z["clutter6"].sum(0), [1, 2, 3, 4, 5])
            if "c_full" in z:
                rec["complete_full"] = metrics(z["c_full"].sum(0), cls)
            if "dedup" in z:
                rec["dedup"] = metrics(z["dedup"].sum(0), cls)
            for r in (1, 3, 7):
                pre = "c_" if "c_full" in z else ""
                if f"{pre}band{r}" in z:
                    rec[f"band{r}"] = metrics(z[f"{pre}band{r}"].sum(0), cls)["miou"]
                    rec[f"int{r}"] = metrics(z[f"{pre}int{r}"].sum(0), cls)["miou"]
                if pre and f"band{r}" in z:
                    rec[f"eroded_band{r}"] = metrics(z[f"band{r}"].sum(0), cls)["miou"]
                    rec[f"eroded_int{r}"] = metrics(z[f"int{r}"].sum(0), cls)["miou"]
            # ignore channel: share of valid pixels predicted as channel 0, and probability mass
            full = z["full"].sum(0).astype(np.float64)
            valid = full[cls].sum()
            rec["share_pred_ignore"] = float(full[cls, 0].sum() / valid)
            pm = z["prob_mass"].sum(0)
            rec["prob_mass_share"] = (pm / pm.sum()).tolist()
            D["accuracy"][ds][m] = rec
    # ---------------- means over benchmark groups ----------------
    for group, members in [("held_out_five", HELD_OUT), ("all_seven", DS)]:
        g = {}
        for m in METHODS:
            vals = [D["accuracy"][ds][m]["full"] for ds in members if m in D["accuracy"].get(ds, {})]
            if len(vals) == len(members):
                g[m] = {"miou": float(np.mean([v["miou"] for v in vals])),
                        "oa": float(np.mean([v["oa"] for v in vals]))}
        D["accuracy"][f"mean_{group}"] = g
    # ---------------- paired cluster statistics ----------------
    for ds in DS:
        tag = main_protocol_tag(ds)
        za = {m: load(tag, m, ds) for m in METHODS}
        if any(v is None for v in za.values()):
            continue
        names = [str(n) for n in za["lora"]["names"]]
        assert all([str(n) for n in za[m]["names"]] == names for m in METHODS)
        cls = eval_classes([str(c) for c in za["lora"]["class_names"]])
        cidx, groups = cluster_index(ds, names)
        res = {"G": len(groups), "n_images": len(names), "groups": groups[:50]}
        for a, b in [("lora", "adapter"), ("lora", "linear_probing"), ("adapter", "linear_probing")]:
            bs = bootstrap(za[a]["full"], za[b]["full"], cidx, cls, n_boot=NB, seed=1)
            pt = sign_flip(za[a]["full"], za[b]["full"], cidx, cls)
            img = bootstrap(za[a]["full"], za[b]["full"], np.arange(len(names)), cls, n_boot=NB, seed=2)
            res[f"{a}-{b}"] = {"diff": pt["diff"], "ci_cluster": bs["ci_diff"], "ci_image": img["ci_diff"],
                               "p_cluster": pt["p"], "p_floor": pt["p_floor"], "exact": pt["exact"],
                               "ci_a": bs["ci_a"], "ci_b": bs["ci_b"]}
        D["stats"][ds] = res
    # ---------------- UAVid input handling ----------------
    for tag in ["main_stretch", "main_tiled", "uavid_letterbox"]:
        D["uavid_protocols"][tag] = {}
        for m in METHODS:
            z = load(tag, m, "uavid")
            if z is None:
                continue
            cn = [str(c) for c in z["class_names"]]
            cls = eval_classes(cn)
            D["uavid_protocols"][tag][m] = {"full": metrics(z["full"].sum(0), cls),
                                            "restrict": metrics(z["restrict"].sum(0), cls),
                                            "clutter8": metrics(z["clutter8"].sum(0), list(range(8)))}
    # ---------------- matched training budget ----------------
    for path in sorted(glob.glob(f"{EV}/matched_*/*")) + sorted(glob.glob(f"{EV}/traj_*/*")):
        tag, run = path.split("/")[-2:]
        p = f"{path}/confusions.npz"
        if not os.path.isfile(p):
            continue
        zz = np.load(p)
        cn = [str(c) for c in zz["class_names"]]
        cls = eval_classes(cn)
        D["matched"].setdefault(tag, {})[run] = {"full": metrics(zz["full"].sum(0), cls)["miou"]}
        if "clutter8" in zz.files:
            D["matched"][tag][run]["clutter8"] = metrics(zz["clutter8"].sum(0), list(range(8)))["miou"]
    # ---------------- channel replacement ----------------
    for path in sorted(glob.glob(f"{EV}/band_*/*")):
        tag, run = path.split("/")[-2:]
        p = f"{path}/confusions.npz"
        if not os.path.isfile(p):
            continue
        m, ds = run.split("_", 1) if not run.startswith("linear_probing") else ("linear_probing", run[len("linear_probing_"):])
        zz = np.load(p)
        base = load("main_stretch", m, ds)
        cn = [str(c) for c in zz["class_names"]]
        cls = eval_classes(cn)
        names = [str(n) for n in zz["names"]]
        assert names == [str(n) for n in base["names"]]
        cidx, groups = cluster_index(ds, names)
        bs = bootstrap(base["full"], zz["full"], cidx, cls, n_boot=NB, seed=3)
        D["band_ops"].setdefault(ds, {}).setdefault(m, {})[tag[len("band_"):]] = {
            "intact": metrics(base["full"].sum(0), cls)["miou"],
            "perturbed": metrics(zz["full"].sum(0), cls)["miou"],
            "drop": metrics(base["full"].sum(0), cls)["miou"] - metrics(zz["full"].sum(0), cls)["miou"],
            "ci_drop_cluster": bs["ci_diff"], "G": bs["G"]}
    # ---------------- training logs and the memory measurement ----------------
    D["training"] = training_logs()
    D["val_curves"] = val_curves()
    mp = f"{EV}/memory_profile.json"
    if os.path.isfile(mp):
        prof = json.load(open(mp))
        fit = {}
        for k, v in prof.items():
            b = np.array([1, 2, 4]); y = np.array([v[str(x)]["peak_above_resting_gib"] for x in b])
            slope, icpt = np.polyfit(b, y, 1)
            fit[k] = {"per_sample_gib": float(slope), "intercept_gib": float(icpt), "batch4_gib": float(y[2]),
                      "resting_gib": float(v["4"]["resting_gib"]),
                      "extrapolated_peak_24_gib": float(v["4"]["resting_gib"] + icpt + 24 * slope)}
        D["memory_profile"] = fit
    # ---------------- stride-14 oracle ----------------
    for p in sorted(glob.glob(f"{EV}/oracle/*.json")):
        D["oracle"][os.path.basename(p)[:-5]] = json.load(open(p))
    # ---------------- outputs of the separate analyses, carried over so that one file holds every number --------
    for key, path in [("loveda_selection", f"{OUT}/loveda_selection.json"),
                      ("matched_stats", f"{OUT}/matched_stats.json"),
                      ("inference_bench", f"{EV}/bench_interleaved.json")]:
        if os.path.isfile(path):
            D[key] = json.load(open(path))
    json.dump(D, open(f"{OUT}/digest.json", "w"), indent=1)
    print("digest written:", f"{OUT}/digest.json")


if __name__ == "__main__":
    main()
