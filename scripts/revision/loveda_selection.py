#!/usr/bin/env python3
"""
How much does selecting and reporting on the same LoveDA validation split inflate the scores?

Inputs are per-image confusion matrices on the validation split for the checkpoints saved at fixed
epochs (every five epochs, independently of the validation curve) and for the selected checkpoint.

  fixed-epoch scores   mean IoU of each fixed-epoch checkpoint on the whole split; no checkpoint here
                       was chosen by the validation curve.
  split-half optimism  the images are halved at random; on half A the best fixed-epoch checkpoint is
                       chosen, and its score on A is compared with its score on the held-out half B.
                       Averaged over many halvings, A minus B estimates how much reporting a checkpoint
                       on the data that selected it overstates its score, for this candidate set.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_cluster import miou  # noqa: E402

EV = "/home/users/marko.petrov/sam3-peft-revision/eval"
CANDS = {"lora": [5, 10], "adapter": [5, 10, 15], "linear_probing": [5, 10, 40]}


def load(path):
    z = np.load(path)
    cn = [str(c) for c in z["class_names"]]
    return [str(n) for n in z["names"]], z["full"] if "full" in z.files else z["per_image"], \
        [i for i, c in enumerate(cn) if i > 0]


def main(n_splits=2000, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    best_paths = {m: f"{EV}/main_stretch/{m}_loveda/confusions.npz" for m in CANDS}
    for m, eps in CANDS.items():
        names, best, cls = load(best_paths[m])
        cand = {}
        for e in eps:
            path = f"{EV}/loveda_e{e}/{m}_loveda/confusions.npz"
            if not os.path.isfile(path):   # epoch 10 of the internal methods is the matched-budget run
                path = f"{EV}/matched_common/{m}_loveda/confusions.npz"
            n2, c, _ = load(path)
            assert n2 == names
            cand[e] = c
        rec = {"selected": float(miou(best.sum(0), cls)),
               "fixed": {e: float(miou(c.sum(0), cls)) for e, c in cand.items()}}
        n = len(names)
        opt, held, best_held = [], [], []
        for _ in range(n_splits):
            perm = rng.permutation(n)
            A, B = perm[: n // 2], perm[n // 2:]
            sA = {e: miou(c[A].sum(0), cls) for e, c in cand.items()}
            e_star = max(sA, key=sA.get)
            sB = miou(cand[e_star][B].sum(0), cls)
            opt.append(sA[e_star] - sB)
            held.append(sB)
            best_held.append(miou(best[B].sum(0), cls))
        rec.update({"optimism_mean": float(np.mean(opt)), "optimism_p95": float(np.percentile(opt, 95)),
                    "heldout_selected_mean": float(np.mean(held)),
                    "bestpth_on_same_heldout_mean": float(np.mean(best_held)), "n_splits": n_splits})
        out[m] = rec
        print(m, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()
                             if k != "fixed"}), {e: round(v, 4) for e, v in rec["fixed"].items()})
    json.dump(out, open("/home/users/marko.petrov/sam3-peft-revision/digest/loveda_selection.json", "w"), indent=1)


if __name__ == "__main__":
    main()
