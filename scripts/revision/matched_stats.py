#!/usr/bin/env python3
"""Cluster intervals and exact or Monte Carlo sign-flip tests for the matched-budget differences
(low-rank minus adapter at the latest epoch both runs saved)."""
import json, sys, numpy as np
sys.path.insert(0, "/home/users/marko.petrov/sam3-peft/scripts/revision")
from stats_cluster import bootstrap, sign_flip, cluster_index, miou
EV = "/home/users/marko.petrov/sam3-peft-revision/eval"
def L(t, m, d): return np.load(f"{EV}/{t}/{m}_{d}/confusions.npz")
common = {"uavid": ("matched_e25_tiled", 25), "loveda": ("matched_common", 10), "potsdam": ("matched_common", 35), "vaihingen": ("matched_common", 30),
          "massachusetts_buildings": ("matched_common", 40), "massachusetts_roads": ("matched_common", 40), "whu_building": ("matched_common", 45)}
out = {}
for ds, (t, e) in common.items():
    zl, za = L(t, "lora", ds), L(t, "adapter", ds)
    cn = [str(c) for c in zl["class_names"]]; cls = [i for i, c in enumerate(cn) if i > 0 and c != "unused"]
    names = [str(n) for n in zl["names"]]; assert names == [str(n) for n in za["names"]]
    cidx, g = cluster_index(ds, names)
    bs = bootstrap(zl["full"], za["full"], cidx, cls, n_boot=10000, seed=7)
    sf = sign_flip(zl["full"], za["full"], cidx, cls)
    pos = 0; G = int(cidx.max()) + 1
    for k in range(G):
        s = cidx == k
        pos += miou(zl["full"][s].sum(0), cls) > miou(za["full"][s].sum(0), cls)
    out[ds] = dict(epoch=e, diff=sf["diff"], ci=bs["ci_diff"], p=sf["p"], exact=sf["exact"], G=G, lora_wins=int(pos))
    print(ds, e, f"{sf['diff']:+.4f}", [round(x, 4) for x in bs["ci_diff"]], f"p={sf['p']:.4g}", "exact" if sf["exact"] else "mc", f"lora wins {pos}/{G}")
json.dump(out, open("/home/users/marko.petrov/sam3-peft-revision/digest/matched_stats.json", "w"), indent=1)
