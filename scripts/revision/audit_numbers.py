#!/usr/bin/env python3
"""
List every decimal number in the manuscript with its context and flag those that do not appear in
the set of reference values derived from the digest (every metric at 4, 3 and 1 decimals, every
pairwise difference of table values, means, and percentages). Flagged numbers are checked by hand;
the point is that nothing reaches the paper without being looked at against its source.
"""
import glob
import itertools
import json
import re
import sys

TEX = sorted(glob.glob("/home/users/marko.petrov/sam3-peft-paper/sections/*.tex")) + \
    ["/home/users/marko.petrov/sam3-peft-paper/main.tex"]
D = json.load(open("/home/users/marko.petrov/sam3-peft-revision/digest/digest.json"))


def walk(x, out):
    if isinstance(x, dict):
        for v in x.values():
            walk(v, out)
    elif isinstance(x, list):
        for v in x:
            walk(v, out)
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        out.append(float(x))


vals = []
walk(D, vals)
for extra in sys.argv[1:]:
    walk(json.load(open(extra)), vals)
ref = set()
for v in vals:
    for nd in (4, 3, 2, 1):
        ref.add(f"{round(v, nd):.{nd}f}")
        ref.add(f"{round(100 * v, nd):.{nd}f}")
        ref.add(f"{round(abs(v), nd):.{nd}f}")
# table values in the manuscript itself, and their pairwise differences
table_vals = set()
for f in TEX:
    for m in re.finditer(r"(?<![\d.])0\.\d{4}(?!\d)", open(f).read()):
        table_vals.add(float(m.group(0)))
tv = sorted(table_vals)
for a, b in itertools.combinations(tv, 2):
    ref.add(f"{abs(a - b):.4f}")
for f in TEX:
    lines = open(f).read().splitlines()
    for i, line in enumerate(lines, 1):
        if line.strip().startswith("%"):
            continue
        for m in re.finditer(r"(?<![\w.\\])[-+]?\d+\.\d+(?![\w.])", line):
            tok = m.group(0).lstrip("+-")
            if tok not in ref:
                ctx = line[max(0, m.start() - 50): m.end() + 30].strip()
                print(f"{f.split('/')[-1]}:{i}: {tok:>9s}   ...{ctx}...")
