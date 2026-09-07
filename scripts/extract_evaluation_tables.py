#!/usr/bin/env python3
"""
Extract all evaluation results (mIoU, OA, per-class IoU) from eval.log files
(and train.log for loveda, which has no eval run) under experiments/ and write
a structured Markdown report.

Run from repo root:
    python scripts/extract_evaluation_tables.py

Output: evaluation_results.md (repo root).
Parses from scratch every run; no caching.
"""

import ast
import os
import re
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
OUTPUT_MD = REPO_ROOT / "evaluation_results.md"

# ---- test.py output patterns ----
PATTERN_RESULTS = re.compile(r"Results — (.+?) \((\w+)\), method=(.+)")
PATTERN_MIOU = re.compile(r"mIoU\s+([\d.]+)")
PATTERN_OA = re.compile(r"Overall Accuracy\s+([\d.]+)")
# Per-class row: "<name>   <iou>   <pixels>" (name may contain spaces)
PATTERN_CLASS_ROW = re.compile(
    r"^(?P<name>[A-Za-z][A-Za-z ]*?)\s{2,}(?P<iou>\d\.\d+)\s+(?P<pixels>[\d,]+)"
)

# ---- train.py output patterns (used for loveda which has no eval run) ----
PATTERN_VAL_MIOU = re.compile(r"\[Val epoch \d+\] mIoU:\s*([\d.]+),\s*OA:\s*([\d.]+)")
PATTERN_PER_CLASS = re.compile(r"Per-class IoU:\s*(\{.*\})")
PATTERN_NEW_BEST = re.compile(r"New best mIoU=([\d.]+),\s*OA=([\d.]+)")

# Methods that showed up as `method=...` in logs. Map to display names.
METHOD_DISPLAY = {
    "sam3_linear_probing": "Linear Probing",
    "sam3_lora": "LoRA",
    "sam3_adapter": "Adapter",
}
METHOD_ORDER = ["Linear Probing", "LoRA", "Adapter"]

# Static sections (not derivable from logs). Keep these hand-authored here so
# re-running the script preserves them in the output.
STATIC_DATASET_TABLE = """## Dataset setup (resolution & split sizes)

All images are resized to **1008×1008** (config `dataset.image_size`) for training and evaluation.

| Dataset | Resolution (crop) | Train | Val | Test |
|---------|-------------------|-------|-----|------|
| massachusetts_buildings | 1008×1008 | 1,233 | 36 | 90 |
| massachusetts_roads | 1008×1008 | 9,972 | 126 | 441 |
| potsdam | 1008×1008 | 2,765 | 691 | 2,016 |
| uavid | 1008×1008 | 8,000 | 2,800 | 150 |
| vaihingen | 1008×1008 | 276 | 68 | 398 |
| loveda | 1008×1008 | 2,522 | 1,669 | — |
| whu_building | 1008×1008 | 5,732 | 1,228 | 1,228 |
"""

STATIC_PARAMS_TABLE = """## Trainable parameters (by method)

| Method | Trainable params | Total params | Trainable % |
|--------|------------------|--------------|-------------|
| Linear Probing | 2,299,652 | 456,339,460 | 0.50% |
| LoRA | 4,539,139 | 460,877,828 | 0.98% |
| Adapter | 337,891 | 456,676,580 | 0.07% |

*(From Massachusetts Buildings runs; slight variation with num_classes.)*
"""


def _display_method(raw: str) -> str:
    return METHOD_DISPLAY.get(raw.strip(), raw.strip())


def find_all_eval_logs(experiments_dir: Path) -> list[Path]:
    if not experiments_dir.is_dir():
        return []
    return sorted(experiments_dir.rglob("eval.log"))


def parse_eval_log(path: Path) -> dict | None:
    """Parse one eval.log. Returns result dict or None on failure."""
    text = path.read_text(encoding="utf-8", errors="replace")
    dataset = split = method = None
    miou = oa = None
    per_class: list[tuple[str, float]] = []

    in_results = False
    for line in text.splitlines():
        m = PATTERN_RESULTS.search(line)
        if m:
            dataset, split, method = m.group(1), m.group(2), m.group(3).strip()
            in_results = True
            continue
        if not in_results:
            continue

        m = PATTERN_MIOU.search(line)
        if m:
            miou = float(m.group(1))
            continue
        m = PATTERN_OA.search(line)
        if m:
            oa = float(m.group(1))
            continue
        m = PATTERN_CLASS_ROW.match(line)
        if m:
            name = m.group("name").strip()
            iou = float(m.group("iou"))
            pixels_raw = m.group("pixels").replace(",", "")
            pixels = int(pixels_raw)
            if name.lower() == "ignore" or pixels == 0:
                continue
            per_class.append((name, iou))

    if dataset is None or method is None or miou is None or oa is None:
        return None
    return {
        "dataset": dataset,
        "split": split,
        "method": _display_method(method),
        "miou": miou,
        "oa": oa,
        "per_class": per_class,
        "log_path": str(path),
    }


def parse_loveda_train_log(path: Path, method_raw: str) -> dict | None:
    """Parse loveda train.log: pick the best val epoch's per-class IoU."""
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    best_miou = best_oa = None
    best_per_class: list[tuple[str, float]] = []
    # Track the most recent (miou, oa, per_class) triple, then commit when a
    # "New best" line matches.
    recent_val: tuple[float, float] | None = None
    recent_per_class: list[tuple[str, float]] = []

    for line in lines:
        m = PATTERN_VAL_MIOU.search(line)
        if m:
            recent_val = (float(m.group(1)), float(m.group(2)))
            recent_per_class = []
            continue
        m = PATTERN_PER_CLASS.search(line)
        if m:
            try:
                d = ast.literal_eval(m.group(1))
                recent_per_class = [(k, float(v)) for k, v in d.items()]
            except (ValueError, SyntaxError):
                pass
            continue
        m = PATTERN_NEW_BEST.search(line)
        if m and recent_val is not None:
            best_miou = float(m.group(1))
            best_oa = float(m.group(2))
            best_per_class = list(recent_per_class)

    if best_miou is None or not best_per_class:
        return None
    return {
        "dataset": "loveda",
        "split": "val",
        "method": _display_method(method_raw),
        "miou": best_miou,
        "oa": best_oa,
        "per_class": best_per_class,
        "log_path": str(path),
    }


def collect_all_results(experiments_dir: Path) -> list[dict]:
    results = [r for r in (parse_eval_log(p) for p in find_all_eval_logs(experiments_dir)) if r]

    # loveda has no eval.log — pick up best val from train.log for each method.
    loveda_sources = {
        "sam3_linear_probing": experiments_dir / "sam3_linear_probing_loveda" / "train.log",
        "sam3_lora": experiments_dir / "sam3_lora_loveda" / "train.log",
        "sam3_adapter": experiments_dir / "sam3_adapter_loveda" / "train.log",
    }
    for method_raw, path in loveda_sources.items():
        row = parse_loveda_train_log(path, method_raw)
        if row:
            results.append(row)

    return results


def _ordered_methods(results: list[dict]) -> list[str]:
    seen = {r["method"] for r in results}
    ordered = [m for m in METHOD_ORDER if m in seen]
    extras = sorted(seen - set(ordered))
    return ordered + extras


def _ordered_classes(rows: list[dict]) -> list[str]:
    """Union of class names across methods, in first-seen order."""
    seen: list[str] = []
    for r in rows:
        for name, _ in r["per_class"]:
            if name not in seen:
                seen.append(name)
    return seen


def build_markdown(results: list[dict], repo_root: Path) -> str:
    if not results:
        return "# Evaluation results\n\nNo evaluation logs found under `experiments/`.\n"

    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_dataset[r["dataset"]].append(r)

    datasets_order = sorted(by_dataset.keys())
    methods_order = _ordered_methods(results)

    lines: list[str] = [
        "# Evaluation results (mIoU, OA & per-class IoU)",
        "",
        "Parsed from `eval.log` files (and loveda `train.log`, which has no eval run) under `experiments/`. Re-run `scripts/extract_evaluation_tables.py` to refresh.",
        "",
        "---",
        "",
        STATIC_DATASET_TABLE,
        "---",
        "",
        STATIC_PARAMS_TABLE,
        "---",
        "",
        "## Summary table (all runs)",
        "",
        "| Dataset | Method | mIoU | OA |",
        "|---------|--------|------|-----|",
    ]

    for ds in datasets_order:
        by_method = {r["method"]: r for r in by_dataset[ds]}
        for method in methods_order:
            if method in by_method:
                r = by_method[method]
                lines.append(f"| {ds} | {method} | {r['miou']:.4f} | {r['oa']:.4f} |")

    # Methods as columns
    lines.extend(["", "---", "", "## Comparison by dataset (methods as columns)", ""])
    header = "| Dataset |"
    for method in methods_order:
        header += f" {method} (mIoU) | {method} (OA) |"
    lines.append(header)
    lines.append("|" + "---------|" + "----------|----------|" * len(methods_order))
    for ds in datasets_order:
        by_method = {r["method"]: r for r in by_dataset[ds]}
        row = f"| {ds} |"
        for method in methods_order:
            if method in by_method:
                r = by_method[method]
                row += f" {r['miou']:.4f} | {r['oa']:.4f} |"
            else:
                row += " — | — |"
        lines.append(row)

    # Per-dataset per-class IoU tables
    lines.extend(["", "---", "", "## Per-class IoU by dataset", ""])
    for ds in datasets_order:
        rows = by_dataset[ds]
        by_method = {r["method"]: r for r in rows}
        class_names = _ordered_classes([by_method[m] for m in methods_order if m in by_method])

        lines.append(f"### {ds}")
        lines.append("")
        header = "| Class |"
        sep = "|-------|"
        for method in methods_order:
            if method in by_method:
                header += f" {method} |"
                sep += "----------|"
        lines.append(header)
        lines.append(sep)

        for cls in class_names:
            row = f"| {cls} |"
            for method in methods_order:
                if method not in by_method:
                    continue
                pc = dict(by_method[method]["per_class"])
                val = pc.get(cls)
                row += f" {val:.4f} |" if val is not None else " — |"
            lines.append(row)

        # Bold mIoU row at the bottom
        miou_row = "| **mIoU** |"
        for method in methods_order:
            if method in by_method:
                miou_row += f" **{by_method[method]['miou']:.4f}** |"
        lines.append(miou_row)
        # OA row
        oa_row = "| OA |"
        for method in methods_order:
            if method in by_method:
                oa_row += f" {by_method[method]['oa']:.4f} |"
        lines.append(oa_row)
        lines.append("")

    # Parsed log paths
    lines.extend(["---", "", "## Parsed log files", ""])
    for r in sorted(results, key=lambda x: (x["dataset"], x["method"], x["log_path"])):
        try:
            rel = Path(r["log_path"]).relative_to(repo_root)
        except ValueError:
            rel = r["log_path"]
        lines.append(
            f"- `{rel}` — {r['dataset']} / {r['method']}: mIoU={r['miou']:.4f}, OA={r['oa']:.4f}"
        )
    lines.append("")

    return "\n".join(lines)


def main():
    experiments_dir = Path(os.environ.get("EXPERIMENTS_DIR", EXPERIMENTS_DIR))
    output_path = Path(os.environ.get("OUTPUT_MD", OUTPUT_MD))

    if not experiments_dir.is_dir():
        print(f"Experiments directory not found: {experiments_dir}")
        return

    results = collect_all_results(experiments_dir)
    print(
        f"Parsed {len(results)} evaluation result(s) "
        f"({len(find_all_eval_logs(experiments_dir))} eval.log + loveda train.log)."
    )

    repo_root = Path(os.environ.get("REPO_ROOT", REPO_ROOT))
    md = build_markdown(results, repo_root)
    output_path.write_text(md, encoding="utf-8")
    print(f"Written: {output_path}")


if __name__ == "__main__":
    main()
