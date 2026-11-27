"""Side-by-side per-class IoU from several evaluate.py result files.

  python scripts/linux/compare_evals.py A.json B.json [C.json ...] [--key test_with_class_bias]

Per class: the mean IoU over the datasets that label it (same as the README numbers).
Also prints the mean over datasets. Weak classes are listed first.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

WEAK = ["connective_tissue", "duct", "vein", "artery", "blood", "pancreas", "ureter",
        "vesicular_glands", "suture", "surgical_object"]


def per_class(res: dict) -> dict:
    acc: dict[str, list[float]] = {}
    for ds, v in res.items():
        if ds == "mean_over_datasets" or not isinstance(v, dict):
            continue
        for c, iou in v["per_class"].items():
            acc.setdefault(c, []).append(iou)
    return {c: sum(v) / len(v) for c, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    cols = []
    for f in args.files:
        rep = json.loads(Path(f).read_text())
        name = Path(f).parent.name + ":" + Path(f).stem.split(".eval_")[-1]
        cols.append((name, rep[args.split]))
        if f"{args.split}_with_class_bias" in rep:
            cols.append((name + "+bias", rep[f"{args.split}_with_class_bias"]))
    tables = [(n, per_class(r), r["mean_over_datasets"]) for n, r in cols]
    classes = WEAK + sorted({c for _, t, _ in tables for c in t} - set(WEAK))
    w = max(18, *(len(n) for n, _, _ in tables))
    print("class".ljust(20) + "".join(n.rjust(w + 2) for n, _, _ in tables))
    for c in classes:
        if not any(c in t for _, t, _ in tables):
            continue
        row = "".join((f"{t[c]:.3f}" if c in t else "-").rjust(w + 2) for _, t, _ in tables)
        print(("* " if c in WEAK else "  ") + c.ljust(18) + row)
    print("MEAN over datasets".ljust(20) + "".join(f"{m:.4f}".rjust(w + 2) for _, _, m in tables))


if __name__ == "__main__":
    main()
