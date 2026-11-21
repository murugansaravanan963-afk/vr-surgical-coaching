"""Refine processed indexes in place (no re-conversion needed).

1. Recount "present" classes per frame with an absolute pixel threshold. The converters
   used >= 0.1% of the frame, which drops thin structures (a visible cystic duct is often
   smaller), so those frames were never oversampled by the rare-class boost.
2. Optionally re-pick a dataset's VALIDATION videos so they actually contain the rare
   classes, while each rare class keeps most of its pixels in TRAIN. Test is untouched.
   (The original CholecSeg8k val videos had almost no duct/vein/connective tissue, so
   validation never measured them.)

  python -m surgseg.tools.refine_index --root $SURGSEG_DATA/processed \
      --auto-val cholecseg8k=2 --rare duct vein connective_tissue blood
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from surgseg.taxonomy import load_taxonomy


def pick_val_groups(group_counts: dict[str, dict[int, int]], candidates: list[str],
                    rare: list[int], n: int, max_val_share: float = 0.5) -> list[str]:
    """Greedy: maximise rare-class coverage in val, never moving more than max_val_share
    of any rare class's pixels out of train (a class only in one video stays in train)."""
    total = {r: sum(group_counts[g].get(r, 0) for g in candidates) for r in rare}
    chosen: list[str] = []
    in_val = {r: 0 for r in rare}
    for _ in range(n):
        best, best_key = None, None
        for g in candidates:
            if g in chosen:
                continue
            if any(total[r] and (in_val[r] + group_counts[g].get(r, 0)) > max_val_share * total[r]
                   for r in rare):
                continue
            newly = sum(1 for r in rare if in_val[r] == 0 and group_counts[g].get(r, 0) > 0)
            share = sum(group_counts[g].get(r, 0) / total[r] for r in rare if total[r])
            key = (newly, share)
            if best_key is None or key > best_key:
                best, best_key = g, key
        if best is None:
            break
        chosen.append(best)
        for r in rare:
            in_val[r] += group_counts[best].get(r, 0)
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--min-pixels", type=int, default=64,
                    help="a class counts as present in a frame from this many pixels")
    ap.add_argument("--auto-val", nargs="*", default=[], help="dataset=N_val_groups")
    ap.add_argument("--rare", nargs="*", default=["duct", "vein", "artery", "ureter",
                                                  "connective_tissue", "blood"])
    args = ap.parse_args()
    root = Path(args.root)
    tax = load_taxonomy()
    rare = [tax.class_id(n) for n in args.rare]
    auto = dict(a.split("=") for a in args.auto_val)

    for idx in sorted(root.glob("*/index.jsonl")):
        ds = idx.parent.name
        recs = [json.loads(l) for l in idx.read_text().splitlines() if l.strip()]
        if not any(r.get("mask") for r in recs):
            continue
        group_counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        changed = 0
        for r in recs:
            if not r.get("mask"):
                continue
            m = cv2.imread(str(root / r["mask"]), cv2.IMREAD_UNCHANGED)
            vals, cnts = np.unique(m, return_counts=True)
            present = set()
            for v, c in zip(vals.tolist(), cnts.tolist()):
                cls = tax.value_class(v)
                if cls is None:
                    continue
                group_counts[r["group"]][cls] += c
                if c >= args.min_pixels:
                    present.add(cls)
            present = sorted(present)
            if present != r.get("present"):
                changed += 1
                r["present"] = present
        msg = f"[{ds}] present updated on {changed}/{len(recs)} frames"

        if ds in auto:
            candidates = sorted({r["group"] for r in recs if r["split"] in ("train", "val")})
            val = pick_val_groups(group_counts, candidates, rare, int(auto[ds]))
            if not val:   # never leave a dataset without validation
                msg += "; WARNING: no val videos fit the rare-class constraint - split unchanged"
            else:
                for r in recs:
                    if r["split"] in ("train", "val"):
                        r["split"] = "val" if r["group"] in val else "train"
                cover = {tax.names[c]: sum(group_counts[g].get(c, 0) for g in val) for c in rare}
                msg += f"; val groups -> {val}; rare-class pixels in val: {cover}"

        bak = idx.with_suffix(".jsonl.bak")
        if not bak.exists():
            shutil.copy(idx, bak)                  # original kept once, for undo
        tmp = idx.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in recs))
        tmp.replace(idx)
        print(msg, flush=True)


if __name__ == "__main__":
    main()
