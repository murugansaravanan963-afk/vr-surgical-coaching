"""Sanity-check processed data: counts, split leakage, labeled sets, and overlay previews.

  python -m surgseg.tools.check_data --root $SURGSEG_DATA/processed

Look at the preview PNGs before training! A wrong colour/value mapping in a converter
is the most likely silent failure, and it is obvious in a picture.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from surgseg.taxonomy import GROUP_BASE, IGNORE, PARTIAL_BASE, load_taxonomy


def overlay(root: Path, rec: dict, tax) -> np.ndarray:
    img = cv2.imread(str(root / rec["image"]))
    if not rec.get("mask"):                     # unlabeled frame: show the image only
        return cv2.resize(img, (512, round(512 * img.shape[0] / img.shape[1])))
    m = cv2.imread(str(root / rec["mask"]), cv2.IMREAD_UNCHANGED)
    col = np.zeros_like(img)
    for v in np.unique(m):
        if v == IGNORE:
            c = (255, 255, 255)
        elif v >= PARTIAL_BASE:
            c = tuple(int(x) for x in tax.colors[v - PARTIAL_BASE][::-1])
        elif v >= GROUP_BASE:
            c = (255, 0, 255)
        else:
            c = tuple(int(x) for x in tax.colors[v][::-1])
        col[m == v] = c
    out = cv2.addWeighted(img, 0.55, col, 0.45, 0)
    names = [tax.value_name(v) for v in np.unique(m) if v != 0]
    cv2.putText(out, ",".join(names)[:90], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)
    return cv2.resize(out, (512, round(512 * out.shape[0] / out.shape[1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--previews", type=int, default=12)
    args = ap.parse_args()
    root = Path(args.root)
    tax = load_taxonomy()
    ok = True
    for idx in sorted(root.glob("*/index.jsonl")):
        recs = [json.loads(l) for l in idx.read_text().splitlines()]
        ds = idx.parent.name
        groups = defaultdict(set)
        for r in recs:
            groups[r["split"]].add(r["group"])
        leak = (groups["train"] & groups["val"]) | (groups["train"] & groups["test"])
        labeled = Counter(tuple(r["labeled"]) for r in recs)
        print(f"\n== {ds}: {len(recs)} samples  " +
              "  ".join(f"{s}={sum(r['split'] == s for r in recs)}" for s in ("train", "val", "test")))
        for lab, n in labeled.most_common(5):
            print(f"   labeled set ({n:5d} samples): {[tax.names[c] for c in lab]}")
        if leak:
            ok = False
            print(f"   !! LEAK: groups in train and val/test: {sorted(leak)[:5]}")
        unlabeled = all(not r.get("mask") for r in recs)
        if unlabeled:
            print("   (unlabeled frames: used only for teacher distillation)")
        elif not groups["val"]:
            print("   (no validation split - model selection relies on other datasets)")
        random.seed(0)
        picks = random.sample(recs, min(args.previews, len(recs)))
        tiles = [overlay(root, r, tax) for r in picks]
        rows = [np.hstack(tiles[i:i + 4] + [np.zeros_like(tiles[0])] * (4 - len(tiles[i:i + 4])))
                for i in range(0, len(tiles), 4)]
        rows = [cv2.resize(r, (rows[0].shape[1], r.shape[0])) for r in rows]
        h = max(r.shape[0] for r in rows)
        rows = [np.pad(r, ((0, h - r.shape[0]), (0, 0), (0, 0))) for r in rows]
        cv2.imwrite(str(root / f"preview_{ds}.jpg"), np.vstack(rows))
        print(f"   preview -> {root / f'preview_{ds}.jpg'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
