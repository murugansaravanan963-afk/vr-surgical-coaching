"""What does the model predict on pixels of a given class? (diagnosing 0.0 IoU classes)

  python -m surgseg.tools.confusion --ckpt runs/student_focus/best.pt \
      --dataset cholecseg8k --split val --classes duct connective_tissue

Prints, for each requested ground-truth class, the share of its pixels predicted as each
class. "connective_tissue -> fat 80%" means a confusion to fix with data/weights;
"-> background 95%" with a tiny pixel count means the class is barely visible.
"""
from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader

from surgseg.data import SegDataset, collate_eval, load_records
from surgseg.model import load_for_inference
from surgseg.taxonomy import load_taxonomy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--classes", nargs="+", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_for_inference(args.ckpt, device)
    tax = load_taxonomy()
    root = os.path.expandvars(args.data_root or ck["cfg"]["data"]["root"])
    recs = load_records(root, [args.dataset], args.split, unlabeled=False)
    ds = SegDataset(root, recs, tax.num_classes, [args.dataset], train=False,
                    eval_long_side=ck["cfg"]["data"].get("eval_long_side", 640))
    loader = DataLoader(ds, batch_size=4, num_workers=args.workers, collate_fn=collate_eval)
    want = [tax.class_id(c) for c in args.classes]
    counts = {c: torch.zeros(tax.num_classes, dtype=torch.long) for c in want}
    with torch.no_grad():
        for x, m, _, _ in loader:
            x, m = x.to(device), m.to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                pred = model(x).argmax(1)
            for c in want:
                sel = pred[m == c]
                if sel.numel():
                    counts[c] += torch.bincount(sel, minlength=tax.num_classes).cpu()
    print(f"{args.dataset}/{args.split}: {len(recs)} frames, ckpt step {ck['step']}")
    for c in want:
        tot = int(counts[c].sum())
        top = sorted(((int(n), i) for i, n in enumerate(counts[c].tolist()) if n), reverse=True)[:5]
        share = ", ".join(f"{tax.names[i]} {100 * n / tot:.1f}%" for n, i in top) if tot else "-"
        print(f"  GT {tax.names[c]:18s} {tot:>10d} px -> predicted as: {share}")


if __name__ == "__main__":
    main()
