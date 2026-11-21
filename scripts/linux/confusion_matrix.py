"""Confusion matrix + per-class scores of a checkpoint on the real test sets.

Same data, resolution and partial-label rules as surgseg.evaluate, but it also SAVES the
confusion matrix (evaluate only keeps the per-class IoU). Needs 1 GPU, ~15 min:

  source scripts/linux/env.sh
  python scripts/linux/confusion_matrix.py --ckpt $SURGSEG_MODELS/student_kd/best.pt --out $SURGSEG_MODELS/student_kd/confusion_test.json

Output JSON: classes, datasets, per dataset {counts[true][pred], labeled classes}, plus
per-class IoU / recall / precision per dataset and pooled over datasets.
Rows = true class, columns = predicted class. Row 0 "background" = "none of the classes
this dataset labels"; predictions of classes a dataset does not label count as 0 there.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from surgseg.data import SegDataset, available_datasets, collate_eval, load_records
from surgseg.metrics import PartialConfusion
from surgseg.model import load_for_inference
from surgseg.taxonomy import load_taxonomy


def scores(cm: np.ndarray, classes: list[int]) -> dict:
    tp = np.diag(cm)
    true_n, pred_n = cm.sum(1), cm.sum(0)
    out = {}
    for c in classes:
        out[c] = {"iou": float(tp[c] / max(true_n[c] + pred_n[c] - tp[c], 1)),
                  "recall": float(tp[c] / max(true_n[c], 1)),
                  "precision": float(tp[c] / max(pred_n[c], 1)),
                  "true_pixels": int(true_n[c])}
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_for_inference(os.path.expanduser(args.ckpt), device)
    cfg = ck["cfg"]
    tax = load_taxonomy()
    root = os.path.expandvars(cfg["data"]["root"])
    names = available_datasets(root, list(cfg["data"]["datasets"]))
    recs = load_records(root, names, args.split, unlabeled=False)
    ds = SegDataset(root, recs, tax.num_classes, names, train=False,
                    eval_long_side=cfg["data"].get("eval_long_side", 640))
    loader = DataLoader(ds, batch_size=4, num_workers=args.workers, collate_fn=collate_eval)

    conf = PartialConfusion(tax.num_classes, len(names), device)
    for x, m, lab, d in loader:
        x, m, lab, d = x.to(device), m.to(device), lab.to(device), d.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x).float()
        conf.update(logits.argmax(1), m, lab, d)

    cms = conf.cm.cpu().numpy()
    seen = conf.seen.cpu().numpy()
    report = {"ckpt": args.ckpt, "step": ck.get("step"), "split": args.split,
              "classes": list(tax.names), "datasets": names, "per_dataset": {}}
    pooled = np.zeros_like(cms[0])
    for i, n in enumerate(names):
        cm = cms[i]
        if cm.sum() == 0:
            continue
        pooled += cm
        cls = [c for c in range(1, tax.num_classes) if seen[i, c] and cm[c].sum() > 0]
        report["per_dataset"][n] = {
            "labeled": [tax.names[c] for c in cls],
            "counts": cm.tolist(),
            "scores": {tax.names[c]: s for c, s in scores(cm, cls).items()},
        }
    pooled_cls = [c for c in range(1, tax.num_classes) if pooled[c].sum() > 0]
    report["pooled"] = {"counts": pooled.tolist(),
                        "scores": {tax.names[c]: s for c, s in scores(pooled, pooled_cls).items()}}
    with open(os.path.expanduser(args.out), "w") as f:
        json.dump(report, f)
    print(f"wrote {args.out}")
    for c, s in report["pooled"]["scores"].items():
        print(f"{c:18s} IoU {s['iou']:.3f}  recall {s['recall']:.3f}  precision {s['precision']:.3f}")


if __name__ == "__main__":
    main()
