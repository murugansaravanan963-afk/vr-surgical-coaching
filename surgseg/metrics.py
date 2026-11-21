"""Per-dataset IoU that respects each sample's labeled set.

A prediction of a class the sample did not label is counted as "background" for that
sample (we cannot tell whether it is right), and group / ignore pixels are skipped.
"""
from __future__ import annotations

import numpy as np
import torch

from surgseg.taxonomy import GROUP_BASE


class PartialConfusion:
    def __init__(self, num_classes: int, num_datasets: int, device="cpu"):
        self.C = num_classes
        self.cm = torch.zeros(num_datasets, num_classes, num_classes, dtype=torch.long,
                              device=device)
        self.seen = torch.zeros(num_datasets, num_classes, dtype=torch.bool, device=device)

    @torch.no_grad()
    def update(self, pred, target, labeled, ds_idx):
        """pred/target (B,H,W) long, labeled (B,C) bool, ds_idx (B,) long."""
        for b in range(pred.shape[0]):
            lab = labeled[b].clone()
            lab[0] = True
            p = torch.where(lab[pred[b]], pred[b], torch.zeros_like(pred[b]))
            t = target[b]
            keep = t < GROUP_BASE
            idx = t[keep] * self.C + p[keep]
            self.cm[ds_idx[b]] += torch.bincount(idx, minlength=self.C ** 2).view(self.C, self.C)
            self.seen[ds_idx[b]] |= lab

    def all_reduce(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.cm)
            seen = self.seen.long()
            torch.distributed.all_reduce(seen)
            self.seen = seen > 0

    def summary(self, class_names, dataset_names) -> dict:
        out = {}
        for d, dname in enumerate(dataset_names):
            cm = self.cm[d].double().cpu().numpy()
            if cm.sum() == 0:
                continue
            tp = np.diag(cm)
            iou = tp / np.maximum(cm.sum(0) + cm.sum(1) - tp, 1)
            cls = [c for c in range(1, self.C) if self.seen[d, c] and cm[c].sum() > 0]
            out[dname] = {"mIoU": float(np.mean(iou[cls])) if cls else float("nan"),
                          "per_class": {class_names[c]: float(iou[c]) for c in cls}}
        vals = [v["mIoU"] for v in out.values() if not np.isnan(v["mIoU"])]
        out["mean_over_datasets"] = float(np.mean(vals)) if vals else float("nan")
        return out
