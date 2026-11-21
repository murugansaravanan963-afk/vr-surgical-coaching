"""Evaluate a checkpoint per dataset and fit a calibration temperature.

  python -m surgseg.evaluate --ckpt runs/student/best.pt --split test --fit-temperature

Calibration matters here: the VR runtime thresholds probabilities ("warn if
P(danger) > 0.3"), and raw network probabilities are typically over-confident.
The fitted temperature is written next to the checkpoint and baked in at export.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from surgseg.data import SegDataset, available_datasets, collate_eval, load_records
from surgseg.losses import PartialLabelLoss
from surgseg.metrics import PartialConfusion
from surgseg.model import load_for_inference
from surgseg.taxonomy import IGNORE, load_taxonomy


@torch.no_grad()
def run(model, loader, tax, names, device, crit=None, max_pixels=0):
    conf = PartialConfusion(tax.num_classes, len(names), device)
    zs, allowed_rows = [], []
    budget = max_pixels
    for x, m, lab, ds in loader:
        x, m, lab, ds = x.to(device), m.to(device), lab.to(device), ds.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x).float()
        conf.update(logits.argmax(1), m, lab, ds)
        if crit is not None and budget > 0:
            valid = m != IGNORE
            z = logits.permute(0, 2, 3, 1)[valid]
            a = crit.allowed(m, lab)[valid]
            keep = torch.randperm(len(z), device=device)[: min(budget, len(z) // 20 + 1)]
            zs.append(z[keep].cpu())
            allowed_rows.append(a[keep].cpu())
            budget -= len(keep)
    calib = (torch.cat(zs), torch.cat(allowed_rows)) if zs else None
    return conf.summary(tax.names, names), calib


def fit_temperature(z: torch.Tensor, allowed: torch.Tensor) -> float:
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def nll():
        opt.zero_grad()
        logp = F.log_softmax(z / log_t.exp(), dim=1)
        loss = -torch.logsumexp(logp.masked_fill(~allowed, float("-inf")), dim=1).mean()
        loss.backward()
        return loss

    opt.step(nll)
    return float(log_t.exp())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--data-root", default=None, help="defaults to the training config's")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--fit-temperature", action="store_true", help="fit T on the val split")
    ap.add_argument("--long-side", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_for_inference(args.ckpt, device)
    cfg = ck["cfg"]
    tax = load_taxonomy()
    root = os.path.expandvars(args.data_root or cfg["data"]["root"])
    names = available_datasets(root, args.datasets or list(cfg["data"]["datasets"]))
    long_side = args.long_side or cfg["data"].get("eval_long_side", 640)
    report = {"ckpt": args.ckpt, "step": ck["step"]}

    def loader(split):
        recs = load_records(root, names, split, unlabeled=False)
        ds = SegDataset(root, recs, tax.num_classes, names, train=False,
                        eval_long_side=long_side)
        return DataLoader(ds, batch_size=4, num_workers=args.workers, collate_fn=collate_eval)

    res, _ = run(model, loader(args.split), tax, names, device)
    report[args.split] = res
    print(json.dumps(res, indent=1))

    if args.fit_temperature:
        crit = PartialLabelLoss(tax).to(device)
        _, (z, a) = run(model, loader("val"), tax, names, device, crit, max_pixels=3_000_000)
        T = fit_temperature(z, a)
        report["temperature"] = T
        print(f"fitted temperature T = {T:.3f} on {len(z)} val pixels")

    out = Path(args.ckpt).with_suffix(f".eval_{args.split}.json")
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
