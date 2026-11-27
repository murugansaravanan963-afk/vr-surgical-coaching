"""Evaluate a checkpoint per dataset and fit a calibration temperature.

  python -m surgseg.evaluate --ckpt runs/student/best.pt --split test --fit-temperature

Calibration matters here: the VR runtime thresholds probabilities ("warn if
P(danger) > 0.3"), and raw network probabilities are typically over-confident.
The fitted temperature is written next to the checkpoint and baked in at export.

  python -m surgseg.evaluate --ckpt ... --split test --fit-class-bias

also fits one additive logit offset per class on the VAL split (to maximise mean IoU,
mostly by un-suppressing rare thin classes) and reports test both with and without it.
Test is only used to check the result, never to tune it. The offsets change only the
argmax "labels" output at export; the calibrated probabilities stay untouched.
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
from surgseg.taxonomy import GROUP_BASE, IGNORE, load_taxonomy


@torch.no_grad()
def run(model, loader, tax, names, device, crit=None, max_pixels=0, bias=None,
        bias_pixels=0, bias_stride=4):
    """Returns (summary, calibration sample, bias-fitting sample).

    bias: optional (C,) logit offsets applied before the argmax.
    bias_pixels: if > 0, also collect about this many exactly-labelled pixels, spread
    evenly over the split (every bias_stride-th row/column, then a random subset), as
    (logits fp16, target, dataset index, labelled set)."""
    conf = PartialConfusion(tax.num_classes, len(names), device)
    zs, allowed_rows = [], []
    bz, bt, bd, bl = [], [], [], []
    budget, keep_p = max_pixels, None
    for x, m, lab, ds in loader:
        x, m, lab, ds = x.to(device), m.to(device), lab.to(device), ds.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x).float()
        scored = logits if bias is None else logits + bias.view(1, -1, 1, 1).to(device)
        conf.update(scored.argmax(1), m, lab, ds)
        if crit is not None and budget > 0:
            valid = m != IGNORE
            z = logits.permute(0, 2, 3, 1)[valid]
            a = crit.allowed(m, lab)[valid]
            keep = torch.randperm(len(z), device=device)[: min(budget, len(z) // 20 + 1)]
            zs.append(z[keep].cpu())
            allowed_rows.append(a[keep].cpu())
            budget -= len(keep)
        if bias_pixels > 0:
            s = bias_stride
            z = logits[:, :, ::s, ::s].permute(0, 2, 3, 1)
            t = m[:, ::s, ::s]
            if keep_p is None:   # spread the budget evenly over the whole split
                keep_p = min(1.0, bias_pixels / (len(loader.dataset) * t[0].numel()))
            keep = (t < GROUP_BASE) & (torch.rand(t.shape, device=device) < keep_p)
            n = int(keep.sum())
            if n:
                img = torch.arange(len(t), device=device)[:, None, None].expand_as(t)[keep]
                bz.append(z[keep].half().cpu())
                bt.append(t[keep].cpu())
                bd.append(ds[img].cpu())
                bl.append(lab[img].cpu())
    calib = (torch.cat(zs), torch.cat(allowed_rows)) if zs else None
    bsample = (torch.cat(bz), torch.cat(bt), torch.cat(bd), torch.cat(bl)) if bz else None
    return conf.summary(tax.names, names), calib, bsample


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


def partial_miou(z, t, d, lab, bias, n_ds: int) -> float:
    """Mean over datasets of per-dataset mIoU, with the same partial-label rules as
    PartialConfusion: predictions of classes a sample does not label count as background."""
    C = z.shape[1]
    pred = (z + bias).argmax(1)
    lab = lab.clone()
    lab[:, 0] = True
    ok = lab.gather(1, pred[:, None]).squeeze(1)
    pred = torch.where(ok, pred, torch.zeros_like(pred))
    cm = torch.bincount((d * C + t) * C + pred, minlength=n_ds * C * C).view(n_ds, C, C).double()
    tp = cm.diagonal(dim1=1, dim2=2)
    iou = tp / (cm.sum(1) + cm.sum(2) - tp).clamp_min(1)
    seen = torch.zeros(n_ds, C, dtype=torch.long, device=z.device)
    seen.index_add_(0, d, lab.long())
    use = (seen > 0) & (cm.sum(2) > 0)
    use[:, 0] = False
    per_ds = [(iou[k][use[k]]).mean() for k in range(n_ds) if use[k].any()]
    return float(torch.stack(per_ds).mean()) if per_ds else float("nan")


def fit_class_bias(sample, n_ds: int, grid=None, passes: int = 2, min_gain: float = 1e-4,
                   device="cpu"):
    """Coordinate ascent over one logit offset per class (background stays 0)."""
    z, t, d, lab = (v.to(device) for v in sample)
    z = z.float()
    C = z.shape[1]
    grid = grid if grid is not None else [x / 4 for x in range(-8, 9)]   # -2.0 .. +2.0
    bias = torch.zeros(C, device=device)
    best = partial_miou(z, t, d, lab, bias, n_ds)
    start = best
    for _ in range(passes):
        improved = False
        for c in range(1, C):
            keep_val = float(bias[c])
            for g in grid:
                if g == keep_val:
                    continue
                bias[c] = g
                s = partial_miou(z, t, d, lab, bias, n_ds)
                if s > best + min_gain:
                    best, keep_val, improved = s, g, True
            bias[c] = keep_val
        if not improved:
            break
    return bias.cpu(), start, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--data-root", default=None, help="defaults to the training config's")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--fit-temperature", action="store_true", help="fit T on the val split")
    ap.add_argument("--fit-class-bias", action="store_true",
                    help="fit per-class logit offsets on val; report the split with and without")
    ap.add_argument("--long-side", type=int, default=None)
    ap.add_argument("--tag", default="", help="suffix for the output json, e.g. ls640")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_for_inference(args.ckpt, device)
    cfg = ck["cfg"]
    tax = load_taxonomy()
    root = os.path.expandvars(args.data_root or cfg["data"]["root"])
    names = available_datasets(root, args.datasets or list(cfg["data"]["datasets"]))
    long_side = args.long_side or cfg["data"].get("eval_long_side", 640)
    report = {"ckpt": args.ckpt, "step": ck["step"], "long_side": long_side}

    def loader(split):
        recs = load_records(root, names, split, unlabeled=False)
        ds = SegDataset(root, recs, tax.num_classes, names, train=False,
                        eval_long_side=long_side)
        return DataLoader(ds, batch_size=4, num_workers=args.workers, collate_fn=collate_eval)

    res, _, _ = run(model, loader(args.split), tax, names, device)
    report[args.split] = res
    print(json.dumps(res, indent=1))
    print(f"{args.split} mean over datasets: {res['mean_over_datasets']:.4f} (long side {long_side})")

    if args.fit_temperature:
        crit = PartialLabelLoss(tax).to(device)
        _, (z, a), _ = run(model, loader("val"), tax, names, device, crit, max_pixels=3_000_000)
        T = fit_temperature(z, a)
        report["temperature"] = T
        print(f"fitted temperature T = {T:.3f} on {len(z)} val pixels")

    if args.fit_class_bias:
        _, _, sample = run(model, loader("val"), tax, names, device, bias_pixels=6_000_000)
        bias, before, after = fit_class_bias(sample, len(names), device=device)
        print(f"class bias fitted on {len(sample[0])} val pixels: val mIoU {before:.4f} -> {after:.4f}")
        print("offsets:", {tax.names[c]: round(float(bias[c]), 2) for c in range(1, len(bias))
                           if bias[c] != 0})
        res_b, _, _ = run(model, loader(args.split), tax, names, device, bias=bias)
        report["class_bias"] = [float(b) for b in bias]
        report["class_bias_val"] = {"before": before, "after": after}
        report[f"{args.split}_with_class_bias"] = res_b
        print(f"{args.split} mean over datasets with class bias: "
              f"{res_b['mean_over_datasets']:.4f} (without: {res['mean_over_datasets']:.4f})")

    suffix = f".eval_{args.split}" + (f"_{args.tag}" if args.tag else "") + ".json"
    out = Path(args.ckpt).with_suffix(suffix)
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
