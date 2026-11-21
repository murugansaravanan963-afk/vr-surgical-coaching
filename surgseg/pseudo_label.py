"""Teacher-ensemble pseudo-labels (self-training / "noisy student").

Two uses, both writing a new dataset "<dataset>_pl" whose val/test records are the
untouched originals (evaluation never sees pseudo-labels):

  unlabeled frames  -> hard labels where the ensemble is confident (else ignore=255).
  labeled frames    -> LABEL COMPLETION: ground truth is kept as-is; pixels whose GT is
                       ambiguous (dataset background, a group like "GI tract", a partial
                       label) get the teacher's class ONLY if that class is permitted by
                       the GT there and the ensemble is confident. EndoVis18 frames gain
                       liver/fat labels, DSAD single-organ frames gain the other organs...

Predictions average several checkpoints x horizontal flip x scales, after each
checkpoint's fitted temperature, so the confidence threshold means something.

  python -m surgseg.pseudo_label --ckpts runs/teacher_a/best.pt runs/teacher_b/best.pt \
      --data-root $DATA/processed --datasets endovis18 sarrarp50 endoscapes \
      --shard 0 --num-shards 8      # one process per GPU: shard 0..7
  python -m surgseg.pseudo_label --merge --data-root $DATA/processed \
      --datasets endovis18 sarrarp50 endoscapes
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from surgseg.data import to_input
from surgseg.model import load_for_inference
from surgseg.taxonomy import IGNORE, load_taxonomy


class Ensemble:
    def __init__(self, ckpts, device, scales=(0.75, 1.0, 1.25), flip=True, long_side=768):
        self.models, self.temps = [], []
        for c in ckpts:
            m, _ = load_for_inference(c, device)
            ev = Path(c).with_suffix(".eval_test.json")
            self.temps.append(json.loads(ev.read_text()).get("temperature", 1.0)
                              if ev.exists() else 1.0)
            self.models.append(m)
        self.device, self.scales, self.flip, self.long_side = device, scales, flip, long_side

    @torch.no_grad()
    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        """(H, W, 3) uint8 -> (C, H, W) averaged calibrated probabilities."""
        h, w = rgb.shape[:2]
        x0 = to_input(torch.from_numpy(rgb).permute(2, 0, 1))[None].to(self.device)
        acc, n = 0, 0
        for s in self.scales:
            k = self.long_side * s / max(h, w)
            sh, sw = max(32, round(h * k / 32) * 32), max(32, round(w * k / 32) * 32)
            x = F.interpolate(x0, size=(sh, sw), mode="bilinear", align_corners=False)
            for flip in ([False, True] if self.flip else [False]):
                xi = x.flip(-1) if flip else x
                for m, t in zip(self.models, self.temps):
                    with torch.autocast(self.device.type, dtype=torch.bfloat16,
                                        enabled=self.device.type == "cuda"):
                        logits = m(xi).float()
                    if flip:
                        logits = logits.flip(-1)
                    p = torch.softmax(logits / t, 1)
                    acc = acc + F.interpolate(p, size=(h, w), mode="bilinear",
                                              align_corners=False)
                    n += 1
        return (acc / n)[0].cpu().numpy()


def complete(gt: np.ndarray, probs: np.ndarray, allowed: np.ndarray, tau: float) -> np.ndarray:
    """Fill ambiguous GT pixels with the teacher's class when permitted and confident."""
    pred = probs.argmax(0)
    conf = probs.max(0)
    exact = allowed[gt].sum(-1) == 1                       # GT already names one class
    ok = allowed[gt, pred] & (conf >= tau) & ~exact & (gt != IGNORE) & (pred > 0)
    out = gt.copy()
    out[ok] = pred[ok]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="*", default=[])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--tau", type=float, default=0.9, help="min ensemble confidence")
    ap.add_argument("--long-side", type=int, default=768)
    ap.add_argument("--scales", type=float, nargs="+", default=[0.75, 1.0, 1.25])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true", help="combine shard outputs")
    args = ap.parse_args()
    root = Path(args.data_root)
    tax = load_taxonomy()

    if args.merge:
        for d in args.datasets:
            parts = sorted((root / f"{d}_pl").glob("index.part*.jsonl"))
            recs = [l for p in parts for l in p.read_text().splitlines() if l.strip()]
            with open(root / d / "index.jsonl") as f:     # untouched val/test records
                keep = [l for l in f.read().splitlines()
                        if l.strip() and json.loads(l)["split"] != "train"]
            fixed = []
            for l in keep:
                r = json.loads(l)
                r["dataset"] = f"{d}_pl"
                fixed.append(json.dumps(r))
            (root / f"{d}_pl" / "index.jsonl").write_text("".join(l + "\n" for l in recs + fixed))
            print(f"{d}_pl: {len(recs)} pseudo-labelled train + {len(fixed)} val/test records")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ens = Ensemble(args.ckpts, device, tuple(args.scales), long_side=args.long_side)
    all_classes = list(range(1, tax.num_classes))
    for d in args.datasets:
        out_dir = root / f"{d}_pl"
        (out_dir / "masks").mkdir(parents=True, exist_ok=True)
        recs = [json.loads(l) for l in (root / d / "index.jsonl").read_text().splitlines()
                if l.strip()]
        recs = [r for r in recs if r["split"] == "train"][args.shard::args.num_shards]
        out, stats = [], {"filled": 0, "kept_ignore": 0, "pixels": 0}
        for i, r in enumerate(recs):
            uid = Path(r["image"]).stem
            rel = f"{d}_pl/masks/{uid}.png"
            labeled = r["labeled"] if r.get("mask") else all_classes
            if (root / rel).exists():          # resumed job: this frame is already done
                new = cv2.imread(str(root / rel), cv2.IMREAD_UNCHANGED)
                present = sorted({int(v) for v in np.unique(new) if 0 < v < tax.num_classes})
                out.append(json.dumps({**r, "mask": rel, "dataset": f"{d}_pl",
                                       "labeled": labeled, "present": present, "pseudo": True}))
                continue
            rgb = cv2.cvtColor(cv2.imread(str(root / r["image"])), cv2.COLOR_BGR2RGB)
            probs = ens(rgb)
            if r.get("mask"):
                gt = cv2.imread(str(root / r["mask"]), cv2.IMREAD_UNCHANGED)
                allowed = tax.allowed_matrix(r["labeled"])
                new = complete(gt, probs, allowed, args.tau)
                stats["filled"] += int((new != gt).sum())
                labeled = r["labeled"]
            else:
                new = probs.argmax(0).astype(np.uint8)
                new[probs.max(0) < args.tau] = IGNORE
                stats["kept_ignore"] += int((new == IGNORE).sum())
                labeled = all_classes                       # teacher predicts every class
            stats["pixels"] += new.size
            tmp = root / f"{d}_pl/masks/{uid}.tmp.png"
            cv2.imwrite(str(tmp), new)
            tmp.replace(root / rel)            # atomic: a killed job never leaves half a mask
            present = sorted({int(v) for v in np.unique(new) if 0 < v < tax.num_classes})
            out.append(json.dumps({**r, "mask": rel, "dataset": f"{d}_pl", "labeled": labeled,
                                   "present": present, "pseudo": True}))
            if i % 200 == 0:
                print(f"[{d} shard {args.shard}] {i}/{len(recs)}", flush=True)
        (out_dir / f"index.part{args.shard:03d}.jsonl").write_text("".join(l + "\n" for l in out))
        print(f"[{d}] shard {args.shard}: {len(out)} frames, "
              f"{100 * stats['filled'] / max(stats['pixels'], 1):.1f}% px completed, "
              f"{100 * stats['kept_ignore'] / max(stats['pixels'], 1):.1f}% px left unlabeled")


if __name__ == "__main__":
    main()
