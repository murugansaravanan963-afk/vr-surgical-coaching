"""Shared helpers for the dataset converters.

Every converter writes the same on-disk format:

    <out>/<dataset>/images/<uid>.jpg
    <out>/<dataset>/masks/<uid>.png        uint8, values per surgseg.taxonomy
    <out>/<dataset>/index.jsonl            one record per sample:
        {"image", "mask", "dataset", "group", "split", "labeled"}

"group" is the video / surgery / case id. Splits MUST be made at this level: frames
from one video are near-duplicates, so a frame-level split leaks and inflates scores.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from surgseg.taxonomy import load_taxonomy


def base_argparser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--src", required=True, help="raw dataset root")
    p.add_argument("--out", required=True, help="processed data root")
    p.add_argument("--taxonomy", default=None)
    p.add_argument("--max-side", type=int, default=1024,
                   help="downscale so the longer side is at most this (saves I/O)")
    p.add_argument("--limit", type=int, default=0, help="debug: stop after N samples")
    return p


class SampleWriter:
    def __init__(self, out_root: str | Path, dataset: str, max_side: int = 1024):
        self.root = Path(out_root) / dataset
        (self.root / "images").mkdir(parents=True, exist_ok=True)
        (self.root / "masks").mkdir(parents=True, exist_ok=True)
        self.dataset = dataset
        self.max_side = max_side
        self.records: list[dict] = []
        self.value_counts: Counter = Counter()
        self.tax = load_taxonomy()
        self.uids: set[str] = set()

    def add(self, uid: str, image_rgb: np.ndarray, mask: np.ndarray | None,
            group: str, split: str, labeled: list[int], **extra) -> None:
        """mask=None stores an UNLABELED frame (used only for teacher distillation)."""
        assert image_rgb.ndim == 3 and image_rgb.shape[2] == 3, image_rgb.shape
        if mask is not None:
            assert mask.shape == image_rgb.shape[:2], (mask.shape, image_rgb.shape)
            assert mask.dtype == np.uint8
        h, w = image_rgb.shape[:2]
        s = self.max_side / max(h, w)
        if s < 1:
            size = (round(w * s), round(h * s))
            image_rgb = cv2.resize(image_rgb, size, interpolation=cv2.INTER_AREA)
            if mask is not None:
                mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        uid = uid.replace("/", "_").replace("\\", "_")
        if uid in self.uids:     # would silently overwrite another frame's files
            raise ValueError(f"[{self.dataset}] duplicate sample id {uid!r}")
        self.uids.add(uid)
        img_rel = f"{self.dataset}/images/{uid}.jpg"
        cv2.imwrite(str(self.root.parent / img_rel), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        rec = {"image": img_rel, "mask": None, "dataset": self.dataset, "group": str(group),
               "split": split, "labeled": sorted(int(c) for c in labeled), "present": []}
        if mask is not None:
            msk_rel = f"{self.dataset}/masks/{uid}.png"
            cv2.imwrite(str(self.root.parent / msk_rel), mask)
            vals, cnts = np.unique(mask, return_counts=True)
            self.value_counts.update(dict(zip(vals.tolist(), cnts.tolist())))
            # classes definitely present (>= 0.1% of the frame) - drives rare-class oversampling
            present = {int(self.tax.value_class(v)) for v, c in zip(vals, cnts)
                       if c >= 1e-3 * mask.size and self.tax.value_class(v) is not None}
            rec.update(mask=msk_rel, present=sorted(present))
        rec.update(extra)
        self.records.append(rec)

    def close(self) -> None:
        with open(self.root / "index.jsonl", "w") as f:
            for r in self.records:
                f.write(json.dumps(r) + "\n")
        tax = self.tax
        splits = Counter(r["split"] for r in self.records)
        groups = {s: len({r["group"] for r in self.records if r["split"] == s}) for s in splits}
        unl = sum(r["mask"] is None for r in self.records)
        print(f"[{self.dataset}] wrote {len(self.records)} samples ({unl} unlabeled) -> {self.root}")
        for s in sorted(splits):
            print(f"   {s:5s}: {splits[s]:6d} frames from {groups[s]} videos/surgeries")
        total = sum(self.value_counts.values()) or 1
        print("   pixel share by mask value:")
        for v, c in sorted(self.value_counts.items()):
            name = tax.value_name(v)
            print(f"     {v:3d} {name:18s} {100 * c / total:6.2f}%")


def read_rgb(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_gray(path: str | Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(path)
    if m.ndim == 3:   # grey masks saved as RGB(A): all channels are equal
        m = m[..., 0]
    return m


def assign_split(group: str, val: set[str], test: set[str]) -> str:
    return "test" if group in test else "val" if group in val else "train"
