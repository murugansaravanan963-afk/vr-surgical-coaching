"""SISVSE: Surgical Scene Segmentation in Robotic Gastrectomy (real + synthetic frames).

Expects the Supervisely export (Dataset Ninja / `dataset-tools`):

    <src>/meta.json
    <src>/<split>/img/<name>.jpg
    <src>/<split>/ann/<name>.jpg.json

Real data comes in 3 cross-validation folds (real_train_K / real_val_K). We use one
fold (real_val_K as validation) plus ALL synthetic splits for training. The synthetic
frames are rendered from a virtual surgery environment, which is exactly the kind of
imagery a VR simulator produces, so they are valuable for sim-to-real robustness.

  python -m surgseg.datasets.prepare_sisvse --src $RAW/sisvse --out $DATA/processed --fold 1
"""
from __future__ import annotations

import base64
import json
import re
import zlib
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from surgseg.datasets.common import SampleWriter, base_argparser, read_rgb
from surgseg.taxonomy import IGNORE, load_taxonomy

SYNTHETIC_SPLITS = ["manual_syn", "sean_manual_syn", "domain_random_syn",
                    "sean_spade_translation", "spade_domain_random_syn"]


def norm(name: str) -> str:
    return re.sub(r"[\s\-]+", "_", name.strip().lower())


def decode_bitmap(data: str) -> np.ndarray:
    """Supervisely bitmap: base64(zlib(PNG)); the mask is the alpha channel if present."""
    png = zlib.decompress(base64.b64decode(data))
    m = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    if m.ndim == 3:
        m = m[..., 3] if m.shape[2] == 4 else m[..., 0]
    return m > 0


REGION_TYPES = {"bitmap", "alpha_mask", "polygon"}


def box_of(obj: dict, h: int, w: int) -> tuple[int, int, int, int]:
    (xa, ya), (xb, yb) = obj["points"]["exterior"][:2]
    x1, x2 = sorted((int(round(xa)), int(round(xb))))
    y1, y2 = sorted((int(round(ya)), int(round(yb))))
    return max(0, x1), max(0, y1), min(w, x2 + 1), min(h, y2 + 1)


def object_mask(obj: dict, h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w), bool)
    gtype = obj["geometryType"]
    if gtype in ("bitmap", "alpha_mask"):
        m = decode_bitmap(obj["bitmap"]["data"])
        x0, y0 = obj["bitmap"]["origin"]
        mh, mw = min(m.shape[0], h - y0), min(m.shape[1], w - x0)
        out[y0:y0 + mh, x0:x0 + mw] = m[:mh, :mw]
    elif gtype == "polygon":
        canvas = np.zeros((h, w), np.uint8)
        ext = np.round(np.array(obj["points"]["exterior"])).astype(np.int32)
        cv2.fillPoly(canvas, [ext], 1)
        for hole in obj["points"].get("interior", []):
            cv2.fillPoly(canvas, [np.round(np.array(hole)).astype(np.int32)], 0)
        out = canvas.astype(bool)
    else:
        raise ValueError(f"unsupported geometry {gtype}")
    return out


def main() -> None:
    p = base_argparser(__doc__)
    p.add_argument("--fold", type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--no-synthetic", action="store_true")
    args = p.parse_args()
    src = Path(args.src)
    tax = load_taxonomy(args.taxonomy) if args.taxonomy else load_taxonomy()
    dmap = {norm(k): v for k, v in tax.dataset_maps["sisvse"].items() if not k.startswith("__")}
    suffixes = tax.dataset_maps["sisvse"]["__instrument_part_suffixes__"]
    labeled = tax.labeled_set("sisvse")
    inst = tax.class_id("instrument")

    def to_value(cls: str) -> int:
        c = norm(cls)
        if c in dmap:
            return tax.resolve(dmap[c])
        if any(c.endswith(s) for s in suffixes):
            return inst
        raise KeyError(f"SISVSE class '{cls}' missing from taxonomy.yaml")

    # Validate every class name up front so we fail before converting 20k images.
    meta = json.loads((src / "meta.json").read_text())
    for c in meta.get("classes", []):
        to_value(c["title"])

    # Paint order: tissue first, organs, objects, then instruments on top.
    def priority(v: int) -> int:
        return 0 if v == 0 else 3 if v == inst else 2 if tax.names[v] in (
            "surgical_object", "suture") else 1

    plan = [(f"real_train_{args.fold}", "train", False), (f"real_val_{args.fold}", "val", False)]
    if not args.no_synthetic:
        plan += [(s, "train", True) for s in SYNTHETIC_SPLITS]
    w = SampleWriter(args.out, "sisvse", args.max_side)
    n = 0
    stats: Counter = Counter()
    for split_dir, split, synthetic in plan:
        ann_dir = src / split_dir / "ann"
        if not ann_dir.is_dir():
            print(f"[sisvse] skipping missing split {split_dir}")
            continue
        for apath in sorted(ann_dir.glob("*.json")):
            ann = json.loads(apath.read_text())
            h, w_ = ann["size"]["height"], ann["size"]["width"]
            mask = np.zeros((h, w_), np.uint8)
            objs = [(to_value(o["classTitle"]), o) for o in ann["objects"]]
            regions = [(v, o) for v, o in objs if o["geometryType"] in REGION_TYPES]
            for v, o in sorted(regions, key=lambda t: priority(t[0])):   # stable sort
                mask[object_mask(o, h, w_)] = v
            # Bounding boxes are NOT masks. A box that duplicates a mask of its class is
            # dropped; a box-only object's area becomes "ignore" where nothing else is
            # painted, so its pixels are never taught as "not that class".
            for v, o in objs:
                g = o["geometryType"]
                if g in REGION_TYPES:
                    continue
                if g != "rectangle":
                    stats[f"skipped_{g}"] += 1
                    continue
                x1, y1, x2, y2 = box_of(o, h, w_)
                patch = mask[y1:y2, x1:x2]
                if patch.size and (patch == v).mean() >= 0.1:
                    stats["box_duplicates_mask"] += 1
                else:
                    patch[patch == 0] = IGNORE
                    stats["box_only_set_to_ignore"] += 1
            ipath = src / split_dir / "img" / apath.name[:-len(".json")]
            w.add(f"{split_dir}_{Path(ipath.name).stem}", read_rgb(ipath), mask,
                  group=split_dir, split=split, labeled=labeled, synthetic=synthetic)
            n += 1
            if args.limit and n >= args.limit:
                break
        if args.limit and n >= args.limit:
            break
    w.close()
    print(f"[sisvse] non-mask annotations: {dict(stats) or 'none'}")


if __name__ == "__main__":
    main()
