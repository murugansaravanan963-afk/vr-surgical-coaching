"""m2caiSeg (laparoscopic cholecystectomy, 307 frames, 19 classes; CC BY-NC-SA 4.0).

Hugging Face: MedOtter/m2caiSeg (parquet: image, mask, image_id; mask value = class id).
Small, but one of the few sources labelling arteries, fat and liver together.
No video ids are published, so its official test split is kept as test and nothing is
used for validation (model selection relies on the other datasets).

  python -m surgseg.datasets.prepare_m2caiseg --src MedOtter/m2caiSeg --out $DATA/processed
"""
from __future__ import annotations

import numpy as np

from surgseg.datasets.common import SampleWriter, base_argparser
from surgseg.taxonomy import IGNORE, load_taxonomy

CLASS_NAMES = ["background", "grasper", "bipolar", "hook", "scissors", "clipper", "irrigator",
               "specimen-bag", "trocars", "clip", "liver", "gall-bladder", "fat", "upperwall",
               "artery", "intestine", "bile", "blood", "unknown"]


def main() -> None:
    p = base_argparser(__doc__)
    args = p.parse_args()
    from datasets import load_dataset

    tax = load_taxonomy(args.taxonomy) if args.taxonomy else load_taxonomy()
    dmap = tax.dataset_maps["m2caiseg"]
    lut = np.full(256, IGNORE, np.uint8)
    for v, name in enumerate(CLASS_NAMES):
        lut[v] = tax.resolve(dmap[name])
    labeled = tax.labeled_set("m2caiseg")
    w = SampleWriter(args.out, "m2caiseg", args.max_side)
    n = 0
    for hf_split, split in (("train", "train"), ("test", "test")):
        for row in load_dataset(args.src, split=hf_split):
            img = np.asarray(row["image"].convert("RGB"))
            raw = np.asarray(row["mask"])
            raw = raw[..., 0] if raw.ndim == 3 else raw
            if raw.max() >= len(CLASS_NAMES):
                raise ValueError(f"{row['image_id']}: mask value {raw.max()} > 18")
            w.add(f"{hf_split}_{row['image_id']}", img, lut[raw.astype(np.uint8)],
                  group=f"{hf_split}_{row['image_id']}", split=split, labeled=labeled)
            n += 1
            if args.limit and n >= args.limit:
                w.close()
                return
    w.close()


if __name__ == "__main__":
    main()
