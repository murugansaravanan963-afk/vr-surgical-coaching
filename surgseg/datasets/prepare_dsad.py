"""Dresden Surgical Anatomy Dataset (robot-assisted rectal resections, da Vinci Xi).

Uses the Hugging Face release (nct-tso/dsad, CC-BY-4.0), which already ships
surgery-level train/validation/test splits (Kolbinger et al.).

  python -m surgseg.datasets.prepare_dsad --src nct-tso/dsad --out $DATA/processed
"""
from __future__ import annotations

import numpy as np

from surgseg.datasets.common import SampleWriter, base_argparser
from surgseg.taxonomy import IGNORE, load_taxonomy

SPLITS = {"train": "train", "validation": "val", "test": "test"}


def binary(pil_img) -> np.ndarray:
    return np.asarray(pil_img.convert("L")) > 127


def main() -> None:
    p = base_argparser(__doc__)
    p.add_argument("--configs", nargs="+", default=["single_organ", "multi_organ"])
    args = p.parse_args()
    from datasets import load_dataset   # heavy import, only needed here

    tax = load_taxonomy(args.taxonomy) if args.taxonomy else load_taxonomy()
    dmap = tax.dataset_maps["dsad"]
    w = SampleWriter(args.out, "dsad", args.max_side)
    n = 0
    for config in args.configs:
        for hf_split, split in SPLITS.items():
            ds = load_dataset(args.src, config, split=hf_split)
            for row in ds:
                img = np.asarray(row["image"].convert("RGB"))
                mask = np.zeros(img.shape[:2], np.uint8)
                if config == "single_organ":
                    organs = {row["organ"]: row["mask"]}
                else:
                    organs = {k[len("mask_"):]: v for k, v in row.items()
                              if k.startswith("mask_") and v is not None}
                labeled, cover = [], np.zeros(img.shape[:2], np.uint8)
                for organ, m in organs.items():
                    cid = tax.resolve(dmap[organ])          # KeyError = unknown organ
                    labeled.append(cid)
                    b = binary(m)
                    if b.shape != mask.shape:
                        raise ValueError(f"mask/image size mismatch for {organ}")
                    mask[b] = cid
                    cover += b
                mask[cover > 1] = IGNORE                     # overlapping annotations
                organ_tag = row.get("organ", "multi")
                uid = f"{config}_{organ_tag}_{row['surgery_id']}_{row['frame_id']}"
                w.add(uid, img, mask, group=f"s{row['surgery_id']}", split=split,
                      labeled=labeled)
                n += 1
                if args.limit and n >= args.limit:
                    w.close()
                    return
    w.close()


if __name__ == "__main__":
    main()
