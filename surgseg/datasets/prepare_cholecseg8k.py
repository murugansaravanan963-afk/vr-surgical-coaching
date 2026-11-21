"""CholecSeg8k (laparoscopic cholecystectomy, 17 Cholec80 videos, CC BY-NC-SA 4.0).

Expects the Kaggle layout:  <src>/videoXX/videoXX_NNNNN/frame_N_endo.png
                                                       frame_N_endo_watershed_mask.png

The watershed mask stores one grey value per class. Different write-ups quote those
values either as hex colour codes (#505050 -> 80) or as the raw digits (50), so the
encoding is auto-detected from the data; override with --value-style if needed.

  python -m surgseg.datasets.prepare_cholecseg8k --src $RAW/cholecseg8k --out $DATA/processed
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from surgseg.datasets.common import (SampleWriter, assign_split, base_argparser, read_gray,
                                     read_rgb)
from surgseg.taxonomy import IGNORE, load_taxonomy

CLASS_NAMES = ["black_background", "abdominal_wall", "liver", "gastrointestinal_tract", "fat",
               "grasper", "connective_tissue", "blood", "cystic_duct", "l_hook_electrocautery",
               "gallbladder", "hepatic_vein", "liver_ligament"]
_CODES = [0x50, 0x11, 0x21, 0x13, 0x12, 0x31, 0x23, 0x24, 0x25, 0x32, 0x22, 0x33, 0x05]
VALUE_STYLES = {
    "hex": {code: i for i, code in enumerate(_CODES)},                  # 80, 17, 33, ...
    "digits": {int(f"{code:x}"): i for i, code in enumerate(_CODES)},   # 50, 11, 21, ...
}
TOLERATED = {0, 255}   # stray values seen at annotation borders -> ignore

# Video-level split (17 videos). Change freely, but keep it at video level.
VAL_VIDEOS = {"video18", "video52"}
TEST_VIDEOS = {"video12", "video20", "video48", "video55"}


def detect_style(mask_paths: list[Path], n: int = 300) -> str:
    seen: set[int] = set()
    step = max(1, len(mask_paths) // n)
    for pth in mask_paths[::step]:
        seen.update(np.unique(read_gray(pth)).tolist())
    seen -= TOLERATED
    fits = [s for s, table in VALUE_STYLES.items() if seen <= set(table)]
    if len(fits) != 1:
        raise SystemExit(f"Cannot auto-detect CholecSeg8k mask encoding; values seen: "
                         f"{sorted(seen)}; matching styles: {fits}. Pass --value-style.")
    print(f"[cholecseg8k] watershed values {sorted(seen)} -> style '{fits[0]}'")
    return fits[0]


def main() -> None:
    p = base_argparser(__doc__)
    p.add_argument("--value-style", choices=list(VALUE_STYLES), default=None)
    args = p.parse_args()
    tax = load_taxonomy(args.taxonomy) if args.taxonomy else load_taxonomy()
    dmap = tax.dataset_maps["cholecseg8k"]
    labeled = tax.labeled_set("cholecseg8k")

    masks = sorted(Path(args.src).rglob("*_endo_watershed_mask.png"))
    if not masks:
        raise SystemExit(f"no *_endo_watershed_mask.png under {args.src}")
    style = args.value_style or detect_style(masks)
    lut = np.full(256, IGNORE, np.uint8)
    for raw, idx in VALUE_STYLES[style].items():
        lut[raw] = tax.resolve(dmap[CLASS_NAMES[idx]])

    w = SampleWriter(args.out, "cholecseg8k", args.max_side)
    for i, mpath in enumerate(masks):
        ipath = mpath.with_name(mpath.name.replace("_endo_watershed_mask.png", "_endo.png"))
        video = next((q for q in mpath.parts if re.fullmatch(r"video\d+", q)), None)
        if video is None:
            raise ValueError(f"cannot find videoXX folder in {mpath}")
        raw = read_gray(mpath)
        unknown = set(np.unique(raw).tolist()) - set(VALUE_STYLES[style]) - TOLERATED
        if unknown:
            raise ValueError(f"{mpath}: unexpected mask values {sorted(unknown)}")
        w.add(f"{mpath.parent.name}_{ipath.stem}", read_rgb(ipath), lut[raw], group=video,
              split=assign_split(video, VAL_VIDEOS, TEST_VIDEOS), labeled=labeled)
        if args.limit and i + 1 >= args.limit:
            break
    w.close()


if __name__ == "__main__":
    main()
