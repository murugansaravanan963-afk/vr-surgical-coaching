"""EndoVis 2017 Robotic Instrument Segmentation (da Vinci; instruments only).

Only the "instrument" class is labeled; with the partial-label loss this still helps,
because it teaches tool appearance without claiming anything about the anatomy.

    <src>/.../instrument_dataset_N/left_frames/frameXXX.png
    <src>/.../instrument_dataset_N/ground_truth/<Tool>_labels/frameXXX.png

  python -m surgseg.datasets.prepare_endovis17 --src $RAW/endovis17 --out $DATA/processed
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from surgseg.datasets.common import SampleWriter, base_argparser, read_gray, read_rgb
from surgseg.taxonomy import load_taxonomy

# Raw frames are 1920x1080 with the 1280x1024 endoscope image at this offset.
CROP_Y, CROP_X, CROP_H, CROP_W = 28, 320, 1024, 1280
VAL_SETS = {"instrument_dataset_8"}


def crop(a: np.ndarray) -> np.ndarray:
    if a.shape[:2] == (1080, 1920):
        return a[CROP_Y:CROP_Y + CROP_H, CROP_X:CROP_X + CROP_W]
    return a


def main() -> None:
    p = base_argparser(__doc__)
    p.add_argument("--val-sets", nargs="*", default=sorted(VAL_SETS))
    args = p.parse_args()
    src = Path(args.src).resolve()
    tax = load_taxonomy(args.taxonomy) if args.taxonomy else load_taxonomy()
    inst = tax.resolve(tax.dataset_maps["endovis17"]["instrument"])
    labeled = tax.labeled_set("endovis17")

    gt_dirs = sorted(d for d in src.rglob("ground_truth") if d.is_dir())
    if not gt_dirs:
        raise SystemExit(f"no instrument_dataset_*/ground_truth/ under {src}")
    w = SampleWriter(args.out, "endovis17", args.max_side)
    n = 0
    for gt in gt_dirs:
        ds_dir = gt.parent
        rel = ds_dir.relative_to(src).as_posix()
        split = ("test" if "test" in rel.lower()
                 else "val" if ds_dir.name in args.val_sets else "train")
        tool_dirs = [d for d in gt.iterdir() if d.is_dir()]
        for fpath in sorted((ds_dir / "left_frames").glob("*.png")):
            mask = None
            for td in tool_dirs:
                if (td / fpath.name).exists():
                    m = read_gray(td / fpath.name) > 0
                    mask = m if mask is None else mask | m
            if mask is None:
                continue   # frame without ground truth
            out = np.where(mask, inst, 0).astype(np.uint8)
            w.add(f"{rel}_{fpath.stem}", crop(read_rgb(fpath)), crop(out), group=rel,
                  split=split, labeled=labeled)
            n += 1
            if args.limit and n >= args.limit:
                w.close()
                return
    w.close()


if __name__ == "__main__":
    main()
