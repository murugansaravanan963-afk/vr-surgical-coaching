"""SAR-RARP50 (da Vinci Si robot-assisted radical prostatectomy, suturing; CC BY-NC-SA 4.0).

Train: https://doi.org/10.5522/04/24932529   Test: https://doi.org/10.5522/04/24932499
(one zip per video; scripts/linux/download_data.sh fetches them via the figshare API)

The closest public match to a da Vinci VR trainer: real da Vinci instruments, needles
and thread in a suturing task. Masks are instrument/object only, so the partial-label
loss treats its background as "not a tool/needle/thread", never as "not an organ".

    <src>/video_XX[_k]/video_left.avi             60 fps
    <src>/video_XX[_k]/segmentation/NNNNNNNNN.png 1 Hz; name = frame index; values 0..9

  python -m surgseg.datasets.prepare_sarrarp50 --src $RAW/sarrarp50 --out $DATA/processed \
      --unlabeled-every 30
"""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

from surgseg.datasets.common import SampleWriter, base_argparser, read_gray
from surgseg.taxonomy import IGNORE, load_taxonomy

# Grey value -> class, from the dataset README.
CLASS_NAMES = ["background", "tool_clasper", "tool_wrist", "tool_shaft", "suturing_needle",
               "thread", "suction_tool", "needle_holder", "clamps", "catheter"]
TEST_VIDEOS = {f"video_{i}" for i in range(41, 51)}          # official challenge test set
VAL_VIDEOS = {"video_02", "video_13", "video_27", "video_36"}


def main() -> None:
    p = base_argparser(__doc__)
    p.add_argument("--unlabeled-every", type=int, default=0,
                   help="also store every Nth video frame of TRAIN videos as unlabeled (0=off)")
    p.add_argument("--unlabeled-only", action="store_true",
                   help="write only sarrarp50_unl (labeled sarrarp50 already converted)")
    args = p.parse_args()
    tax = load_taxonomy(args.taxonomy) if args.taxonomy else load_taxonomy()
    dmap = tax.dataset_maps["sarrarp50"]
    lut = np.full(256, IGNORE, np.uint8)
    for v, name in enumerate(CLASS_NAMES):
        lut[v] = tax.resolve(dmap[name])
    labeled = tax.labeled_set("sarrarp50")

    vdirs = sorted({q.parent for q in Path(args.src).rglob("video_left.avi")})
    if not vdirs:
        raise SystemExit(f"no */video_left.avi under {args.src}")
    w = None if args.unlabeled_only else SampleWriter(args.out, "sarrarp50", args.max_side)
    u = SampleWriter(args.out, "sarrarp50_unl", args.max_side) if args.unlabeled_every else None
    n = 0
    for vdir in vdirs:
        group = re.match(r"(video_\d+)", vdir.name).group(1)    # video_11_1/_2 stay together
        split = "test" if group in TEST_VIDEOS else "val" if group in VAL_VIDEOS else "train"
        masks = {int(q.stem): q for q in (vdir / "segmentation").glob("*.png")}
        unl_every = args.unlabeled_every if split == "train" else 0
        if w is None and not unl_every:
            continue                     # --unlabeled-only: nothing to take from this video
        cap = cv2.VideoCapture(str(vdir / "video_left.avi"))
        idx, last = 0, max(masks) if masks else -1
        while True:
            want_mask = idx in masks and w is not None
            # Unlabeled frames sit half-way between the 1 Hz labels (frames 30, 90, ...).
            # Labels are every 60 frames, so an offset that is a multiple of 60 would put
            # every unlabeled pick ON a labeled frame.
            want_unl = (unl_every and idx % unl_every == 30 % unl_every and idx % 60 != 0
                        and idx not in masks)
            if (idx > last or w is None) and not unl_every:
                break
            ok = cap.grab()
            if not ok:
                break
            if want_mask or want_unl:
                ok, bgr = cap.retrieve()
                if not ok:
                    raise IOError(f"{vdir}: cannot decode frame {idx}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if want_mask:
                    raw = read_gray(masks[idx])
                    if raw.max() >= len(CLASS_NAMES):
                        raise ValueError(f"{masks[idx]}: value {raw.max()} > 9")
                    if raw.shape != rgb.shape[:2]:
                        raise ValueError(f"{masks[idx]}: mask {raw.shape} vs frame {rgb.shape}")
                    w.add(f"{vdir.name}_{idx:09d}", rgb, lut[raw], group=group, split=split,
                          labeled=labeled)
                    n += 1
                else:
                    u.add(f"{vdir.name}_{idx:09d}", rgb, None, group=group, split=split,
                          labeled=[])
            idx += 1
        cap.release()
        missing = [i for i in masks if i >= idx]
        if missing and w is not None:
            raise IOError(f"{vdir}: video ended at frame {idx}, masks up to {max(missing)}")
        if args.limit and n >= args.limit:
            break
    if w is not None:
        w.close()
    if u is not None:
        u.close()


if __name__ == "__main__":
    main()
