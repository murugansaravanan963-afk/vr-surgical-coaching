"""Endoscapes2023 (laparoscopic cholecystectomy, CAMMA; CC BY-NC-SA 4.0).

Public zip: https://s3.unistra.fr/camma_public/datasets/endoscapes/endoscapes.zip (6.3 GB)

Why it matters here: it labels the cystic ARTERY and cystic DUCT, the structures whose
injury is the classic cholecystectomy complication, i.e. exactly our DANGER classes.

    <src>/seg_label_map.txt          class names, line order = mask value
    <src>/semseg/<vid>_<frame>.png   493 masks (3 identical channels)
    <src>/all/<vid>_<frame>.jpg      58k frames at 1 fps -> optional UNLABELED frames
    <src>/{val,test}_seg_vids.txt, {val,test}_vids.txt   video ids (stored as floats)

  python -m surgseg.datasets.prepare_endoscapes --src $RAW/endoscapes --out $DATA/processed \
      --unlabeled-stride 5
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from surgseg.datasets.common import SampleWriter, base_argparser, read_gray, read_rgb
from surgseg.taxonomy import IGNORE, load_taxonomy


def read_vids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    for tok in path.read_text().split():
        try:
            out.add(str(int(float(tok))))          # "1.37e+02" -> "137"
        except ValueError:
            pass                                   # symlink stubs like "25/train_vids_2.txt"
    return out


def main() -> None:
    p = base_argparser(__doc__)
    p.add_argument("--unlabeled-stride", type=int, default=0,
                   help="also add every Nth 1-fps frame of TRAIN videos as unlabeled (0=off)")
    args = p.parse_args()
    src = Path(args.src)
    if not (src / "seg_label_map.txt").exists() and (src / "endoscapes").is_dir():
        src = src / "endoscapes"
    tax = load_taxonomy(args.taxonomy) if args.taxonomy else load_taxonomy()
    dmap = tax.dataset_maps["endoscapes"]
    names = [l.strip() for l in (src / "seg_label_map.txt").read_text().splitlines() if l.strip()]
    lut = np.full(256, IGNORE, np.uint8)
    for i, n in enumerate(names):
        lut[i] = tax.resolve(dmap[n])                # KeyError = unexpected class name
    labeled = tax.labeled_set("endoscapes")

    val = read_vids(src / "val_seg_vids.txt")
    test = read_vids(src / "test_seg_vids.txt")
    held_out = val | test | read_vids(src / "val_vids.txt") | read_vids(src / "test_vids.txt")
    if not val or not test:
        raise SystemExit("could not read val/test video lists")
    images = {q.stem: q for q in (src / "all").glob("*.jpg")}

    w = SampleWriter(args.out, "endoscapes", args.max_side)
    n = 0
    # Values outside seg_label_map.txt (seen: 255 = not annotated, and a stray 7) become
    # IGNORE via the lut. They are counted, and a large share aborts: that would mean
    # the label map does not describe these masks.
    known = set(range(len(names)))
    unknown: Counter = Counter()
    total = 0
    for mpath in sorted((src / "semseg").glob("*.png")):
        vid = mpath.stem.split("_")[0]
        raw = read_gray(mpath)
        vals, cnts = np.unique(raw, return_counts=True)
        for v, c in zip(vals.tolist(), cnts.tolist()):
            if v not in known:
                unknown[v] += c
        total += raw.size
        split = "test" if vid in test else "val" if vid in val else "train"
        w.add(f"seg_{mpath.stem}", read_rgb(images[mpath.stem]), lut[raw], group=vid,
              split=split, labeled=labeled)
        n += 1
        if args.limit and n >= args.limit:
            break
    share = {v: f"{100 * c / max(total, 1):.3f}%" for v, c in sorted(unknown.items())}
    print(f"[endoscapes] values not in seg_label_map.txt -> ignore: {share or 'none'}")
    if sum(c for v, c in unknown.items() if v != IGNORE) > 0.02 * total:
        raise SystemExit("too many undocumented mask values - label map does not fit; stopping")
    w.close()
    if args.unlabeled_stride:
        u = SampleWriter(args.out, "endoscapes_unl", args.max_side)
        k = 0
        for stem, ipath in sorted(images.items()):
            vid = stem.split("_")[0]
            if vid in held_out:
                continue
            k += 1
            if k % args.unlabeled_stride == 0:
                u.add(f"unl_{stem}", read_rgb(ipath), None, group=vid, split="train", labeled=[])
        u.close()


if __name__ == "__main__":
    main()
