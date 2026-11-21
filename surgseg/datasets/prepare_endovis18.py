"""EndoVis 2018 Robotic Scene Segmentation (da Vinci, porcine nephrectomy).

Download from the grand-challenge page (registration required). Expected layout,
searched recursively so train/test releases can sit side by side:

    <src>/.../seq_N/left_frames/frameXXX.png
    <src>/.../seq_N/labels/frameXXX.png         RGB colour-coded classes
    labels.json   (in seq_N/ or any parent)      [{"name", "color": [r,g,b], "classid"}]

Class names come from labels.json and are mapped through configs/taxonomy.yaml, so
the converter fails loudly if the release uses a name the taxonomy doesn't know.

  python -m surgseg.datasets.prepare_endovis18 --src $RAW/endovis18 --out $DATA/processed
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from surgseg.datasets.common import SampleWriter, base_argparser, read_rgb
from surgseg.taxonomy import IGNORE, load_taxonomy

VAL_SEQS = {"seq_2", "seq_5", "seq_9", "seq_15"}   # common validation choice in the literature
MAX_UNKNOWN_FRACTION = 0.005


def find_labels_json(seq_dir: Path, src: Path) -> Path:
    d = seq_dir
    while True:
        if (d / "labels.json").exists():
            return d / "labels.json"
        if d == src or d.parent == d:
            raise FileNotFoundError(f"no labels.json for {seq_dir}")
        d = d.parent


def main() -> None:
    p = base_argparser(__doc__)
    p.add_argument("--val-seqs", nargs="*", default=sorted(VAL_SEQS))
    args = p.parse_args()
    src = Path(args.src).resolve()
    tax = load_taxonomy(args.taxonomy) if args.taxonomy else load_taxonomy()
    dmap = tax.dataset_maps["endovis18"]
    labeled = tax.labeled_set("endovis18")

    label_dirs = sorted(d for d in src.rglob("labels") if d.is_dir() and any(d.glob("*.png")))
    if not label_dirs:
        raise SystemExit(f"no seq_*/labels/ folders under {src}")
    w = SampleWriter(args.out, "endovis18", args.max_side)
    n = 0
    for ldir in label_dirs:
        seq_dir = ldir.parent
        frames_dir = next((seq_dir / c for c in ("left_frames", "images", "frames")
                           if (seq_dir / c).is_dir()), None)
        if frames_dir is None:
            raise FileNotFoundError(f"no left_frames/ next to {ldir}")
        classes = json.loads(find_labels_json(seq_dir, src).read_text())
        colour_to_value = {}
        for c in classes:
            name = c["name"].strip().lower()
            if name not in dmap:
                raise KeyError(f"EndoVis18 class '{name}' missing from taxonomy.yaml")
            r, g, b = c["color"]
            colour_to_value[(r << 16) | (g << 8) | b] = tax.resolve(dmap[name])

        rel = seq_dir.relative_to(src).as_posix()
        is_test = "test" in rel.lower()
        split = "test" if is_test else "val" if seq_dir.name in args.val_seqs else "train"
        for lpath in sorted(ldir.glob("*.png")):
            rgb = read_rgb(lpath).astype(np.int32)
            key = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
            mask = np.full(key.shape, IGNORE, np.uint8)
            for k, v in colour_to_value.items():
                mask[key == k] = v
            unknown = (mask == IGNORE).mean()
            if unknown > MAX_UNKNOWN_FRACTION:
                cols = np.unique(key[mask == IGNORE])[:10]
                raise ValueError(f"{lpath}: {unknown:.1%} pixels have colours not in "
                                 f"labels.json (e.g. {[hex(c) for c in cols]})")
            ipath = frames_dir / lpath.name
            w.add(f"{rel}_{lpath.stem}", read_rgb(ipath), mask, group=rel, split=split,
                  labeled=labeled)
            n += 1
            if args.limit and n >= args.limit:
                w.close()
                return
    w.close()


if __name__ == "__main__":
    main()
