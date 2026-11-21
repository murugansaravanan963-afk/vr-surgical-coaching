"""Frames rendered by YOUR simulator, with ground-truth masks exported by the engine.

Every VR engine can render a per-pixel object/semantic ID buffer for free (Isaac Sim
Replicator "semantic_segmentation", Unity Perception, Unreal custom-depth stencil,
PyBullet/SurRoL getCameraImage segmentation). A few hundred such frames, fine-tuned on
top of the real-data model, close most of the sim-vs-real appearance gap.

    <src>/<episode>/images/*.png
    <src>/<episode>/masks/*.png        integer object/semantic id per pixel (same name)

--label-map is a YAML mapping each simulator id to a unified class name, e.g.
    0: background
    3: liver
    7: gallbladder
    12: instrument
Ids not in the map become "background" (anything unmapped is treated as "none of the
labeled classes"); use "ignore" for ids that should not be supervised.

  python -m surgseg.datasets.prepare_custom --src $RAW/sim --label-map sim_labels.yaml \
      --out $DATA/processed --name sim --val-episodes ep07 ep08
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from surgseg.datasets.common import (SampleWriter, assign_split, base_argparser, read_gray,
                                     read_rgb)
from surgseg.taxonomy import GROUP_BASE, load_taxonomy


def main() -> None:
    p = base_argparser(__doc__)
    p.add_argument("--label-map", required=True)
    p.add_argument("--name", default="sim")
    p.add_argument("--val-episodes", nargs="*", default=[])
    p.add_argument("--test-episodes", nargs="*", default=[])
    args = p.parse_args()
    tax = load_taxonomy(args.taxonomy) if args.taxonomy else load_taxonomy()
    raw_map = yaml.safe_load(Path(args.label_map).read_text())
    lut = np.zeros(65536, np.uint8)                 # sim ids may be 16-bit
    for sim_id, name in raw_map.items():
        lut[int(sim_id)] = tax.resolve(name)
    labeled = sorted({int(v) for v in lut if 0 < v < GROUP_BASE})

    w = SampleWriter(args.out, args.name, args.max_side)
    n = 0
    for mdir in sorted(Path(args.src).rglob("masks")):
        episode = mdir.parent.name
        split = assign_split(episode, set(args.val_episodes), set(args.test_episodes))
        for mpath in sorted(mdir.glob("*.png")):
            mask = lut[read_gray(mpath).astype(np.int64)]
            img = read_rgb(mdir.parent / "images" / mpath.name)
            w.add(f"{episode}_{mpath.stem}", img, mask, group=episode, split=split,
                  labeled=labeled, synthetic=True)
            n += 1
            if args.limit and n >= args.limit:
                w.close()
                return
    w.close()


if __name__ == "__main__":
    main()
