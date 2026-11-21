"""Run the exported model on a folder of images (e.g. Unity screenshots) and save overlays.

  python -m surgseg.tools.demo_images --model models/surgseg.onnx --images shots/ --out demo_out/

For each image writes <name>_demo.png: original | classes coloured | target(green)/danger(red),
using the AUTO-mode procedure profile given by --procedure.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from surgseg.runtime.engine import SegEngine
from surgseg.runtime.roles import RoleAssigner, colorize_roles

EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def legend(names, colors, present, width):
    rows = [(n, c) for i, (n, c) in enumerate(zip(names, colors)) if i in present and i > 0]
    h = 22 * max(1, len(rows)) + 8
    img = np.full((h, width, 3), 30, np.uint8)
    for k, (n, c) in enumerate(rows):
        y = 8 + 22 * k
        cv2.rectangle(img, (8, y), (26, y + 16), tuple(int(v) for v in c[::-1]), -1)
        cv2.putText(img, n, (34, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="demo_out")
    ap.add_argument("--procedure", default="cholecystectomy")
    ap.add_argument("--device-id", type=int, default=None)
    args = ap.parse_args()
    eng = SegEngine(args.model, device_id=args.device_id)
    roles = RoleAssigner(eng.class_names, mode="auto", procedure=args.procedure)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in EXT)
    print(f"model input {eng.input_hw}, provider {eng.providers[0]}, {len(files)} images")
    for p in files:
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        seg = eng(rgb)
        ro = roles.assign(seg)
        h, w = rgb.shape[:2]
        up = lambda a: cv2.resize(a, (w, h), interpolation=cv2.INTER_NEAREST)
        cls = up(eng.colorize(seg.labels))
        rol = up(colorize_roles(ro.role_map))
        rmask = up(ro.role_map)[..., None]
        blend_c = (0.5 * rgb + 0.5 * cls).astype(np.uint8)
        blend_r = np.where((rmask == 1) | (rmask == 2), (0.45 * rgb + 0.55 * rol), rgb).astype(np.uint8)
        row = np.hstack([rgb, blend_c, blend_r])
        present = set(np.unique(seg.labels).tolist())
        panel = np.vstack([row, legend(eng.class_names, eng.colors, present, row.shape[1])])
        cv2.imwrite(str(out / f"{p.stem}_demo.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        found = ", ".join(f"{r['role']}:{r['class']}" for r in ro.regions) or "no target/danger regions"
        print(f"  {p.name}: {found}")
    print(f"saved to {out.resolve()}")


if __name__ == "__main__":
    main()
