"""Latency of the exported model, the number that decides student size for the VR loop.

  python -m surgseg.tools.benchmark --model deploy/surgseg.onnx --sizes 384x480 512x640
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from surgseg.runtime.engine import SegEngine
from surgseg.runtime.roles import RoleAssigner
from surgseg.runtime.safety import SafetyMonitor, Tip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sizes", nargs="*", default=["384x480", "512x640", "768x960"])
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--device-id", type=int, default=None)
    args = ap.parse_args()
    frame = (np.random.rand(1024, 1280, 3) * 255).astype(np.uint8)
    for s in args.sizes:
        h, w = map(int, s.split("x"))
        eng = SegEngine(args.model, input_hw=(h, w), device_id=args.device_id)
        roles = RoleAssigner(eng.class_names, procedure="cholecystectomy")
        safety = SafetyMonitor()
        tips = [Tip("psm1", 0.4, 0.5), Tip("psm2", 0.6, 0.5)]
        for _ in range(10):
            eng(frame)
        tm, tt = [], []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            seg = eng(frame)
            t1 = time.perf_counter()
            ro = roles.assign(seg)
            safety.update(ro.role_map, seg.labels, eng.class_names, tips)
            t2 = time.perf_counter()
            tm.append(t1 - t0)
            tt.append(t2 - t0)
        tm, tt = np.array(tm) * 1000, np.array(tt) * 1000
        print(f"{h}x{w} [{eng.providers[0]}]  model p50 {np.median(tm):6.1f} ms  "
              f"p95 {np.percentile(tm, 95):6.1f} ms | full pipeline p50 {np.median(tt):6.1f} ms "
              f"-> {1000 / np.median(tt):5.1f} FPS")


if __name__ == "__main__":
    main()
