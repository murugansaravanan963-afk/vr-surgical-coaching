"""Engine-agnostic sidecar: any simulator talks to the model over ZeroMQ.

  python -m surgseg.runtime.server --model deploy/surgseg_b2.onnx --port 5555

Why a sidecar: Unity (C#), Unreal (C++), Isaac Sim / ORBIT-Surgical (Python),
SurRoL (PyBullet) and dVRK (ROS) all have ZeroMQ bindings, so one process serves
every simulator, and it runs on the GPU with TensorRT/CUDA regardless of engine.
(If you prefer in-engine inference, load the same .onnx in Unity Sentis / Unreal NNE
and port roles.py + safety.py - they are ~300 lines of plain array code.)

Protocol (REQ/REP, multipart): request = [json header, optional payload bytes...]
  {"cmd": "infer", "frame_id": 17, "timestamp_ms": <sender unix ms>,
   "encoding": "raw_rgb8" | "jpeg" | "png", "width": 1280, "height": 1024,
   "flip_y": false,                      # true for bottom-up buffers (Unity ReadPixels)
   "tips": [{"id": "psm1", "u": 0.41, "v": 0.63, "xyz": [x, y, z]}],   # xyz optional
   "anchors": [{"u": 0.5, "v": 0.4, "role": "danger"}],  # user marks, re-projected each frame
   "depth": {"width": W, "height": H, "fx":..,"fy":..,"cx":..,"cy":..},  # optional;
                                         #   float32 depth (metres) as 2nd payload frame
   "return": ["role_rgba"]}              # optional extras, in this order:
                                         #   "role_map" (role ids), "class_map" (class ids),
                                         #   "role_rgba" (transparent coloured overlay PNG)
  -> [json reply, png(role_map)?, png(class_map)?]
  {"cmd": "set_mode", "mode": "auto" | "manual", "procedure": "cholecystectomy", "step": null}
  {"cmd": "select", "u": 0.5, "v": 0.4, "role": "target" | "danger" | "neutral",
   "scope": "class" | "region"}          # applies to the most recent frame
  {"cmd": "clear"}   {"cmd": "info"}   {"cmd": "ping"}
"""
from __future__ import annotations

import argparse
import json
import time

import cv2
import numpy as np
import zmq

from surgseg.runtime.engine import SegEngine
from surgseg.runtime.roles import Role, RoleAssigner
from surgseg.runtime.safety import Intrinsics, SafetyConfig, SafetyMonitor, Tip

# Role -> BGRA for the "role_rgba" overlay: neutral/instrument transparent, target
# translucent green, danger translucent red (OpenCV writes PNGs in BGRA order).
ROLE_BGRA = np.zeros((256, 4), np.uint8)
ROLE_BGRA[int(Role.TARGET)] = (0, 200, 0, 110)
ROLE_BGRA[int(Role.DANGER)] = (0, 0, 255, 150)
ROLE_BGRA[int(Role.INSTRUMENT)] = (255, 140, 0, 70)   # faint blue: tools recognised


class SurgSegService:
    def __init__(self, engine: SegEngine, mode="auto", procedure="default",
                 safety: SafetyConfig | None = None):
        self.engine = engine
        self.roles = RoleAssigner(engine.class_names, mode=mode, procedure=procedure)
        self.safety = SafetyMonitor(safety or SafetyConfig())
        self.last_seg = None

    def handle(self, header: dict, payload: list[bytes]) -> list[bytes]:
        cmd = header.get("cmd")
        if cmd == "infer":
            return self.infer(header, payload)
        if cmd == "set_mode":
            self.roles.set_mode(header["mode"], header.get("procedure"), header.get("step"))
            return self._reply(ok=True, mode=self.roles.mode, procedure=self.roles.procedure,
                               step=self.roles.step)
        if cmd == "select":
            if self.last_seg is None:
                return self._reply(ok=False, reason="no frame yet")
            res = self.roles.select(self.last_seg, header["u"], header["v"], header["role"],
                                    header.get("scope", "class"))
            return self._reply(**res)
        if cmd == "clear":
            self.roles.clear()
            return self._reply(ok=True)
        if cmd == "info":
            return self._reply(ok=True, classes=self.engine.meta["classes"],
                               roles={r.name.lower(): int(r) for r in Role},
                               procedures={k: {"description": p.description,
                                               "steps": list(p.steps)}
                                           for k, p in self.roles.profiles.items()},
                               mode=self.roles.mode, procedure=self.roles.procedure,
                               input_hw=list(self.engine.input_hw),
                               providers=self.engine.providers)
        if cmd == "ping":
            return self._reply(ok=True, t_ms=time.time() * 1000)
        return self._reply(ok=False, reason=f"unknown cmd {cmd!r}")

    def infer(self, h: dict, payload: list[bytes]) -> list[bytes]:
        t0 = time.perf_counter()
        # Age on ARRIVAL (transport + queueing). Measuring after inference would make every
        # frame "stale" on a slow GPU even though the guidance matches the frame it came from.
        age = time.time() * 1000 - h["timestamp_ms"] if "timestamp_ms" in h else 0.0
        frame = decode_image(h, payload[0])
        seg = self.engine(frame)
        self.last_seg = seg
        t_model = time.perf_counter()
        ro = self.roles.assign(seg, h.get("anchors"), frame)   # user marks (manual mode)
        depth = K = None
        if "depth" in h and len(payload) > 1:
            d = h["depth"]
            depth = np.frombuffer(payload[1], np.float32).reshape(d["height"], d["width"])
            if h.get("flip_y"):
                depth = depth[::-1]
            K = Intrinsics(d["fx"], d["fy"], d["cx"], d["cy"], h["width"], h["height"])
        tips = [Tip(t["id"], t["u"], t["v"], tuple(t["xyz"]) if t.get("xyz") else None)
                for t in h.get("tips", [])]
        tip_status = self.safety.update(ro.role_map, seg.labels, self.engine.class_names, tips,
                                        age, depth, K)
        t1 = time.perf_counter()
        extras = []
        want = h.get("return", [])
        if "role_map" in want:
            extras.append(cv2.imencode(".png", ro.role_map)[1].tobytes())
        if "class_map" in want:
            extras.append(cv2.imencode(".png", seg.labels)[1].tobytes())
        if "role_rgba" in want:   # ready-to-display overlay: engines just draw it on top
            extras.append(cv2.imencode(".png", ROLE_BGRA[ro.role_map])[1].tobytes())
        return self._reply(
            ok=True, frame_id=h.get("frame_id"), mode=self.roles.mode,
            procedure=self.roles.procedure, step=self.roles.step,
            map_hw=list(seg.hw), tips=tip_status, regions=ro.regions,
            class_roles={self.engine.class_names[c]: r.name.lower()
                         for c, r in ro.class_roles.items()},
            timing_ms={"model": round((t_model - t0) * 1000, 2),
                       "total": round((t1 - t0) * 1000, 2), "frame_age": round(age, 1)},
            extras=[w for w in ("role_map", "class_map", "role_rgba") if w in want],
            _extras=extras)

    @staticmethod
    def _reply(_extras=(), **kw) -> list[bytes]:
        return [json.dumps(kw).encode()] + list(_extras)


def decode_image(h: dict, data: bytes) -> np.ndarray:
    enc = h.get("encoding", "raw_rgb8")
    if enc == "raw_rgb8":
        img = np.frombuffer(data, np.uint8).reshape(h["height"], h["width"], 3)
    elif enc == "raw_rgba8":
        img = np.frombuffer(data, np.uint8).reshape(h["height"], h["width"], 4)[..., :3]
    elif enc in ("jpeg", "png"):
        img = cv2.cvtColor(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR),
                           cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"unknown encoding {enc}")
    if h.get("flip_y"):
        img = img[::-1]
    return np.ascontiguousarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to the exported .onnx")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--bind", default="127.0.0.1",
                    help="use 0.0.0.0 only if the simulator runs on another machine")
    ap.add_argument("--mode", default="auto", choices=["auto", "manual"])
    ap.add_argument("--procedure", default="default")
    ap.add_argument("--input-hw", type=int, nargs=2, default=None)
    ap.add_argument("--device-id", type=int, default=None, help="GPU index (laptops: often 1)")
    ap.add_argument("--max-frame-age-ms", type=float, default=1500.0,
                    help="frames older than this on arrival are reported as 'stale'")
    args = ap.parse_args()

    engine = SegEngine(args.model, input_hw=args.input_hw, device_id=args.device_id)
    svc = SurgSegService(engine, args.mode, args.procedure,
                         SafetyConfig(max_frame_age_ms=args.max_frame_age_ms))
    sock = zmq.Context.instance().socket(zmq.REP)
    sock.bind(f"tcp://{args.bind}:{args.port}")
    print(f"surgseg server on tcp://{args.bind}:{args.port}  providers={engine.providers}  "
          f"input={engine.input_hw}  mode={args.mode}/{args.procedure}", flush=True)
    while True:
        parts = sock.recv_multipart()
        try:
            reply = svc.handle(json.loads(parts[0]), parts[1:])
        except Exception as e:   # never leave a REQ client hanging
            reply = SurgSegService._reply(ok=False, reason=f"{type(e).__name__}: {e}")
        sock.send_multipart(reply)


if __name__ == "__main__":
    main()
