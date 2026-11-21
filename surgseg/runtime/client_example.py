"""Minimal simulator-side client. Copy this pattern into your sim's Python loop
(Isaac Sim / ORBIT-Surgical, SurRoL, a dVRK ROS node, ...).

  python -m surgseg.runtime.client_example --image some_frame.png --procedure cholecystectomy
"""
from __future__ import annotations

import argparse
import json
import time

import cv2
import numpy as np
import zmq


class SurgSegClient:
    def __init__(self, address="tcp://127.0.0.1:5555", timeout_ms=1000):
        self.address, self.timeout_ms = address, timeout_ms
        self.ctx = zmq.Context.instance()
        self._connect()

    def _connect(self):
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(self.address)

    def call(self, header: dict, *payload: bytes):
        self.sock.send_multipart([json.dumps(header).encode(), *payload])
        try:
            parts = self.sock.recv_multipart()
        except zmq.Again:
            # REQ sockets are stuck after a timeout; recreate. Treat as "no guidance".
            self.sock.close()
            self._connect()
            return {"ok": False, "reason": "timeout"}, []
        return json.loads(parts[0]), parts[1:]

    def infer(self, rgb: np.ndarray, tips=(), frame_id=0, role_map=True, depth=None, K=None):
        h, w = rgb.shape[:2]
        hdr = {"cmd": "infer", "frame_id": frame_id, "timestamp_ms": time.time() * 1000,
               "encoding": "raw_rgb8", "width": w, "height": h, "tips": list(tips),
               "return": ["role_map"] if role_map else []}
        payload = [np.ascontiguousarray(rgb, np.uint8).tobytes()]
        if depth is not None:
            hdr["depth"] = {"width": depth.shape[1], "height": depth.shape[0], **K}
            payload.append(np.ascontiguousarray(depth, np.float32).tobytes())
        reply, extras = self.call(hdr, *payload)
        rmap = (cv2.imdecode(np.frombuffer(extras[0], np.uint8), cv2.IMREAD_UNCHANGED)
                if extras else None)
        return reply, rmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--address", default="tcp://127.0.0.1:5555")
    ap.add_argument("--procedure", default="default")
    args = ap.parse_args()
    cli = SurgSegClient(args.address, timeout_ms=10000)
    print(cli.call({"cmd": "set_mode", "mode": "auto", "procedure": args.procedure})[0])
    rgb = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    tips = [{"id": "psm1", "u": 0.45, "v": 0.55}, {"id": "psm2", "u": 0.6, "v": 0.5}]
    reply, rmap = cli.infer(rgb, tips)
    print(json.dumps(reply, indent=1))
    # Mode 1 example: user points at the centre of the view and marks it as danger.
    print(cli.call({"cmd": "select", "u": 0.5, "v": 0.5, "role": "danger", "scope": "region"})[0])


if __name__ == "__main__":
    main()
