"""ONNX inference wrapper. Depends only on numpy, opencv and onnxruntime (no torch)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

PREFERRED_PROVIDERS = ["TensorrtExecutionProvider", "CUDAExecutionProvider",
                       "DmlExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]


@dataclass
class SegResult:
    probs: np.ndarray        # (C, h, w) float32, calibrated
    labels: np.ndarray       # (h, w) uint8
    confidence: np.ndarray   # (h, w) float32
    frame_hw: tuple[int, int]

    @property
    def hw(self) -> tuple[int, int]:
        return self.labels.shape

    def to_px(self, u: float, v: float) -> tuple[int, int]:
        """Normalised [0,1] image coords -> (row, col) at model resolution."""
        h, w = self.hw
        return (int(np.clip(round(v * h - 0.5), 0, h - 1)),
                int(np.clip(round(u * w - 0.5), 0, w - 1)))


class SegEngine:
    def __init__(self, onnx_path: str | Path, input_hw: tuple[int, int] | None = None,
                 providers: list | None = None, device_id: int | None = None):
        """device_id picks the GPU for CUDA/TensorRT/DirectML. On laptops DirectML's device 0
        is often the integrated Intel/AMD chip and the discrete GPU is device 1."""
        onnx_path = Path(onnx_path)
        self.meta = json.loads(onnx_path.with_suffix(".json").read_text())
        self.class_names = [c["name"] for c in self.meta["classes"]]
        self.colors = np.array([c["color"] for c in self.meta["classes"]], np.uint8)
        avail = ort.get_available_providers()
        providers = providers or [p for p in PREFERRED_PROVIDERS if p in avail]
        if device_id is not None:
            gpu_eps = {"TensorrtExecutionProvider", "CUDAExecutionProvider", "DmlExecutionProvider"}
            providers = [(p, {"device_id": device_id}) if p in gpu_eps else p for p in providers]
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.providers = self.session.get_providers()
        self.input_hw = tuple(input_hw or self.meta["input"]["recommended_hw"])
        assert all(s % 32 == 0 for s in self.input_hw), "input size must be multiples of 32"

    def __call__(self, frame_rgb: np.ndarray) -> SegResult:
        h, w = self.input_hw
        x = cv2.resize(frame_rgb, (w, h), interpolation=cv2.INTER_AREA)
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        probs, labels, conf = self.session.run(None, {"image": x})
        return SegResult(probs[0], labels[0], conf[0], frame_rgb.shape[:2])

    def colorize(self, labels: np.ndarray) -> np.ndarray:
        return self.colors[labels]
