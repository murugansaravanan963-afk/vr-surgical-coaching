"""Export an engine-agnostic ONNX model + metadata for any VR simulator.

  python -m surgseg.export_onnx --ckpt runs/student/best.pt --out deploy/surgseg_b2

The graph is self-contained so every engine can call it the same way:
  input   image       float32 [N,3,H,W], RGB in [0,1]   (H, W multiples of 32)
  outputs probs       float32 [N,C,H,W]  calibrated softmax (temperature baked in)
          labels      uint8   [N,H,W]    argmax class id
          confidence  float32 [N,H,W]    max probability
Normalisation and calibration live INSIDE the graph, so Unity Sentis, Unreal NNE,
ONNX Runtime, TensorRT, etc. only need to hand over the rendered frame.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from surgseg.data import IMAGENET_MEAN, IMAGENET_STD
from surgseg.model import load_for_inference
from surgseg.taxonomy import load_taxonomy


class Deployable(torch.nn.Module):
    def __init__(self, model, temperature: float):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.inv_t = 1.0 / temperature

    def forward(self, image):
        logits = self.model((image - self.mean) / self.std) * self.inv_t
        probs = torch.softmax(logits, dim=1)
        conf, labels = probs.max(dim=1)
        return probs, labels.to(torch.uint8), conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True, help="output path prefix (no extension)")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--temperature", type=float, default=None,
                    help="default: read from <ckpt>.eval_test.json if present, else 1.0")
    ap.add_argument("--static", action="store_true",
                    help="fixed input size (some engines/TensorRT builds prefer this)")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    model, ck = load_for_inference(args.ckpt, "cpu")
    T = args.temperature
    if T is None:
        ev = Path(args.ckpt).with_suffix(".eval_test.json")
        T = json.loads(ev.read_text()).get("temperature", 1.0) if ev.exists() else 1.0
    dep = Deployable(model, T).eval()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.rand(1, 3, args.height, args.width)
    dyn = None if args.static else {
        "image": {0: "batch", 2: "height", 3: "width"},
        "probs": {0: "batch", 2: "height", 3: "width"},
        "labels": {0: "batch", 1: "height", 2: "width"},
        "confidence": {0: "batch", 1: "height", 2: "width"},
    }
    torch.onnx.export(dep, (dummy,), str(out.with_suffix(".onnx")), opset_version=args.opset,
                      input_names=["image"], output_names=["probs", "labels", "confidence"],
                      dynamic_axes=dyn, dynamo=False)

    # Verify the exported graph against PyTorch before anyone ships it.
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out.with_suffix(".onnx")), providers=["CPUExecutionProvider"])
    test = torch.rand(1, 3, args.height, args.width)
    with torch.no_grad():
        ref = dep(test)
    got = sess.run(None, {"image": test.numpy()})
    err = float(np.abs(got[0] - ref[0].numpy()).max())
    agree = float((got[1] == ref[1].numpy()).mean())
    print(f"ONNX check: max |dprob| = {err:.2e}, label agreement = {agree:.4%}")
    if err > 1e-3 or agree < 0.999:
        raise SystemExit("exported model does not match PyTorch")

    tax = load_taxonomy()
    meta = {
        "format": "surgseg-onnx-v1",
        "input": {"name": "image", "layout": "NCHW", "dtype": "float32", "range": [0, 1],
                  "channels": "RGB", "multiple_of": 32,
                  "recommended_hw": [args.height, args.width], "dynamic": not args.static},
        "outputs": ["probs", "labels", "confidence"],
        "classes": [{"id": i, "name": n, "color": tax.colors[i].tolist()}
                    for i, n in enumerate(ck["class_names"])],
        "temperature": T,
        "train_step": ck["step"],
        "model_preset": ck["cfg"]["model"]["preset"],
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out.with_suffix('.onnx')} and {out.with_suffix('.json')} (T={T:.3f})")


if __name__ == "__main__":
    main()
