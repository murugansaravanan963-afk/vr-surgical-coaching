"""Model construction and checkpoint I/O (segmentation_models_pytorch + surgical backbones)."""
from __future__ import annotations

import copy

import segmentation_models_pytorch as smp
import torch

from surgseg.backbones import SURGICAL_BACKBONES, load_into_smp_encoder, patch_activations

# "surgical" presets start from encoders self-supervised on surgical video (see
# backbones.py) - the recommended choice. The ImageNet presets remain as baselines.
PRESETS = {
    # --- recommended ---
    "lemonfm-upernet": dict(arch="upernet", encoder_name="tu-convnext_large",
                            surgical="lemonfm"),                         # ~230M, teacher
    "surgenetxl-fpn": dict(arch="fpn", encoder_name="tu-caformer_s18",
                           surgical="surgenetxl"),                       # ~28M, VR student
    "surgenetxl-upernet": dict(arch="upernet", encoder_name="tu-caformer_s18",
                               surgical="surgenetxl"),
    # --- ImageNet baselines ---
    "segformer-b0": dict(arch="segformer", encoder_name="mit_b0"),
    "segformer-b2": dict(arch="segformer", encoder_name="mit_b2"),
    "segformer-b5": dict(arch="segformer", encoder_name="mit_b5"),
    "upernet-convnext-l": dict(arch="upernet", encoder_name="tu-convnext_large"),
    "unet-effb4": dict(arch="unet", encoder_name="efficientnet-b4"),
}


def build_model(preset: str | None = None, num_classes: int = 22, pretrained: bool = True,
                **overrides) -> torch.nn.Module:
    cfg = copy.deepcopy(PRESETS[preset]) if preset else {}
    cfg.update(overrides)
    arch = cfg.pop("arch")
    surgical = cfg.pop("surgical", None)
    imagenet = "imagenet" if pretrained and not surgical else None
    model = smp.create_model(arch, classes=num_classes, encoder_weights=imagenet, **cfg)
    if surgical:
        # Architecture patch applies whether or not weights are loaded (checkpoints too).
        patch_activations(model.encoder, SURGICAL_BACKBONES[surgical]["format"])
        if pretrained:
            load_into_smp_encoder(model, surgical)
    return model


def save_checkpoint(path, model, optimizer, scaler_state, step, best, cfg, class_names,
                    ema=None):
    m = model.module if hasattr(model, "module") else model
    torch.save({"model": (ema.module if ema is not None else m).state_dict(),
                "raw_model": m.state_dict() if ema is not None else None,
                "ema": ema is not None,
                "optimizer": optimizer.state_dict() if optimizer else None,
                "scaler": scaler_state, "step": step, "best": best, "cfg": cfg,
                "class_names": class_names}, path)


def load_for_inference(path, device="cpu") -> tuple[torch.nn.Module, dict]:
    """Loads the EMA weights when the run used EMA (they are what was validated)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    model = build_model(cfg["model"]["preset"], num_classes=len(ck["class_names"]),
                        pretrained=False, **cfg["model"].get("overrides", {}))
    model.load_state_dict(ck["model"])
    return model.eval().to(device), ck
