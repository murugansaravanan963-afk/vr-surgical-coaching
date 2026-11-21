"""Surgical foundation-model backbones for the smp encoders.

ImageNet features know cats and cars, not fascia and ducts. These encoders were
self-supervised on large amounts of surgical video and are the largest single accuracy
lever available. Both reports compare against ImageNet initialisation:

  lemonfm     ConvNeXt-Large, DINO-style on LEMON (938 h, 4k videos).  Apache-2.0.
              Che et al. 2025, arXiv 2503.19740 - CholecSeg8k mDice 81.3 vs 69.0 SurgeNetXL.
              -> teacher backbone
  surgenetxl  CaFormer-S18 (26M params), DINO on 4.7M surgical frames.  CC-BY-NC-4.0.
              Jaspers et al. 2025, MedIA / arXiv 2501.09436.
              -> real-time student backbone

Checkpoints come in their authors' formats; this module converts them to the timm
layout that segmentation_models_pytorch's "tu-*" encoders use. Conversions are covered
by tests/test_backbones.py (numerical equivalence, not just "keys loaded").
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import timm
import torch
import torch.nn as nn

SURGICAL_BACKBONES = {
    "lemonfm": {
        "timm": "convnext_large",
        "url": "https://huggingface.co/visurg/LemonFM/resolve/main/lemonfm.pth",
        "format": "torchvision_convnext_dino",
        "license": "Apache-2.0",
    },
    "surgenetxl": {
        "timm": "caformer_s18",
        "url": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/"
               "SurgeNetXL_checkpoint_epoch0050_teacher.pth",
        "format": "metaformer_relu",
        "license": "CC-BY-NC-4.0",
    },
}


def cache_dir() -> Path:
    d = Path(os.environ.get("SURGSEG_WEIGHTS", Path.home() / ".cache" / "surgseg"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def download(name: str) -> Path:
    spec = SURGICAL_BACKBONES[name]
    dst = cache_dir() / Path(spec["url"]).name
    if not dst.exists():
        print(f"downloading {name} weights -> {dst}")
        torch.hub.download_url_to_file(spec["url"], str(dst), progress=True)
    return dst


def _safe_load(path: Path):
    import argparse as _ap
    torch.serialization.add_safe_globals([_ap.Namespace])   # DINO checkpoints store args
    return torch.load(path, map_location="cpu", weights_only=True)


# ------------------------------------------------------------------ architecture patches
def patch_activations(module: nn.Module, fmt: str) -> nn.Module:
    """SurgeNet's CaFormer uses plain ReLU where stock CaFormer uses StarReLU.

    This is part of the ARCHITECTURE: loading SurgeNet weights into an unpatched timm
    caformer_s18 'works' (only act scale/bias keys missing) but computes relu(x)^2.
    """
    if fmt == "metaformer_relu":
        from timm.models.metaformer import StarReLU
        for parent in list(module.modules()):
            for cname, child in list(parent.named_children()):
                if isinstance(child, StarReLU):
                    setattr(parent, cname, nn.ReLU())
    return module


# ------------------------------------------------------------------- key conversions
def convert_torchvision_convnext(sd: dict) -> dict:
    """torchvision.models.convnext_* state dict -> timm convnext_* state dict."""
    out = {}
    stage_of = {1: 0, 3: 1, 5: 2, 7: 3}
    down_of = {2: 1, 4: 2, 6: 3}
    for k, v in sd.items():
        m = re.match(r"features\.(\d+)\.(.*)", k)
        if not m:
            if k.startswith("classifier.0."):
                out["head.norm." + k[len("classifier.0."):]] = v
            continue
        f, rest = int(m.group(1)), m.group(2)
        if f == 0:
            out["stem." + rest] = v
        elif f in down_of:
            out[f"stages.{down_of[f]}.downsample.{rest}"] = v
        else:
            s = stage_of[f]
            b, sub = rest.split(".", 1)
            if sub == "layer_scale":
                out[f"stages.{s}.blocks.{b}.gamma"] = v.flatten()
                continue
            sub = (sub.replace("block.0.", "conv_dw.").replace("block.2.", "norm.")
                      .replace("block.3.", "mlp.fc1.").replace("block.5.", "mlp.fc2."))
            out[f"stages.{s}.blocks.{b}.{sub}"] = v
    return out


def load_timm_backbone_state(name: str, path: Path | None = None) -> tuple[nn.Module, dict]:
    """Full timm model (with the right activations) loaded with surgical weights."""
    spec = SURGICAL_BACKBONES[name]
    raw = _safe_load(path or download(name))
    full = patch_activations(timm.create_model(spec["timm"], pretrained=False), spec["format"])
    if spec["format"] == "metaformer_relu":
        from timm.models.metaformer import checkpoint_filter_fn
        sd = checkpoint_filter_fn(raw, full)
    elif spec["format"] == "torchvision_convnext_dino":
        teacher = raw["teacher"] if "teacher" in raw else raw
        bb = {k[len("backbone."):]: v for k, v in teacher.items() if k.startswith("backbone.")}
        sd = convert_torchvision_convnext(bb)
    else:
        raise ValueError(spec["format"])
    res = full.load_state_dict(sd, strict=False)
    bad_missing = [k for k in res.missing_keys if not k.startswith("head.")]
    if bad_missing or res.unexpected_keys:
        raise RuntimeError(f"{name}: weights do not fit {spec['timm']}: missing "
                           f"{bad_missing[:5]} unexpected {res.unexpected_keys[:5]}")
    return full, full.state_dict()


def _feature_key(k: str) -> str:
    # timm FeatureListNet flattens top-level Sequentials: "stages_0.x" <-> "stages.0.x"
    return re.sub(r"^([A-Za-z]+)_(\d+)\.", r"\1.\2.", k)


def load_into_smp_encoder(smp_model: nn.Module, name: str, path: Path | None = None) -> None:
    """Load surgical weights into an smp model built with encoder 'tu-<timm name>'."""
    _, full_sd = load_timm_backbone_state(name, path)
    feat = smp_model.encoder.model
    target = feat.state_dict()
    new = {}
    for k, v in target.items():
        src = _feature_key(k)
        if src not in full_sd or full_sd[src].shape != v.shape:
            raise RuntimeError(f"{name}: no matching weight for encoder param {k}")
        new[k] = full_sd[src]
    feat.load_state_dict(new, strict=True)
    print(f"loaded {name} surgical weights into encoder ({len(new)} tensors, "
          f"license {SURGICAL_BACKBONES[name]['license']})")


def main():   # pre-download on a login/interactive node:  python -m surgseg.backbones
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=list(SURGICAL_BACKBONES))
    for n in ap.parse_args().names:
        load_timm_backbone_state(n)
        print(f"{n}: OK")


if __name__ == "__main__":
    main()
