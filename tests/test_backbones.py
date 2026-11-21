"""Backbone conversion tests: converted weights must compute the SAME features."""
from __future__ import annotations

import timm
import torch
import torchvision

from surgseg.backbones import _feature_key, convert_torchvision_convnext


def test_torchvision_convnext_to_timm_is_exact():
    torch.manual_seed(0)
    tv = torchvision.models.convnext_tiny().eval()
    for p in tv.parameters():                    # random, non-trivial values everywhere
        p.data.normal_(0, 0.05)
    tm = timm.create_model("convnext_tiny", pretrained=False).eval()
    res = tm.load_state_dict(convert_torchvision_convnext(tv.state_dict()), strict=False)
    assert not res.unexpected_keys and all(k.startswith("head.fc") for k in res.missing_keys), res
    x = torch.randn(1, 3, 128, 160)
    with torch.no_grad():
        a = tv.features(x)
        b = tm.forward_features(x)
    assert torch.allclose(a, b, atol=1e-4), float((a - b).abs().max())


def test_feature_key_mapping():
    assert _feature_key("stages_0.blocks.1.norm.weight") == "stages.0.blocks.1.norm.weight"
    assert _feature_key("stem_0.weight") == "stem.0.weight"
    assert _feature_key("stem.conv.weight") == "stem.conv.weight"


if __name__ == "__main__":
    test_torchvision_convnext_to_timm_is_exact()
    test_feature_key_mapping()
    print("PASS backbones")
