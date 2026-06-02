"""Tests for the self-supervised DINOv2 backbone wiring.

The routing/contract checks run offline. The end-to-end extraction test needs
the DINOv2 weights (downloaded once via torch.hub) and is skipped when torch is
absent or the weights cannot be loaded (e.g. offline CI with no cache).
"""
from __future__ import annotations

import numpy as np
import pytest

from netinspect.dino_backbone import is_dino_backbone


def test_is_dino_backbone_routing():
    assert is_dino_backbone("dinov2_vits14")
    assert is_dino_backbone("dino_vits14")
    assert not is_dino_backbone("resnet18")
    assert not is_dino_backbone("wide_resnet50_2")


def test_patchcore_routes_dino_backbone(monkeypatch):
    """A dino* backbone must select the DINO extractor, not the torchvision CNN."""
    pytest.importorskip("torch")
    import netinspect.patchcore as pc

    created = {}

    class _Fake:
        def __init__(self, cfg):
            created["dino"] = True
            self.cfg = cfg

    monkeypatch.setattr("netinspect.dino_backbone.DinoFeatureExtractor", _Fake)
    pc._EXTRACTOR_CACHE.clear()
    cfg = pc.PatchCoreConfig(backbone="dinov2_vits14", input_size=224)
    ex = pc._get_extractor(cfg)
    assert created.get("dino") and isinstance(ex, _Fake)
    pc._EXTRACTOR_CACHE.clear()


def test_dino_extract_shape():
    pytest.importorskip("torch")
    from netinspect.dino_backbone import DinoFeatureExtractor
    from netinspect.patchcore import PatchCoreConfig

    cfg = PatchCoreConfig(backbone="dinov2_vits14", input_size=224, neighbourhood=1)
    try:
        ex = DinoFeatureExtractor(cfg)
    except RuntimeError as exc:  # offline + uncached weights
        pytest.skip(f"DINOv2 weights unavailable: {exc}")
    img = (np.random.default_rng(0).integers(0, 255, (240, 320, 3))).astype(np.uint8)
    feat, (h, w) = ex.extract(img)
    assert (h, w) == (16, 16)            # 224 / 14
    assert feat.shape == (256, 384)      # 16*16 tokens, ViT-S/14 dim
