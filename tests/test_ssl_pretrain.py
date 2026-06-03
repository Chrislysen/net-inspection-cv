"""Tests for the from-scratch SimCLR pieces (NT-Xent loss + config)."""
from __future__ import annotations

import pytest

from netinspect.ssl_pretrain import SimCLRConfig


def test_config_defaults():
    cfg = SimCLRConfig(epochs=10, batch=32)
    assert cfg.epochs == 10 and cfg.batch == 32 and cfg.proj_dim == 128


def test_nt_xent_rewards_aligned_views():
    """Loss must be lower when the two views of each item are clearly the closest."""
    torch = pytest.importorskip("torch")
    from netinspect.ssl_pretrain import _nt_xent

    torch.manual_seed(0)
    n, d = 8, 16
    base = torch.nn.functional.normalize(torch.randn(n, d), dim=1)
    # Aligned: view2 ≈ view1 (positives are each other's nearest neighbour).
    z1 = base
    z2 = torch.nn.functional.normalize(base + 0.01 * torch.randn(n, d), dim=1)
    aligned = _nt_xent(torch, z1, z2, temperature=0.5).item()
    # Misaligned: view2 is an unrelated random set.
    z2r = torch.nn.functional.normalize(torch.randn(n, d), dim=1)
    misaligned = _nt_xent(torch, z1, z2r, temperature=0.5).item()
    assert aligned < misaligned
    assert aligned >= 0.0


def test_nt_xent_finite():
    torch = pytest.importorskip("torch")
    from netinspect.ssl_pretrain import _nt_xent
    z = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
    loss = _nt_xent(torch, z, z, temperature=0.5)
    assert torch.isfinite(loss).all()
