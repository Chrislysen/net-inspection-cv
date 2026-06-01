"""Tests for the anomaly model and SOLAQUA data structures (offline)."""
from __future__ import annotations

import numpy as np

from netinspect.anomaly import AnomalyConfig, fit, score_image
from netinspect.solaqua import DataFile


def _normal_net(seed: int) -> np.ndarray:
    """A simple repeating-texture 'normal' image (stand-in for net mesh)."""
    rng = np.random.default_rng(seed)
    img = np.full((160, 200, 3), (40, 110, 90), dtype=np.uint8)
    img[::8, :, :] = (180, 200, 150)   # horizontal mesh lines
    img[:, ::8, :] = (180, 200, 150)   # vertical mesh lines
    img = np.clip(img.astype(int) + rng.integers(-8, 8, img.shape), 0, 255).astype(np.uint8)
    return img


def test_anomaly_fit_and_score_runs():
    cfg = AnomalyConfig(resize=160, grid=8)
    model = fit([_normal_net(i) for i in range(5)], cfg)
    res = score_image(_normal_net(99), model)
    assert res.score_map.ndim == 2
    assert res.max_score >= 0.0


def test_anomaly_flags_out_of_distribution_region():
    cfg = AnomalyConfig(resize=160, grid=8, threshold_percentile=95.0)
    model = fit([_normal_net(i) for i in range(6)], cfg)
    # Inject a large dark, textureless block (unlike the regular mesh).
    anomalous = _normal_net(7)
    anomalous[40:120, 60:150] = (5, 5, 5)
    res = score_image(anomalous, model)
    normal_res = score_image(_normal_net(8), model)
    # The anomalous frame should score higher than a clean one.
    assert res.max_score > normal_res.max_score


def test_anomaly_model_save_load(tmp_path):
    cfg = AnomalyConfig(resize=160, grid=8)
    model = fit([_normal_net(i) for i in range(4)], cfg)
    path = tmp_path / "m"
    model.save(path)
    from netinspect.anomaly import AnomalyModel
    loaded = AnomalyModel.load(path)
    assert loaded.threshold == model.threshold
    assert loaded.cov_inv.shape == model.cov_inv.shape


def test_solaqua_datafile_properties():
    f = DataFile("abc", "2024-08-22_video.bag", 960_000_000, "fe-x")
    assert f.is_video_bag
    assert round(f.size_mb, 1) == 915.5
    assert not DataFile("d", "x_data.bag", 1000, "fe-x").is_video_bag
