"""Tests for damage compositing and the unified inference facade."""
from __future__ import annotations

import numpy as np

from netinspect.compose import ComposeConfig, build_dataset, composite_damage
from netinspect.inference import NetInspector
from netinspect.utils import write_image


def _fake_net(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((180, 240, 3), (40, 110, 90), dtype=np.uint8)
    img[::6, :] = (180, 200, 150)
    img[:, ::6] = (180, 200, 150)
    return np.clip(img.astype(int) + rng.integers(-6, 6, img.shape), 0, 255).astype(np.uint8)


def test_composite_damage_returns_boxes_and_polys():
    img = _fake_net()
    out, boxes, polys = composite_damage(img, np.random.default_rng(1), num=3,
                                         cfg=ComposeConfig())
    assert out.shape == img.shape
    assert len(boxes) == 3 and len(polys) == 3
    # Compositing must visibly modify the frame (seamless blend can lighten or
    # darken, so we only assert it changed, not the sign).
    assert not np.array_equal(out, img)
    for b in boxes:
        assert b.x2 > b.x1 and b.y2 > b.y1
        assert b.class_name == "damage"  # single-class default


def test_composite_damage_is_reproducible():
    img = _fake_net()
    a, ba, _ = composite_damage(img, np.random.default_rng(7), 2)
    b, bb, _ = composite_damage(img, np.random.default_rng(7), 2)
    assert np.array_equal(a, b)
    assert [x.to_list() for x in ba] == [x.to_list() for x in bb]


def test_build_dataset_splits(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    paths = []
    for i in range(10):
        p = frames_dir / f"f{i:02d}.jpg"
        write_image(p, _fake_net(i))
        paths.append(p)
    info = build_dataset(paths, tmp_path / "ds", seed=0)
    total = sum(c["images"] for c in info["splits"].values())
    assert total == 10
    assert (tmp_path / "ds" / "dataset.yaml").exists()
    assert (tmp_path / "ds" / "images" / "train").exists()


def test_inspector_classical_always_available():
    insp = NetInspector()
    assert "classical" in insp.available_methods()
    res = insp.predict(_fake_net(), method="classical")
    assert res.method == "classical"
    assert res.elapsed_ms >= 0


def test_inspector_unknown_method_raises():
    insp = NetInspector()
    try:
        insp.predict(_fake_net(), method="nope")
        assert False, "should have raised"
    except ValueError:
        pass
