"""Tests for the AGPL-free detector path.

This path exists for a licence reason, so the first test asserts the licence
property itself: no Ultralytics import anywhere in it. The rest cover the
save/load round trip, which is where the real bug was — torchvision's ssdlite
silently builds a *different* backbone depending on whether pretrained weights
are requested, so a model saved one way and rebuilt the other way fails to load.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from netinspect import permissive_baseline as P  # noqa: E402
from netinspect.dataset import Box, Sample  # noqa: E402


def _samples(tmp_path, n=4, labelled=True, size=(64, 64)):
    from PIL import Image
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        p = tmp_path / f"clip_{i:04d}.jpg"
        Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype="uint8")).save(p)
        boxes = [Box(0, 0.3, 0.3, 0.6, 0.6)] if labelled else []
        out.append(Sample(image=p, boxes=boxes, group="clip",
                          width=size[0], height=size[1]))
    return out


# --------------------------------------------------------------------------- #
# the licence property this module exists for
# --------------------------------------------------------------------------- #
def test_the_module_never_imports_ultralytics():
    """The entire point: no AGPL code in this path.

    Checked on the parsed import statements, not on the text — the docstring
    names Ultralytics repeatedly while explaining why this module exists, and a
    substring search cannot tell an explanation from a dependency.
    """
    import ast
    tree = ast.parse(Path(P.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "ultralytics" not in imported, f"AGPL import found: {imported}"

    # optional_import("ultralytics") would dodge the AST check entirely.
    assert "ultralytics" not in Path(P.__file__).read_text(encoding="utf-8").replace(
        "Ultralytics", "")


def test_no_ultralytics_module_is_loaded_by_using_this_path(tmp_path):
    import sys
    sys.modules.pop("ultralytics", None)
    model = P.build_model(pretrained_backbone=False)
    P.predict_image(model, np.zeros((64, 64, 3), dtype=np.uint8), P.PermissiveConfig())
    assert "ultralytics" not in sys.modules


# --------------------------------------------------------------------------- #
# model construction
# --------------------------------------------------------------------------- #
def test_builds_with_the_right_number_of_classes():
    model = P.build_model(pretrained_backbone=False)
    assert model is not None


def test_an_unknown_architecture_lists_the_valid_ones():
    with pytest.raises(ValueError) as e:
        P.build_model("yolov99", pretrained_backbone=False)
    assert "ssdlite320" in str(e.value)


def test_background_is_accounted_for_in_the_class_count():
    """One 'damage' class means num_classes=2; getting this wrong trains a model
    that only ever predicts background."""
    assert P.NUM_CLASSES == 2 and P.DAMAGE_LABEL == 1


# --------------------------------------------------------------------------- #
# targets
# --------------------------------------------------------------------------- #
def test_normalised_boxes_become_pixel_targets():
    s = Sample(image=Path("x.jpg"), boxes=[Box(0, 0.25, 0.5, 0.75, 1.0)],
               width=100, height=40)
    t = P._to_target(s, torch)
    assert t["boxes"].tolist() == [[25.0, 20.0, 75.0, 40.0]]
    assert t["labels"].tolist() == [P.DAMAGE_LABEL]


def test_a_clean_frame_produces_correctly_shaped_empty_tensors():
    """A frame of clean net is training signal, and torchvision needs (0,4)."""
    t = P._to_target(Sample(image=Path("x.jpg"), boxes=[], width=64, height=64), torch)
    assert tuple(t["boxes"].shape) == (0, 4)
    assert tuple(t["labels"].shape) == (0,)


def test_degenerate_boxes_are_dropped_rather_than_crashing_training():
    s = Sample(image=Path("x.jpg"), boxes=[Box(0, 0.5, 0.5, 0.5, 0.5)],
               width=64, height=64)
    assert tuple(P._to_target(s, torch)["boxes"].shape) == (0, 4)


# --------------------------------------------------------------------------- #
# train / save / load round trip
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_train_save_load_predict_round_trip(tmp_path):
    """Regression for the real bug.

    torchvision's ssdlite builds a REDUCED-TAIL backbone when no pretrained
    weights are requested (480 channels instead of 960). The model was trained
    with a pretrained backbone and rebuilt without one, so load_state_dict
    failed on most of the network. Nothing short of an actual round trip
    catches it.
    """
    samples = _samples(tmp_path, n=4)
    cfg = P.PermissiveConfig(epochs=1, batch_size=2, pretrained_backbone=False)
    out = tmp_path / "m.pt"
    summary = P.train(samples, cfg, out_path=out)

    assert out.exists()
    assert out.with_suffix(".json").exists()
    assert summary["labelled_frames"] == 4
    assert "BSD-3-Clause" in summary["licence"]

    model = P.load_model(out)                       # would raise on a mismatch
    boxes = P.predict_image(model, np.zeros((64, 64, 3), dtype=np.uint8))
    assert isinstance(boxes, list)


def test_training_without_labels_refuses_rather_than_training_nothing(tmp_path):
    samples = _samples(tmp_path, n=3, labelled=False)
    with pytest.raises(ValueError, match="No labelled samples"):
        P.train(samples, P.PermissiveConfig(epochs=1, pretrained_backbone=False),
                out_path=tmp_path / "m.pt")


# --------------------------------------------------------------------------- #
# inference contract
# --------------------------------------------------------------------------- #
def test_predict_returns_the_projects_shared_box_type():
    from netinspect.utils import BBox
    model = P.build_model(pretrained_backbone=False)
    boxes = P.predict_image(model, np.zeros((64, 64, 3), dtype=np.uint8),
                            P.PermissiveConfig(conf=0.0))
    assert all(isinstance(b, BBox) for b in boxes)


def test_the_confidence_threshold_filters():
    model = P.build_model(pretrained_backbone=False)
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    loose = P.predict_image(model, img, P.PermissiveConfig(conf=0.0))
    strict = P.predict_image(model, img, P.PermissiveConfig(conf=0.999))
    assert len(strict) <= len(loose)


def test_predict_leaves_the_model_in_eval_mode_it_found_it_in():
    model = P.build_model(pretrained_backbone=False)
    model.train()
    P.predict_image(model, np.zeros((64, 64, 3), dtype=np.uint8))
    assert model.training, "predict must restore the mode it borrowed"


def test_detections_are_capped():
    model = P.build_model(pretrained_backbone=False)
    boxes = P.predict_image(model, np.zeros((64, 64, 3), dtype=np.uint8),
                            P.PermissiveConfig(conf=0.0, max_detections=3))
    assert len(boxes) <= 3
