"""Tests for dataset discovery and YOLO label parsing."""
from __future__ import annotations

import numpy as np

from netinspect.data import (load_dataset, parse_yolo_label, summarize_dataset)
from netinspect.utils import write_image


def _make_image(path, w=100, h=80):
    write_image(path, np.zeros((h, w, 3), dtype=np.uint8))


def test_parse_detection_label(tmp_path):
    label = tmp_path / "a.txt"
    label.write_text("0 0.5 0.5 0.2 0.4\n1 0.25 0.25 0.1 0.1\n")
    boxes, polys, kind, errors = parse_yolo_label(label, 100, 80)
    assert kind == "detection"
    assert not errors
    assert len(boxes) == 2 and not polys
    # First box: centre (50,40), size (20,32) -> [40,24,60,56]
    b = boxes[0]
    assert abs(b.x1 - 40) < 1e-6 and abs(b.y1 - 24) < 1e-6
    assert abs(b.x2 - 60) < 1e-6 and abs(b.y2 - 56) < 1e-6


def test_parse_segmentation_label(tmp_path):
    label = tmp_path / "a.txt"
    label.write_text("0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n")  # 4-point polygon
    boxes, polys, kind, errors = parse_yolo_label(label, 100, 80)
    assert kind == "segmentation"
    assert len(polys) == 1 and len(polys[0].points) == 4
    assert not errors


def test_parse_invalid_label_reports_error(tmp_path):
    label = tmp_path / "a.txt"
    label.write_text("0 0.5 0.5\n")  # 2 coords -> invalid (not 4, not even>=6)
    boxes, polys, kind, errors = parse_yolo_label(label, 100, 80)
    assert errors and not boxes and not polys


def test_load_dataset_with_and_without_labels(tmp_path):
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir(); lbl_dir.mkdir()
    _make_image(img_dir / "with.jpg")
    _make_image(img_dir / "without.jpg")
    (lbl_dir / "with.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    samples = load_dataset(img_dir, lbl_dir)
    assert len(samples) == 2
    by_name = {s.image_path.name: s for s in samples}
    assert by_name["with.jpg"].has_labels
    assert not by_name["without.jpg"].has_labels


def test_summarize_dataset(tmp_path):
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir(); lbl_dir.mkdir()
    _make_image(img_dir / "a.jpg")
    (lbl_dir / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n")
    samples = load_dataset(img_dir, lbl_dir)
    summary = summarize_dataset(samples)
    assert summary["num_images"] == 1
    assert summary["num_boxes"] == 2
    assert summary["class_distribution"]["damage"] == 1
    assert summary["class_distribution"]["hole"] == 1
    assert summary["num_label_errors"] == 0


def test_missing_directory_returns_empty(tmp_path):
    assert load_dataset(tmp_path / "does_not_exist") == []
