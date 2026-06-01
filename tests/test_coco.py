"""Tests for the COCO->YOLO adapter and per-class evaluation."""
from __future__ import annotations

import json

import numpy as np

from netinspect.coco import convert_coco_to_yolo
from netinspect.evaluate import evaluate_per_class
from netinspect.utils import BBox, write_image


def _make_coco(tmp_path):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    write_image(img_dir / "a.jpg", np.zeros((100, 200, 3), dtype=np.uint8))
    coco = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 200, "height": 100}],
        "categories": [{"id": 5, "name": "debris"}, {"id": 9, "name": "plant"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 5, "bbox": [40, 20, 20, 40], "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 9, "bbox": [0, 0, 100, 50], "iscrowd": 0},
        ],
    }
    j = tmp_path / "anns.json"
    j.write_text(json.dumps(coco), encoding="utf-8")
    return j, img_dir


def test_coco_to_yolo_box_conversion(tmp_path):
    j, img_dir = _make_coco(tmp_path)
    res = convert_coco_to_yolo(j, img_dir, tmp_path / "out")
    assert res.num_images == 1 and res.num_labels == 1 and res.num_instances == 2
    assert res.class_names == ["debris", "plant"]
    label = (tmp_path / "out" / "labels" / "a.txt").read_text().splitlines()
    # First ann: bbox [40,20,20,40] in 200x100 -> xc=(40+10)/200=0.25, yc=(20+20)/100=0.4
    cls, xc, yc, w, h = label[0].split()
    assert cls == "0"
    assert abs(float(xc) - 0.25) < 1e-6 and abs(float(yc) - 0.4) < 1e-6
    assert abs(float(w) - 0.1) < 1e-6 and abs(float(h) - 0.4) < 1e-6
    assert (tmp_path / "out" / "dataset.yaml").exists()


def test_coco_single_class(tmp_path):
    j, img_dir = _make_coco(tmp_path)
    res = convert_coco_to_yolo(j, img_dir, tmp_path / "out", single_class=True)
    assert res.class_names == ["object"]
    label = (tmp_path / "out" / "labels" / "a.txt").read_text().splitlines()
    assert all(line.split()[0] == "0" for line in label)


def test_coco_skips_missing_image(tmp_path):
    j, img_dir = _make_coco(tmp_path)
    (img_dir / "a.jpg").unlink()
    res = convert_coco_to_yolo(j, img_dir, tmp_path / "out")
    assert res.num_images == 0 and "a.jpg" in res.skipped


def _box(x1, y1, x2, y2, cls, score=0.9):
    return BBox(x1, y1, x2, y2, cls, f"c{cls}", score)


def test_evaluate_per_class():
    gts = {"img": [_box(0, 0, 10, 10, 0), _box(50, 50, 60, 60, 1)]}
    preds = {"img": [_box(0, 0, 10, 10, 0, 0.9)]}   # got class 0, missed class 1
    res = evaluate_per_class(preds, gts, iou_threshold=0.3,
                             class_names={0: "hole", 1: "tear"})
    assert res["per_class"]["hole"]["recall"] == 1.0
    assert res["per_class"]["tear"]["recall"] == 0.0
    assert 0.0 <= res["macro"]["f1"] <= 1.0
    assert res["overall_class_agnostic"]["tp"] == 1
