"""Tests for geometry and evaluation metrics."""
from __future__ import annotations

import numpy as np

from netinspect.evaluate import (average_precision, confidence_sweep,
                                  evaluate_detection, evaluate_image_level,
                                  match_detections)
from netinspect.utils import BBox, iou_xyxy, mask_iou


def test_iou_identical_and_disjoint():
    assert iou_xyxy([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou_xyxy([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_iou_half_overlap():
    # Two 10x10 boxes overlapping in a 10x5 region -> inter=50, union=150.
    assert abs(iou_xyxy([0, 0, 10, 10], [0, 5, 10, 15]) - (50 / 150)) < 1e-9


def test_mask_iou():
    a = np.zeros((10, 10), dtype=bool); a[:5, :] = True
    b = np.zeros((10, 10), dtype=bool); b[:5, :5] = True
    # inter=25, union=50 -> 0.5
    assert abs(mask_iou(a, b) - 0.5) < 1e-9


def _box(x1, y1, x2, y2, score=1.0, cls=0):
    return BBox(x1, y1, x2, y2, cls, "damage", score)


def test_match_perfect():
    gt = [_box(0, 0, 10, 10)]
    pred = [_box(0, 0, 10, 10, score=0.9)]
    m = match_detections(pred, gt, iou_threshold=0.5)
    assert m.tp == 1 and m.fp == 0 and m.fn == 0


def test_match_false_positive_and_negative():
    gt = [_box(0, 0, 10, 10)]
    pred = [_box(50, 50, 60, 60, score=0.9)]  # nowhere near GT
    m = match_detections(pred, gt, iou_threshold=0.5)
    assert m.tp == 0 and m.fp == 1 and m.fn == 1
    assert m.fp_pred_idx == [0] and m.fn_gt_idx == [0]


def test_one_gt_matches_only_one_pred():
    gt = [_box(0, 0, 10, 10)]
    preds = [_box(0, 0, 10, 10, score=0.9), _box(0, 0, 10, 10, score=0.8)]
    m = match_detections(preds, gt, iou_threshold=0.5)
    assert m.tp == 1 and m.fp == 1 and m.fn == 0


def test_evaluate_detection_aggregate():
    gts = {"a": [_box(0, 0, 10, 10)], "b": [_box(0, 0, 10, 10)]}
    preds = {"a": [_box(0, 0, 10, 10, 0.9)], "b": [_box(99, 99, 100, 100, 0.9)]}
    res = evaluate_detection(preds, gts, iou_threshold=0.5)
    o = res["overall"]
    assert o["tp"] == 1 and o["fp"] == 1 and o["fn"] == 1
    assert abs(o["precision"] - 0.5) < 1e-9
    assert abs(o["recall"] - 0.5) < 1e-9


def test_average_precision_perfect_is_one():
    gts = {"a": [_box(0, 0, 10, 10)]}
    preds = {"a": [_box(0, 0, 10, 10, 0.9)]}
    assert abs(average_precision(preds, gts, 0.5) - 1.0) < 1e-6


def test_image_level_eval():
    gts = {"pos": [_box(0, 0, 10, 10)], "neg": []}
    preds = {"pos": [_box(0, 0, 10, 10, 0.9)], "neg": [_box(0, 0, 5, 5, 0.9)]}
    res = evaluate_image_level(preds, gts, conf_threshold=0.25)
    assert res["tp"] == 1 and res["fp"] == 1 and res["fn"] == 0


def test_confidence_sweep_monotonic_filtering():
    gts = {"a": [_box(0, 0, 10, 10)]}
    preds = {"a": [_box(0, 0, 10, 10, 0.4)]}
    sweep = confidence_sweep(preds, gts, 0.5, thresholds=[0.3, 0.5])
    # At thr 0.3 the pred survives (TP); at 0.5 it is filtered out (FN).
    assert sweep[0]["tp"] == 1
    assert sweep[1]["tp"] == 0 and sweep[1]["fn"] == 1


def test_no_ground_truth_is_handled():
    res = evaluate_detection({"a": [_box(0, 0, 10, 10, 0.9)]}, {})
    assert res["overall"]["num_scored_images"] == 0
