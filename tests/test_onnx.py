"""Tests for the torch-free ONNX inference path.

The NMS/letterbox helpers run anywhere; the full detector + parity smoke test
skip unless onnxruntime and the committed ONNX model are present.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from netinspect.onnx_infer import _letterbox, _nms

ONNX_MODEL = Path("models/yolo_damage_v1.onnx")


def test_nms_suppresses_overlap():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=float)
    scores = np.array([0.9, 0.8, 0.7])
    keep = _nms(boxes, scores, iou_thr=0.5)
    assert 0 in keep and 2 in keep and 1 not in keep   # near-duplicate of 0 removed


def test_nms_empty():
    assert _nms(np.zeros((0, 4)), np.zeros((0,)), 0.5) == []


def test_letterbox_is_square_and_keeps_ratio():
    pytest.importorskip("cv2")
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    lb, r, dw, dh = _letterbox(img, new_shape=128)
    assert lb.shape == (128, 128, 3)
    assert abs(r - 0.64) < 1e-6                          # 128/200
    assert dh > 0 and dw == 0                            # padded vertically


def test_onnx_detector_smoke():
    pytest.importorskip("onnxruntime")
    if not ONNX_MODEL.exists():
        pytest.skip("ONNX model not present (run scripts/export_onnx.py)")
    from netinspect.onnx_infer import OnnxDetector
    det = OnnxDetector(ONNX_MODEL, conf=0.25)
    out = det.predict(np.zeros((480, 480, 3), dtype=np.uint8))
    assert isinstance(out, list)                         # no crash; boxes are BBox
