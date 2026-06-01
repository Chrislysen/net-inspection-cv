"""Torch-free YOLO inference via ONNX Runtime.

This is the **deployable** inference path: it runs an exported YOLOv8 detector
with only ``onnxruntime`` + ``numpy`` + ``cv2`` — **no PyTorch and no
ultralytics**. That matters for production, where you want a small, dependency-
light runtime on the target device (and the same ONNX graph compiles to
TensorRT for FP16/INT8 on a Jetson).

It re-implements YOLOv8's pre/post-processing (letterbox, decode, NMS) so the
output matches the training-time path. ``scripts/export_onnx.py`` produces the
``.onnx``; a parity test checks this path agrees with the torch path.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .utils import BBox, optional_import


def _letterbox(img: np.ndarray, new_shape: int = 480, color: int = 114):
    """Resize keeping aspect ratio and pad to a square (YOLO-style)."""
    cv2 = optional_import("cv2")
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    if cv2 is not None:
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    else:
        ys = (np.linspace(0, h - 1, nh)).astype(int)
        xs = (np.linspace(0, w - 1, nw)).astype(int)
        resized = img[ys][:, xs]
    canvas = np.full((new_shape, new_shape, 3), color, dtype=img.dtype)
    dh, dw = (new_shape - nh) // 2, (new_shape - nw) // 2
    canvas[dh:dh + nh, dw:dw + nw] = resized
    return canvas, r, dw, dh


def _nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """Greedy NMS; returns kept indices (pure NumPy, no cv2 dependency)."""
    if len(boxes_xyxy) == 0:
        return []
    x1, y1, x2, y2 = boxes_xyxy.T
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


class OnnxDetector:
    """YOLOv8 ONNX detector with no torch/ultralytics dependency."""

    def __init__(self, onnx_path: str | Path, conf: float = 0.25, iou: float = 0.5,
                 class_names: list[str] | None = None, providers: list[str] | None = None):
        ort = optional_import("onnxruntime")
        if ort is None:
            raise RuntimeError("onnxruntime is required. Install `.[export]`.")
        self.sess = ort.InferenceSession(
            str(onnx_path), providers=providers or ["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0]
        self.iname = self.inp.name
        # Static square input size from the model, else default 480.
        shape = self.inp.shape
        self.imgsz = int(shape[-1]) if isinstance(shape[-1], int) else 480
        self.conf = conf
        self.iou = iou
        self.class_names = class_names or ["damage"]

    def predict(self, image_rgb: np.ndarray) -> list[BBox]:
        lb, r, dw, dh = _letterbox(image_rgb, self.imgsz)
        x = (lb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        out = self.sess.run(None, {self.iname: x})[0]  # [1, 4+nc, N]
        preds = np.squeeze(out, 0).T                    # [N, 4+nc]
        if preds.shape[1] < 5:
            return []
        xywh = preds[:, :4]
        cls_scores = preds[:, 4:]
        cls_id = cls_scores.argmax(1)
        conf = cls_scores.max(1)
        keep = conf >= self.conf
        xywh, cls_id, conf = xywh[keep], cls_id[keep], conf[keep]
        if len(xywh) == 0:
            return []
        # cx,cy,w,h (letterboxed pixels) -> xyxy, then undo letterbox to original.
        cx, cy, bw, bh = xywh.T
        x1 = (cx - bw / 2 - dw) / r
        y1 = (cy - bh / 2 - dh) / r
        x2 = (cx + bw / 2 - dw) / r
        y2 = (cy + bh / 2 - dh) / r
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        idx = _nms(boxes, conf, self.iou)
        h, w = image_rgb.shape[:2]
        out_boxes = []
        for i in idx:
            bx1, by1, bx2, by2 = boxes[i]
            ci = int(cls_id[i])
            out_boxes.append(BBox(
                float(np.clip(bx1, 0, w)), float(np.clip(by1, 0, h)),
                float(np.clip(bx2, 0, w)), float(np.clip(by2, 0, h)),
                ci, self.class_names[ci] if ci < len(self.class_names) else f"class_{ci}",
                float(conf[i])))
        out_boxes.sort(key=lambda b: b.score, reverse=True)
        return out_boxes
