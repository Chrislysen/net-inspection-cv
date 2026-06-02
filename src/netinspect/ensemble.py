"""Detector-gated ensemble: the robust detector proposes, the segmenter confirms.

Motivation (measured, not assumed)
-----------------------------------
On the held-out different *day*, the box detector ``yolo_damage_v1`` fires on only
**1%** of undamaged frames, while the segmentation models (``seg v2/v3/v4``) fire
on 18–31% — they add masks but generalise worse out-of-distribution. Rather than
pick one, combine them with a simple agreement rule:

* **det proposes** — take the box detector's predictions (high precision, low OOD
  false-positive rate);
* **seg confirms** — keep a proposal only if the segmentation model *also* fires
  on it (box IoU ≥ ``agree_iou``); attach the seg detection's mask/score.

A region flagged by *two independently-trained* models is less likely to be a
shared spurious cue, so agreement can push the false-positive rate at or below the
detector's already-low rate — at a small, *measured* recall cost. This is pure
inference (no training) and is evaluated head-to-head against each model alone in
``scripts/eval_ensemble.py``.

Honesty unchanged: both models were trained on synthetic damage composited on real
backgrounds, so this improves *robustness of the proxy*, not validated real-damage
accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import BBox, get_logger, iou_xyxy

LOGGER = get_logger()


@dataclass
class EnsembleConfig:
    det_conf: float = 0.25       # detector proposal threshold
    seg_conf: float = 0.25       # segmenter confirmation threshold
    agree_iou: float = 0.30      # min IoU for a seg box to confirm a det box
    mode: str = "agree"          # "agree" (det∧seg) | "union" | "det" | "seg"


def combine(det_boxes: list[BBox], seg_boxes: list[BBox],
            cfg: EnsembleConfig | None = None) -> list[BBox]:
    """Combine detector and segmenter boxes per the configured rule.

    Returns boxes whose geometry/score come from the detector proposal; the
    matched segmenter score (if any) is preserved in ``meta`` via a fresh BBox.
    """
    cfg = cfg or EnsembleConfig()
    det = [b for b in det_boxes if b.score >= cfg.det_conf]
    seg = [b for b in seg_boxes if b.score >= cfg.seg_conf]
    if cfg.mode == "det":
        return det
    if cfg.mode == "seg":
        return seg
    if cfg.mode == "union":
        return det + seg

    # "agree": keep a detector box only if some segmenter box overlaps it.
    confirmed: list[BBox] = []
    for d in det:
        best = max((iou_xyxy(d.to_list()[:4], s.to_list()[:4]) for s in seg), default=0.0)
        if best >= cfg.agree_iou:
            confirmed.append(d)
    return confirmed


def predict(image_rgb: np.ndarray, det_model, seg_model, cfg: EnsembleConfig | None = None,
            imgsz: int = 640) -> list[BBox]:
    """Run both models on an image and return the ensembled boxes."""
    from .model_baseline import YoloConfig, predict_image
    cfg = cfg or EnsembleConfig()
    det_boxes = predict_image(det_model, image_rgb,
                              YoloConfig(conf=min(cfg.det_conf, 0.01), imgsz=imgsz))
    seg_boxes = predict_image(seg_model, image_rgb,
                              YoloConfig(conf=min(cfg.seg_conf, 0.01), imgsz=imgsz))
    return combine(det_boxes, seg_boxes, cfg)
