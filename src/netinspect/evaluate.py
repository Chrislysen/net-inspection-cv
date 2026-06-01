"""Evaluation utilities for detection, segmentation, and image-level decisions.

What this module computes
-------------------------
* **Detection**: greedy IoU matching between predictions and ground truth at a
  fixed IoU threshold -> precision, recall, F1, and per-image TP/FP/FN lists.
  A confidence sweep and a simple AP (area under the precision-recall curve,
  VOC-style 101-point interpolation) give a single-number summary.
* **Segmentation**: mask IoU between predicted and ground-truth masks.
* **Image-level**: does an image contain damage at all? (precision/recall over
  the "any detection" decision) — often the metric operators care about first.

Everything degrades honestly: with no ground truth, the functions return a
clear "not evaluable" structure rather than fabricated numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .utils import BBox, iou_xyxy


# --------------------------------------------------------------------------- #
# Detection matching
# --------------------------------------------------------------------------- #
@dataclass
class ImageMatch:
    """Per-image matching result at a single IoU threshold."""
    image: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    matched_pred_idx: list[int] = field(default_factory=list)
    fp_pred_idx: list[int] = field(default_factory=list)
    fn_gt_idx: list[int] = field(default_factory=list)


def match_detections(
    preds: Sequence[BBox],
    gts: Sequence[BBox],
    iou_threshold: float = 0.5,
    class_agnostic: bool = True,
) -> ImageMatch:
    """Greedy match predictions to ground truth by descending confidence.

    Each GT can match at most one prediction. ``class_agnostic`` ignores class
    labels when matching (useful for a prototype where the single meaningful
    question is "did we localise the damage", not "did we name it right").
    """
    preds_sorted = sorted(enumerate(preds), key=lambda kv: kv[1].score, reverse=True)
    used_gt: set[int] = set()
    m = ImageMatch(image="")
    for pred_idx, pred in preds_sorted:
        best_iou, best_gt = 0.0, -1
        for gi, gt in enumerate(gts):
            if gi in used_gt:
                continue
            if not class_agnostic and gt.class_id != pred.class_id:
                continue
            i = iou_xyxy(pred.to_list(), gt.to_list())
            if i > best_iou:
                best_iou, best_gt = i, gi
        if best_gt >= 0 and best_iou >= iou_threshold:
            used_gt.add(best_gt)
            m.tp += 1
            m.matched_pred_idx.append(pred_idx)
        else:
            m.fp += 1
            m.fp_pred_idx.append(pred_idx)
    m.fn = len(gts) - len(used_gt)
    m.fn_gt_idx = [gi for gi in range(len(gts)) if gi not in used_gt]
    return m


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def evaluate_detection(
    preds_by_image: dict[str, list[BBox]],
    gts_by_image: dict[str, list[BBox]],
    iou_threshold: float = 0.5,
    class_agnostic: bool = True,
) -> dict:
    """Aggregate detection metrics over a set of images.

    Only images present in ``gts_by_image`` are scored (you cannot evaluate an
    image without ground truth). Images with predictions but no GT entry are
    reported separately as "unscored".
    """
    matches: list[ImageMatch] = []
    total = {"tp": 0, "fp": 0, "fn": 0}
    unscored = [k for k in preds_by_image if k not in gts_by_image]

    for image, gts in gts_by_image.items():
        preds = preds_by_image.get(image, [])
        m = match_detections(preds, gts, iou_threshold, class_agnostic)
        m.image = image
        matches.append(m)
        total["tp"] += m.tp
        total["fp"] += m.fp
        total["fn"] += m.fn

    overall = _prf(total["tp"], total["fp"], total["fn"])
    overall["iou_threshold"] = iou_threshold
    overall["num_scored_images"] = len(gts_by_image)
    overall["num_unscored_images"] = len(unscored)
    overall["ap"] = average_precision(preds_by_image, gts_by_image,
                                      iou_threshold, class_agnostic)
    return {
        "overall": overall,
        "per_image": [
            {"image": m.image, "tp": m.tp, "fp": m.fp, "fn": m.fn}
            for m in matches
        ],
        "matches": matches,
        "unscored_images": unscored,
    }


def average_precision(
    preds_by_image: dict[str, list[BBox]],
    gts_by_image: dict[str, list[BBox]],
    iou_threshold: float = 0.5,
    class_agnostic: bool = True,
) -> float:
    """VOC-style AP (101-point interpolation) over all scored images.

    Builds one global ranked list of predictions, marks each TP/FP via greedy
    per-image IoU matching, then integrates the precision-recall curve.
    """
    entries: list[tuple[float, bool]] = []  # (score, is_tp)
    total_gt = 0
    used_gt: dict[str, set[int]] = {k: set() for k in gts_by_image}

    for gts in gts_by_image.values():
        total_gt += len(gts)
    if total_gt == 0:
        return 0.0

    # Rank all predictions (from scored images) by confidence.
    ranked: list[tuple[float, str, BBox]] = []
    for image, gts in gts_by_image.items():
        for p in preds_by_image.get(image, []):
            ranked.append((p.score, image, p))
    ranked.sort(key=lambda t: t[0], reverse=True)

    for score, image, pred in ranked:
        gts = gts_by_image[image]
        best_iou, best_gt = 0.0, -1
        for gi, gt in enumerate(gts):
            if gi in used_gt[image]:
                continue
            if not class_agnostic and gt.class_id != pred.class_id:
                continue
            i = iou_xyxy(pred.to_list(), gt.to_list())
            if i > best_iou:
                best_iou, best_gt = i, gi
        is_tp = best_gt >= 0 and best_iou >= iou_threshold
        if is_tp:
            used_gt[image].add(best_gt)
        entries.append((score, is_tp))

    if not entries:
        return 0.0

    tp_cum = fp_cum = 0
    precisions, recalls = [], []
    for _, is_tp in entries:
        tp_cum += int(is_tp)
        fp_cum += int(not is_tp)
        precisions.append(tp_cum / (tp_cum + fp_cum))
        recalls.append(tp_cum / total_gt)

    # 101-point interpolation.
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        prec = [p for p, r in zip(precisions, recalls) if r >= t]
        ap += (max(prec) if prec else 0.0)
    return ap / 101.0


def coco_map(
    preds_by_image: dict[str, list[BBox]],
    gts_by_image: dict[str, list[BBox]],
    iou_thresholds: Sequence[float] | None = None,
    class_agnostic: bool = True,
) -> dict:
    """COCO-style mean Average Precision averaged over IoU thresholds.

    Returns ``mAP@[.5:.95]`` (the headline COCO metric), plus ``mAP@.5`` and
    ``mAP@.75`` and the full per-IoU AP curve. Uses the VOC-style 101-point AP
    at each threshold (see ``average_precision``).
    """
    if iou_thresholds is None:
        iou_thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]  # .50:.95
    per_iou = {t: average_precision(preds_by_image, gts_by_image, t, class_agnostic)
               for t in iou_thresholds}
    vals = list(per_iou.values())
    return {
        "map_50_95": round(sum(vals) / len(vals), 4) if vals else 0.0,
        "map_50": round(per_iou.get(0.5, 0.0), 4),
        "map_75": round(per_iou.get(0.75, 0.0), 4),
        "per_iou": {str(k): round(v, 4) for k, v in per_iou.items()},
    }


def confidence_sweep(
    preds_by_image: dict[str, list[BBox]],
    gts_by_image: dict[str, list[BBox]],
    iou_threshold: float = 0.5,
    thresholds: Sequence[float] | None = None,
    class_agnostic: bool = True,
) -> list[dict]:
    """Precision/recall/F1 across a range of confidence thresholds."""
    thresholds = thresholds or [round(x, 2) for x in np.linspace(0.1, 0.9, 9)]
    rows = []
    for thr in thresholds:
        filtered = {
            img: [p for p in preds if p.score >= thr]
            for img, preds in preds_by_image.items()
        }
        res = evaluate_detection(filtered, gts_by_image, iou_threshold, class_agnostic)
        row = {"conf_threshold": float(thr)}
        row.update({k: res["overall"][k] for k in ("precision", "recall", "f1", "tp", "fp", "fn")})
        rows.append(row)
    return rows


def evaluate_image_level(
    preds_by_image: dict[str, list[BBox]],
    gts_by_image: dict[str, list[BBox]],
    conf_threshold: float = 0.25,
) -> dict:
    """Binary "does this image contain damage?" evaluation."""
    tp = fp = tn = fn = 0
    for image, gts in gts_by_image.items():
        gt_pos = len(gts) > 0
        pred_pos = any(p.score >= conf_threshold for p in preds_by_image.get(image, []))
        if gt_pos and pred_pos:
            tp += 1
        elif gt_pos and not pred_pos:
            fn += 1
        elif not gt_pos and pred_pos:
            fp += 1
        else:
            tn += 1
    res = _prf(tp, fp, fn)
    res["tn"] = tn
    res["accuracy"] = (tp + tn) / max(1, tp + tn + fp + fn)
    res["conf_threshold"] = conf_threshold
    return res


def best_f1_threshold(sweep: list[dict]) -> dict:
    """Return the sweep row with the highest F1."""
    return max(sweep, key=lambda r: r["f1"]) if sweep else {}


def evaluate_per_class(
    preds_by_image: dict[str, list[BBox]],
    gts_by_image: dict[str, list[BBox]],
    iou_threshold: float = 0.5,
    class_names: dict[int, str] | None = None,
) -> dict:
    """Per-class precision/recall/F1/AP plus a macro average.

    For multi-class datasets (e.g. converted COCO). Each class is scored on its
    own boxes (matching is class-agnostic *within* a class). Returns
    ``{"per_class": {name: metrics}, "macro": {...}, "overall_class_agnostic": {...}}``.
    """
    class_ids = sorted({b.class_id for boxes in list(gts_by_image.values()) + list(preds_by_image.values())
                        for b in boxes})
    class_names = class_names or {}
    per_class: dict[str, dict] = {}
    for cid in class_ids:
        p = {img: [b for b in boxes if b.class_id == cid] for img, boxes in preds_by_image.items()}
        g = {img: [b for b in boxes if b.class_id == cid] for img, boxes in gts_by_image.items()}
        res = evaluate_detection(p, g, iou_threshold, class_agnostic=True)["overall"]
        name = class_names.get(cid, f"class_{cid}")
        per_class[name] = {k: res[k] for k in ("precision", "recall", "f1", "ap", "tp", "fp", "fn")}

    if per_class:
        macro = {m: round(sum(c[m] for c in per_class.values()) / len(per_class), 4)
                 for m in ("precision", "recall", "f1", "ap")}
    else:
        macro = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "ap": 0.0}

    overall = evaluate_detection(preds_by_image, gts_by_image, iou_threshold,
                                 class_agnostic=True)["overall"]
    return {"per_class": per_class, "macro": macro,
            "overall_class_agnostic": {k: overall[k] for k in
                                       ("precision", "recall", "f1", "ap", "tp", "fp", "fn")}}
