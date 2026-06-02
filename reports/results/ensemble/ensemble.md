# Ensemble — robust detector proposes, segmenter confirms

det=yolo_damage_v1.pt, seg=yolo_damage_seg_v3.pt, agree_iou=0.3, conf=0.25

## False positives on REAL UNDAMAGED net (lower is better)

| Model | bag1 (train backgrounds) | bag2 (same site, other clip) | different DAY |
|---|---|---|---|
| det v1 | 0% | 0% | 1% |
| seg v3 | 0% | 2% | 18% |
| ensemble (det∧seg) | 0% | 0% | 0% |

## Damage recall (F1) on composited test sets

| Model | in-clip | cross-clip (bag2) | different-day |
|---|---|---|---|
| det v1 | 0.968 | 0.979 | 0.557 |
| seg v3 | 0.947 | 0.97 | 0.926 |
| ensemble (det∧seg) | 0.968 | 0.979 | 0.571 |

> Agreement of two independently-trained models suppresses model-specific spurious cues. Both models are trained on synthetic damage on real backgrounds: this strengthens proxy robustness, not validated real-damage accuracy.