# Self-supervised vs supervised features for anomaly detection

Same PatchCore detector, same normal training frames — only the patch **backbone** differs. DINOv2 features are self-supervised (no labels); ResNet18 features are ImageNet-supervised.

**Headline metric is image-level AUROC** — threshold-free separability of damaged vs undamaged frames by the anomaly score. It is the fair test of feature quality because it does not depend on a threshold (and the threshold is exactly what fails to transfer across days). Localisation/FP columns use the default 2x-median threshold at conf=0.25, IoU=0.3.

## Image-level AUROC (damaged vs undamaged frames) — higher is better

| Backbone | in-clip | different-day |
|---|---|---|
| resnet18 | 0.984 | 0.994 |
| dinov2 | 1.0 | 0.957 |

## Localisation of (synthetic) damage — boxes at the default threshold

| Backbone | Background | Precision | Recall | F1 | AP |
|---|---|---|---|---|---|
| resnet18 | in-clip | 0.115 | 0.234 | 0.154 | 0.112 |
| resnet18 | different-day | 0.0 | 0.0 | 0.0 | 0.0 |
| dinov2 | in-clip | 0.041 | 0.064 | 0.05 | 0.012 |
| dinov2 | different-day | 0.0 | 0.0 | 0.0 | 0.0 |

## False alarms on REAL UNDAMAGED net (default threshold)

| Backbone | Set | Frames | Mean det/frame | FP frame rate |
|---|---|---|---|---|
| resnet18 | in-clip | 38 | 2.5 | 68% |
| resnet18 | different-day | 80 | 2.475 | 100% |
| dinov2 | in-clip | 38 | 0.5 | 32% |
| dinov2 | different-day | 80 | 1.0 | 100% |

> DINOv2 here is pretrained on natural images, NOT on SOLAQUA — this measures transfer of published self-supervised features, not the deferred SOLAQUA-pretraining experiment. Damage is synthetic and the net is undamaged: this characterises behaviour, not real-damage performance.