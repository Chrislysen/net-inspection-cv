# Method comparison

Test set: `data/processed/bag2_composite/images/test` (120 images, IoU=0.3, class-agnostic)

| Method | Precision | Recall | F1 | AP | Image-level acc |
|---|---|---|---|---|---|
| classical | 0.592 | 0.774 | 0.671 | 0.765 | 0.917 |
| anomaly | 0.074 | 0.038 | 0.050 | 0.006 | 0.692 |
| yolo | 0.981 | 0.976 | 0.979 | 0.970 | 1.000 |

> Numbers are on **synthetic damage composited on real backgrounds** from a single clip. Optimistic vs. truly independent sites; treat as relative comparison, not validated absolute performance.