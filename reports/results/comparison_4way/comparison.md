# Method comparison

Test set: `data/processed/real_composite/images/test` (29 images, IoU=0.3, class-agnostic)

| Method | Precision | Recall | F1 | AP | Image-level acc |
|---|---|---|---|---|---|
| classical | 0.411 | 0.638 | 0.500 | 0.549 | 0.793 |
| anomaly | 0.101 | 0.149 | 0.121 | 0.024 | 0.724 |
| patchcore | 0.868 | 0.702 | 0.776 | 0.703 | 0.931 |
| yolo | 0.958 | 0.979 | 0.968 | 0.970 | 1.000 |

> Numbers are on **synthetic damage composited on real backgrounds** from a single clip. Optimistic vs. truly independent sites; treat as relative comparison, not validated absolute performance.