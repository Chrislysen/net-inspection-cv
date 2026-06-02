# Adversarial evaluation — `yolo`

## 1. False positives on REAL UNDAMAGED net (no damage present → every detection is a false alarm)

| Frame set | Frames | Detections | Mean/frame | FP frame rate |
|---|---|---|---|---|
| bag1 (train backgrounds) | 38 | 2 | 0.053 | 5% |
| bag2 (same site, other clip) | 120 | 5 | 0.042 | 3% |
| different DAY | 200 | 107 | 0.535 | 36% |

## 2. Damage recall by background distance (composited damage)

| Background | Precision | Recall | F1 | AP |
|---|---|---|---|---|
| in-clip | 0.93 | 0.851 | 0.889 | 0.85 |
| cross-clip (bag2) | 0.931 | 0.892 | 0.911 | 0.885 |
| different-day | 0.698 | 0.846 | 0.765 | 0.803 |

## 3. FROC (different-day): FP per undamaged frame vs recall

| conf | FP/undamaged frame | recall | precision |
|---|---|---|---|
| 0.05 | 1.855 | 0.962 | 0.439 |
| 0.1 | 1.055 | 0.942 | 0.583 |
| 0.2 | 0.63 | 0.904 | 0.662 |
| 0.3 | 0.46 | 0.846 | 0.733 |
| 0.4 | 0.315 | 0.75 | 0.78 |
| 0.5 | 0.175 | 0.712 | 0.841 |
| 0.6 | 0.085 | 0.673 | 0.946 |
| 0.7 | 0.04 | 0.615 | 0.941 |
| 0.8 | 0.0 | 0.385 | 1.0 |

> Passing these rules out the cheapest cheating (background/artifact keying) and characterises behaviour. It does NOT prove real-damage performance — the damage is still synthetic. Real labelled damage remains required.