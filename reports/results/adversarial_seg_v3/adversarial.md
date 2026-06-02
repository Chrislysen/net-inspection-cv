# Adversarial evaluation — `yolo`

## 1. False positives on REAL UNDAMAGED net (no damage present → every detection is a false alarm)

| Frame set | Frames | Detections | Mean/frame | FP frame rate |
|---|---|---|---|---|
| bag1 (train backgrounds) | 38 | 0 | 0.0 | 0% |
| bag2 (same site, other clip) | 120 | 3 | 0.025 | 2% |
| different DAY | 200 | 36 | 0.18 | 18% |

## 2. Damage recall by background distance (composited damage)

| Background | Precision | Recall | F1 | AP |
|---|---|---|---|---|
| in-clip | 0.938 | 0.957 | 0.947 | 0.95 |
| cross-clip (bag2) | 0.963 | 0.976 | 0.97 | 0.97 |
| different-day | 0.893 | 0.962 | 0.926 | 0.959 |

## 3. FROC (different-day): FP per undamaged frame vs recall

| conf | FP/undamaged frame | recall | precision |
|---|---|---|---|
| 0.05 | 0.43 | 0.962 | 0.82 |
| 0.1 | 0.32 | 0.962 | 0.833 |
| 0.2 | 0.21 | 0.962 | 0.893 |
| 0.3 | 0.165 | 0.962 | 0.893 |
| 0.4 | 0.12 | 0.962 | 0.926 |
| 0.5 | 0.08 | 0.962 | 0.926 |
| 0.6 | 0.035 | 0.942 | 0.961 |
| 0.7 | 0.015 | 0.942 | 0.98 |
| 0.8 | 0.0 | 0.827 | 1.0 |

> Passing these rules out the cheapest cheating (background/artifact keying) and characterises behaviour. It does NOT prove real-damage performance — the damage is still synthetic. Real labelled damage remains required.