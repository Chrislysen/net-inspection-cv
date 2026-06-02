# Adversarial evaluation — `yolo`

## 1. False positives on REAL UNDAMAGED net (no damage present → every detection is a false alarm)

| Frame set | Frames | Detections | Mean/frame | FP frame rate |
|---|---|---|---|---|
| bag1 (train backgrounds) | 38 | 0 | 0.0 | 0% |
| bag2 (same site, other clip) | 120 | 0 | 0.0 | 0% |
| different DAY | 200 | 24 | 0.12 | 11% |

## 2. Damage recall by background distance (composited damage)

| Background | Precision | Recall | F1 | AP |
|---|---|---|---|---|
| in-clip | 0.939 | 0.979 | 0.958 | 0.97 |
| cross-clip (bag2) | 0.958 | 0.967 | 0.962 | 0.96 |
| different-day | 0.962 | 0.962 | 0.962 | 0.96 |

## 3. FROC (different-day): FP per undamaged frame vs recall

| conf | FP/undamaged frame | recall | precision |
|---|---|---|---|
| 0.05 | 0.25 | 0.962 | 0.862 |
| 0.1 | 0.21 | 0.962 | 0.962 |
| 0.2 | 0.14 | 0.962 | 0.962 |
| 0.3 | 0.09 | 0.962 | 0.962 |
| 0.4 | 0.055 | 0.962 | 0.98 |
| 0.5 | 0.035 | 0.962 | 1.0 |
| 0.6 | 0.01 | 0.962 | 1.0 |
| 0.7 | 0.01 | 0.962 | 1.0 |
| 0.8 | 0.0 | 0.923 | 1.0 |

> Passing these rules out the cheapest cheating (background/artifact keying) and characterises behaviour. It does NOT prove real-damage performance — the damage is still synthetic. Real labelled damage remains required.