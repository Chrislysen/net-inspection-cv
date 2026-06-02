# Adversarial evaluation — `yolo`

## 1. False positives on REAL UNDAMAGED net (no damage present → every detection is a false alarm)

| Frame set | Frames | Detections | Mean/frame | FP frame rate |
|---|---|---|---|---|
| bag1 (train backgrounds) | 38 | 0 | 0.0 | 0% |
| bag2 (same site, other clip) | 120 | 0 | 0.0 | 0% |
| different DAY | 200 | 52 | 0.26 | 18% |

## 2. Damage recall by background distance (composited damage)

| Background | Precision | Recall | F1 | AP |
|---|---|---|---|---|
| in-clip | 0.957 | 0.957 | 0.957 | 0.95 |
| cross-clip (bag2) | 0.936 | 0.972 | 0.954 | 0.97 |
| different-day | 0.862 | 0.962 | 0.909 | 0.958 |

## 3. FROC (different-day): FP per undamaged frame vs recall

| conf | FP/undamaged frame | recall | precision |
|---|---|---|---|
| 0.05 | 0.34 | 0.962 | 0.781 |
| 0.1 | 0.32 | 0.962 | 0.806 |
| 0.2 | 0.28 | 0.962 | 0.833 |
| 0.3 | 0.235 | 0.962 | 0.862 |
| 0.4 | 0.195 | 0.962 | 0.877 |
| 0.5 | 0.16 | 0.962 | 0.893 |
| 0.6 | 0.105 | 0.962 | 0.926 |
| 0.7 | 0.075 | 0.962 | 0.98 |
| 0.8 | 0.005 | 0.846 | 1.0 |

> Passing these rules out the cheapest cheating (background/artifact keying) and characterises behaviour. It does NOT prove real-damage performance — the damage is still synthetic. Real labelled damage remains required.