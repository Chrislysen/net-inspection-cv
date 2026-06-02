# Adversarial evaluation — `yolo`

## 1. False positives on REAL UNDAMAGED net (no damage present → every detection is a false alarm)

| Frame set | Frames | Detections | Mean/frame | FP frame rate |
|---|---|---|---|---|
| bag1 (train backgrounds) | 38 | 0 | 0.0 | 0% |
| bag2 (same site, other clip) | 120 | 0 | 0.0 | 0% |
| different DAY | 80 | 1 | 0.013 | 1% |

## 2. Damage recall by background distance (composited damage)

| Background | Precision | Recall | F1 | AP |
|---|---|---|---|---|
| in-clip | 0.939 | 0.979 | 0.958 | 0.97 |
| cross-clip (bag2) | 0.958 | 0.967 | 0.962 | 0.96 |
| different-day | 0.926 | 0.992 | 0.958 | 0.989 |

## 3. FROC (different-day): FP per undamaged frame vs recall

| conf | FP/undamaged frame | recall | precision |
|---|---|---|---|
| 0.05 | 0.138 | 0.992 | 0.696 |
| 0.1 | 0.075 | 0.992 | 0.818 |
| 0.2 | 0.013 | 0.992 | 0.9 |
| 0.3 | 0.013 | 0.992 | 0.933 |
| 0.4 | 0.013 | 0.984 | 0.954 |
| 0.5 | 0.013 | 0.984 | 0.977 |
| 0.6 | 0.0 | 0.984 | 0.984 |
| 0.7 | 0.0 | 0.969 | 0.984 |
| 0.8 | 0.0 | 0.961 | 1.0 |

> Passing these rules out the cheapest cheating (background/artifact keying) and characterises behaviour. It does NOT prove real-damage performance — the damage is still synthetic. Real labelled damage remains required.