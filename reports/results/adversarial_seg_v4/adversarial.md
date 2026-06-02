# Adversarial evaluation — `yolo`

## 1. False positives on REAL UNDAMAGED net (no damage present → every detection is a false alarm)

| Frame set | Frames | Detections | Mean/frame | FP frame rate |
|---|---|---|---|---|
| bag1 (train backgrounds) | 38 | 0 | 0.0 | 0% |
| bag2 (same site, other clip) | 120 | 0 | 0.0 | 0% |
| different DAY | 80 | 28 | 0.35 | 22% |

## 2. Damage recall by background distance (composited damage)

| Background | Precision | Recall | F1 | AP |
|---|---|---|---|---|
| in-clip | 0.957 | 0.957 | 0.957 | 0.95 |
| cross-clip (bag2) | 0.936 | 0.972 | 0.954 | 0.97 |
| different-day | 0.785 | 0.976 | 0.87 | 0.968 |

## 3. FROC (different-day): FP per undamaged frame vs recall

| conf | FP/undamaged frame | recall | precision |
|---|---|---|---|
| 0.05 | 0.45 | 0.984 | 0.691 |
| 0.1 | 0.412 | 0.976 | 0.725 |
| 0.2 | 0.35 | 0.976 | 0.752 |
| 0.3 | 0.312 | 0.976 | 0.8 |
| 0.4 | 0.263 | 0.976 | 0.821 |
| 0.5 | 0.237 | 0.976 | 0.849 |
| 0.6 | 0.15 | 0.969 | 0.918 |
| 0.7 | 0.075 | 0.953 | 0.945 |
| 0.8 | 0.0 | 0.929 | 0.992 |

> Passing these rules out the cheapest cheating (background/artifact keying) and characterises behaviour. It does NOT prove real-damage performance — the damage is still synthetic. Real labelled damage remains required.