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
| in-clip | 0.958 | 0.979 | 0.968 | 0.97 |
| cross-clip (bag2) | 0.981 | 0.976 | 0.979 | 0.97 |
| different-day | 0.976 | 0.976 | 0.976 | 0.97 |

## 3. FROC (different-day): FP per undamaged frame vs recall

| conf | FP/undamaged frame | recall | precision |
|---|---|---|---|
| 0.05 | 0.163 | 0.984 | 0.791 |
| 0.1 | 0.087 | 0.984 | 0.906 |
| 0.2 | 0.025 | 0.976 | 0.947 |
| 0.3 | 0.013 | 0.976 | 0.984 |
| 0.4 | 0.0 | 0.969 | 0.984 |
| 0.5 | 0.0 | 0.969 | 0.984 |
| 0.6 | 0.0 | 0.969 | 0.992 |
| 0.7 | 0.0 | 0.961 | 0.992 |
| 0.8 | 0.0 | 0.921 | 1.0 |

> Passing these rules out the cheapest cheating (background/artifact keying) and characterises behaviour. It does NOT prove real-damage performance — the damage is still synthetic. Real labelled damage remains required.