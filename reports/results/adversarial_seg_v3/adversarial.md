# Adversarial evaluation — `yolo`

## 1. False positives on REAL UNDAMAGED net (no damage present → every detection is a false alarm)

| Frame set | Frames | Detections | Mean/frame | FP frame rate |
|---|---|---|---|---|
| bag1 (train backgrounds) | 38 | 0 | 0.0 | 0% |
| bag2 (same site, other clip) | 120 | 3 | 0.025 | 2% |
| different DAY | 80 | 15 | 0.188 | 18% |

## 2. Damage recall by background distance (composited damage)

| Background | Precision | Recall | F1 | AP |
|---|---|---|---|---|
| in-clip | 0.938 | 0.957 | 0.947 | 0.95 |
| cross-clip (bag2) | 0.963 | 0.976 | 0.97 | 0.97 |
| different-day | 0.85 | 0.984 | 0.912 | 0.972 |

## 3. FROC (different-day): FP per undamaged frame vs recall

| conf | FP/undamaged frame | recall | precision |
|---|---|---|---|
| 0.05 | 0.312 | 0.984 | 0.74 |
| 0.1 | 0.225 | 0.984 | 0.791 |
| 0.2 | 0.212 | 0.984 | 0.833 |
| 0.3 | 0.175 | 0.984 | 0.88 |
| 0.4 | 0.125 | 0.984 | 0.912 |
| 0.5 | 0.075 | 0.976 | 0.925 |
| 0.6 | 0.025 | 0.961 | 0.953 |
| 0.7 | 0.0 | 0.929 | 0.975 |
| 0.8 | 0.0 | 0.827 | 0.981 |

> Passing these rules out the cheapest cheating (background/artifact keying) and characterises behaviour. It does NOT prove real-damage performance — the damage is still synthetic. Real labelled damage remains required.