# Adversarial evaluation — `yolo`

## 1. False positives on REAL UNDAMAGED net (no damage present → every detection is a false alarm)

| Frame set | Frames | Detections | Mean/frame | FP frame rate |
|---|---|---|---|---|
| bag1 (train backgrounds) | 38 | 0 | 0.0 | 0% |
| bag2 (same site, other clip) | 120 | 0 | 0.0 | 0% |
| bag3 (same DAY, third clip) | 199 | 107 | 0.538 | 31% |
| different DAY | 200 | 2 | 0.01 | 1% |

## 2. Damage recall by background distance (composited damage)

| Background | Precision | Recall | F1 | AP |
|---|---|---|---|---|
| in-clip | 0.958 | 0.979 | 0.968 | 0.97 |
| cross-clip (bag2) | 0.981 | 0.976 | 0.979 | 0.97 |
| different-day | 0.815 | 0.423 | 0.557 | 0.426 |

## 3. FROC (different-day): FP per undamaged frame vs recall

| conf | FP/undamaged frame | recall | precision |
|---|---|---|---|
| 0.05 | 0.135 | 0.442 | 0.657 |
| 0.1 | 0.035 | 0.423 | 0.815 |
| 0.2 | 0.02 | 0.423 | 0.815 |
| 0.3 | 0.01 | 0.423 | 0.846 |
| 0.4 | 0.0 | 0.423 | 0.88 |
| 0.5 | 0.0 | 0.423 | 0.917 |
| 0.6 | 0.0 | 0.423 | 0.957 |
| 0.7 | 0.0 | 0.423 | 0.957 |
| 0.8 | 0.0 | 0.423 | 1.0 |

> Passing these rules out the cheapest cheating (background/artifact keying) and characterises behaviour. It does NOT prove real-damage performance — the damage is still synthetic. Real labelled damage remains required.