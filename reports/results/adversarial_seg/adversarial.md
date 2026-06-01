# Adversarial evaluation — `yolo`

## 1. False positives on REAL UNDAMAGED net (no damage present → every detection is a false alarm)

| Frame set | Frames | Detections | Mean/frame | FP frame rate |
|---|---|---|---|---|
| bag1 (train backgrounds) | 38 | 2 | 0.053 | 5% |
| bag2 (same site, other clip) | 120 | 5 | 0.042 | 3% |
| different DAY | 80 | 40 | 0.5 | 31% |

## 2. Damage recall by background distance (composited damage)

| Background | Precision | Recall | F1 | AP |
|---|---|---|---|---|
| in-clip | 0.93 | 0.851 | 0.889 | 0.85 |
| cross-clip (bag2) | 0.931 | 0.892 | 0.911 | 0.885 |
| different-day | 0.699 | 0.858 | 0.77 | 0.798 |

## 3. FROC (different-day): FP per undamaged frame vs recall

| conf | FP/undamaged frame | recall | precision |
|---|---|---|---|
| 0.05 | 1.837 | 0.969 | 0.353 |
| 0.1 | 0.875 | 0.945 | 0.533 |
| 0.2 | 0.588 | 0.882 | 0.675 |
| 0.3 | 0.425 | 0.843 | 0.743 |
| 0.4 | 0.237 | 0.732 | 0.823 |
| 0.5 | 0.175 | 0.661 | 0.875 |
| 0.6 | 0.062 | 0.567 | 0.935 |
| 0.7 | 0.0 | 0.449 | 0.983 |
| 0.8 | 0.0 | 0.252 | 0.97 |

> Passing these rules out the cheapest cheating (background/artifact keying) and characterises behaviour. It does NOT prove real-damage performance — the damage is still synthetic. Real labelled damage remains required.