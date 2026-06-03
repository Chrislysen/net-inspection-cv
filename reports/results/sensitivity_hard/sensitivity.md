# Sensitivity sweep — `yolo_damage_seg_gpu.pt`

Damage: `data/processed/hard_composite/images/test` (30 frames) · undamaged: `data/processed/solaqua_diffday` (200 frames). Lower conf = more sensitive.

| conf | damage recall | precision | F1 | undamaged FP rate |
|---|---|---|---|---|
| 0.05 | 0.92 | 0.958 | 0.939 | 8% |
| 0.1 | 0.9 | 0.978 | 0.938 | 4% |
| 0.15 | 0.9 | 0.978 | 0.938 | 2% |
| 0.2 | 0.9 | 0.978 | 0.938 | 2% |
| 0.3 | 0.9 | 1.0 | 0.947 | 0% |
| 0.4 | 0.88 | 1.0 | 0.936 | 0% |
| 0.5 | 0.82 | 1.0 | 0.901 | 0% |
| 0.6 | 0.8 | 1.0 | 0.889 | 0% |

> Sensitivity is a dial, not a fact: lower threshold catches more (faint) damage but raises false alarms. Damage is synthetic — pick the operating point with stakeholders on REAL labelled data.