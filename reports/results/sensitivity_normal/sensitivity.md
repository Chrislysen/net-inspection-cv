# Sensitivity sweep — `yolo_damage_seg_gpu.pt`

Damage: `data/processed/diffday_composite/images/test` (30 frames) · undamaged: `data/processed/solaqua_diffday` (200 frames). Lower conf = more sensitive.

| conf | damage recall | precision | F1 | undamaged FP rate |
|---|---|---|---|---|
| 0.05 | 0.962 | 0.943 | 0.952 | 8% |
| 0.1 | 0.962 | 0.962 | 0.962 | 4% |
| 0.15 | 0.962 | 1.0 | 0.98 | 2% |
| 0.2 | 0.962 | 1.0 | 0.98 | 2% |
| 0.3 | 0.962 | 1.0 | 0.98 | 0% |
| 0.4 | 0.962 | 1.0 | 0.98 | 0% |
| 0.5 | 0.962 | 1.0 | 0.98 | 0% |
| 0.6 | 0.962 | 1.0 | 0.98 | 0% |

> Sensitivity is a dial, not a fact: lower threshold catches more (faint) damage but raises false alarms. Damage is synthetic — pick the operating point with stakeholders on REAL labelled data.