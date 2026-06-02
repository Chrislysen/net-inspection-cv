# Test-time domain normalisation (gray-world WB + CLAHE) — different day

80 undamaged frames; conf=0.25, IoU=0.3.

| Model | FP rate (raw) | FP rate (normalised) | Recall F1 (raw) | Recall F1 (normalised) |
|---|---|---|---|---|
| det v1 | 0% | 28% | 0.98 | 0.856 |
| seg v3 | 9% | 8% | 0.958 | 0.943 |

> Models were trained on RAW frames, so this is test-time-only normalisation (a train/test mismatch). To actually exploit normalisation it must be applied in BOTH training and inference — a retrain. Reported as measured, not assumed.