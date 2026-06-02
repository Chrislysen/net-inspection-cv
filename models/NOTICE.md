# Model artifacts — provenance, licensing, and honest scope

These are **prototype** model artifacts committed so the repo is runnable without
re-training/re-downloading. They are **not** validated damage detectors.

## Files

| File | What it is |
|---|---|
| `yolo_damage_v1.pt` | YOLOv8n detector — the most ROBUST model (0-1% undamaged FP, F1 0.97 different-day) |
| `yolo_damage_seg_v2.pt` | YOLOv8n-seg (masks), single-clip; regressed out-of-distribution (31% different-day FP) — kept as the documented "before" of the closed-loop fix |
| `yolo_damage_seg_v3.pt` | YOLOv8n-seg trained on DIVERSE multi-clip backgrounds; recovered most of v2's OOD gap (different-day FP 31%->18%, recall F1 0.77->0.91). See reports/results/adversarial_seg_v3/ |
| `yolo_damage_v1_training_args.yaml` / `_training_results.csv` | training config + per-epoch metrics |
| `anomaly_normal_net.npz` | patch-feature Mahalanobis "normal net" anomaly model (weak, F1 0.12) |
| `patchcore_normal_net.npz` | PatchCore deep-feature anomaly model (label-free, F1 0.78) |

## How they were produced

* **`yolo_damage_v1.pt`** — fine-tuned from Ultralytics `yolov8n.pt` on
  **synthetic damage composited onto real SOLAQUA net frames** (`src/netinspect/compose.py`).
  The *backgrounds* are real; the *damage* is synthetic (one generator, identical
  distribution in train and test). Reported F1 ≈ 0.97 in-clip / ≈ 0.98 on a
  held-out clip is a strong **relative/proxy** result, **NOT** validated
  real-damage performance. Cross-clip = same site/camera, not cross-site.
* **`anomaly_normal_net.npz`** — fitted on ~38 real undamaged SOLAQUA frames. It
  flags *deviation from normal net* (biofouling/lighting too), not confirmed damage.

## Licensing (read before making this repo public)

* Prototype code: **MIT**.
* `yolo_damage_v1.pt` derives from Ultralytics YOLOv8 (**AGPL-3.0**) and from
  SOLAQUA data (**CC BY-SA 4.0**). The anomaly model derives from SOLAQUA data
  (**CC BY-SA 4.0**). These artifacts are committed here only because this repo is
  **private**. Before any public distribution or networked service use, comply
  with AGPL-3.0 (Ultralytics) and CC BY-SA 4.0 (SOLAQUA: attribution + share-alike),
  or retrain from a permissively-licensed base on your own licensed data.

## Reproduce

```
python scripts/fetch_solaqua.py --smallest-video --frames-out data/processed/solaqua_frames_dense --every-n 5 --max-frames 200
python scripts/make_real_dataset.py --frames data/processed/solaqua_frames_dense --out data/processed/real_composite
python scripts/train_yolo.py --data data/processed/real_composite/dataset.yaml --epochs 60 --imgsz 480
```
