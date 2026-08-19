# Model artifacts — provenance, licensing, and honest scope

These are **prototype** model artifacts committed so the repo is runnable without
re-training/re-downloading. They are **not** validated damage detectors.

## Files

| File | What it is |
|---|---|
| `yolo_damage_v1.pt` | YOLOv8n detector — the most ROBUST model (0-1% undamaged FP, F1 0.97 different-day) |
| `yolo_damage_seg_v2.pt` | YOLOv8n-seg (masks), single-clip; regressed out-of-distribution (31% different-day FP) — kept as the documented "before" of the closed-loop fix |
| `yolo_damage_seg_v3.pt` | YOLOv8n-seg trained on DIVERSE multi-clip backgrounds; recovered most of v2's OOD gap (different-day FP 31%->18%, recall F1 0.77->0.91). **Best segmenter.** See reports/results/adversarial_seg_v3/ |
| `yolo_damage_seg_v4.pt` | YOLOv8n-seg, multi-clip + STRONG photometric augmentation + more negatives. Tested the hypothesis that simulated day-to-day jitter would shrink the residual OOD gap; it did NOT (different-day FP 22% vs v3's 18%; better in-distribution masks, no better OOD). Kept as the honest negative result. See reports/results/adversarial_seg_v4/ |
| `yolo_damage_seg_gpu.pt` | **YOLOv8s-seg (bigger), trained on 3 real clips on an A100.** On the 200-frame different day: best **recall (0.98)** of any model, and FP **11%** — a real improvement over the nano segmenters' 18% (capacity + real data), though not the detector's 1%. (An earlier "1% FP" was an 80-frame sampling artifact, corrected on the larger sample.) The high-recall option in the precision/recall trade-off. Still synthetic damage — proxy, not validated real-damage performance. See reports/results/adversarial_seg_gpu/ |
| `yolo_damage_v1_training_args.yaml` / `_training_results.csv` | training config + per-epoch metrics |
| `anomaly_normal_net.npz` | patch-feature Mahalanobis "normal net" anomaly model (weak, F1 0.12) |
| `patchcore_normal_net.npz` | PatchCore deep-feature anomaly model (label-free, F1 0.78) |
| `patchcore_resnet18.npz` | PatchCore with an ImageNet-**supervised** ResNet18 backbone — the baseline arm of the self-supervised-vs-supervised ablation (`reports/results/ssl_dino/`) |
| `patchcore_dino_vits14.npz` | PatchCore with a **self-supervised** DINOv2 ViT-S/14 backbone — the SSL arm of the same ablation (image-level AUROC 1.00 in-clip / 0.96 different-day; cleaner false-alarm behaviour than ResNet18) |
| `ssl_resnet18_solaqua.pt` | ResNet18 backbone **SimCLR-pretrained from scratch on 508 unlabelled SOLAQUA training-day frames** (`scripts/pretrain_ssl.py`). Domain SSL proof-of-concept; underperforms transfer at this data scale (see reports/results/ssl_dino/). NOT pretrained on the held-out day. |
| `patchcore_ssl_solaqua.npz` | PatchCore using the SOLAQUA-SimCLR backbone above (requires `ssl_resnet18_solaqua.pt`). The domain-SSL arm of the 3-way backbone ablation. |

## How they were produced

* **`yolo_damage_v1.pt`** — fine-tuned from Ultralytics `yolov8n.pt` on
  **synthetic damage composited onto real SOLAQUA net frames** (`src/netinspect/compose.py`).
  The *backgrounds* are real; the *damage* is synthetic (one generator, identical
  distribution in train and test). Reported F1 ≈ 0.97 in-clip / ≈ 0.98 on a
  held-out clip is a strong **relative/proxy** result, **NOT** validated
  real-damage performance. Cross-clip = same site/camera, not cross-site.
* **`anomaly_normal_net.npz`** — fitted on ~38 real undamaged SOLAQUA frames. It
  flags *deviation from normal net* (biofouling/lighting too), not confirmed damage.

## Licensing — READ THIS BEFORE USING THE WEIGHTS COMMERCIALLY

**The repository is public, and it is MIT only in part.** An earlier version of
this notice said these artifacts were committed "only because this repo is
private" and deferred compliance until public distribution. That condition has
already occurred, so the obligations below are live now, not later.

| artifact | licence that governs it |
|---|---|
| all source code in `src/`, `scripts/`, `web/` | **MIT** |
| `yolo_damage_v1.pt`, `yolo_damage_seg_*.pt` | derived from Ultralytics YOLOv8 → **AGPL-3.0**, and trained on SOLAQUA → **CC BY-SA 4.0** |
| `patchcore_*.npz`, `anomaly_normal_net.npz`, `ssl_resnet18_solaqua.pt` | fitted on SOLAQUA frames → **CC BY-SA 4.0** |

Two consequences that matter to anyone evaluating this for commercial use:

**AGPL-3.0 is viral over a network.** Ultralytics licenses YOLOv8 under
AGPL-3.0, and offering an AGPL work as a network service triggers the source
disclosure obligation for the combined work — which is exactly what
`netinspect serve` does. Most corporate legal teams block AGPL for this reason.
Ultralytics sells a commercial licence that removes it.

**CC BY-SA 4.0 requires attribution and share-alike.** SOLAQUA is © SINTEF
Ocean, CC BY-SA 4.0. Anything derived from those frames — including model
weights fitted on them — carries attribution and share-alike obligations.

### The AGPL-free path is implemented, not just described

`netinspect` ships a second detector built on torchvision (BSD-3-Clause). No
Ultralytics import exists anywhere in `src/netinspect/permissive_baseline.py`,
and a test parses the module's imports to keep it that way.

Measured against the AGPL model on the same held-out split, same threshold:
`yolo` 0% false alarms / 100% recall, `permissive` 0% / 88% (21 of 24 damaged
frames, 29 frames total, 12 CPU epochs). The gate passes it. Ultralytics remains
the better training stack; this trades some recall for a licence you can deploy,
and the trade is a number rather than a claim.

`models/permissive_v1.pt` is trained on the SOLAQUA-derived composite set, so it
still inherits CC BY-SA 4.0 — but it carries **no AGPL obligation**. For an
artifact free of both, retrain on footage you own.

### If you want a completely clean licence position

The pipeline is the deliverable; the weights are a demonstration. Retrain from a
permissively-licensed detector on your own footage and none of the above
applies to the result:

```
netinspect onboard ./your_footage --out data/yoursite
netinspect train --data data/yoursite/dataset.yaml
netinspect gate  --data data/yoursite --weights runs/detect/train/weights/best.pt
```

This is not legal advice. It is a statement of what the upstream licences say,
so that the question reaches your counsel before it reaches your product.

## Reproduce

```
python scripts/fetch_solaqua.py --smallest-video --frames-out data/processed/solaqua_frames_dense --every-n 5 --max-frames 200
python scripts/make_real_dataset.py --frames data/processed/solaqua_frames_dense --out data/processed/real_composite
python scripts/train_yolo.py --data data/processed/real_composite/dataset.yaml --epochs 60 --imgsz 480
```
