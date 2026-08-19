# Model card — net-damage detector (prototype)

Following the spirit of standard model cards. **This documents a prototype model
that is NOT validated for real-world use.**

## Model details
- **Name / version:** `yolo_damage_v1.pt` (YOLOv8n detection), `yolo_damage_seg_v2/v3` (YOLOv8n-seg).
- **Type:** single-class ("damage") object detector / instance segmenter.
- **Inputs:** RGB underwater net-pen frames. **Outputs:** boxes/masks + confidence.
- **Also exported:** `yolo_damage_v1.onnx` (torch-free inference, parity-verified).
- **Date / owner:** prototype, 2026; see `models/NOTICE.md` for provenance.

## Intended use
- **Intended:** *decision support* for human net inspection — surface candidate
  damage regions in offline review of ROV/camera footage, with a human confirming.
- **Intended users:** aquaculture inspection engineers/operators, with ML support.

## Out-of-scope / prohibited use
- **Not** an autonomous authority on net integrity. **Do not** use it to decide a
  net is safe without human review — a false "net is fine" is a fish-escape risk.
- Not validated on real damage, other species/net types, or unseen sites.

## Training data
- **Backgrounds:** real SOLAQUA ROV frames of *undamaged* net (SINTEF, CC BY-SA 4.0).
- **Damage:** **synthetic**, composited onto those frames (`compose.py`) — holes/tears
  with see-through fill, frayed edges, plus unlabelled hard negatives. The damage
  *appearance* is modelled, not observed.
- v3 adds multi-clip backgrounds (bag1+bag2) for out-of-distribution robustness.

## Evaluation
- Metrics: class-agnostic P/R/F1, COCO mAP@[.5:.95], FROC, image-level, mask IoU.
- Proxy results (composite test): F1 ≈ 0.97 (det) **in-clip**; on a different day
  the same model measures F1 0.56 (recall 0.42).
- **Adversarial** check on all four undamaged clips: det v1 fires on **11% of 557
  real frames** — 0% bag1, 0% bag2, **31% bag3**, 1% different-day. An earlier
  three-clip sample that omitted bag3 reported ~0%, and that number is retracted.
  The suite also caught the seg v2 model regressing to 35.5% different-day FP.
  See `reports/results/`.
- **All on synthetic damage** — see Limitations.

## Limitations & risks (read this)
- **No validated real-world performance.** The 0.97 is on synthetic damage from one
  generator; the model may key on compositing cues rather than real damage.
- Underwater domain shift (turbidity, light, biofouling, camera) is unmeasured.
- Asymmetric error costs: a missed defect (FN) is far more costly (escape) than a
  false alarm (FP); the operating point must be set with stakeholders.
- Mitigations in repo: adversarial evaluation, temporal confirmation, human-in-the-loop.

## Licensing / provenance
- Code: MIT. Weights derive from Ultralytics YOLOv8 (**AGPL-3.0**) and SOLAQUA
  (**CC BY-SA 4.0**). **This repository is public**, so treat those terms as
  binding rather than deferred — an earlier version of this line said the weights
  were committed "only because the repo is private", which is no longer true and
  was never a licence exemption. An AGPL-free path is implemented
  (`netinspect.permissive_baseline`, the `permissive` extra); verify any build
  with `netinspect sbom --fail-on copyleft`. See `models/NOTICE.md`.

## How to validate before deployment
Provide real labelled damage (see `DATA_COLLECTION.md`), retrain with the existing
scripts, and evaluate **by held-out site**. Only then does this card get real numbers.
