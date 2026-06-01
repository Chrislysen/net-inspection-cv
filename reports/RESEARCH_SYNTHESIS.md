# Research synthesis & decisions

This document records how external **deep research** (multiple ChatGPT
deep-research reports, generated from `reports/DEEP_RESEARCH_PROMPT.md`) was used
to improve the project — and, just as importantly, **what was rejected and why**.
Critically filtering an automated research dump is itself part of the engineering;
this trail shows the reasoning rather than blindly executing recommendations.

## How the research was used

The reports were thorough and well-structured, but (a) written against an
**outdated snapshot** — roughly half their "high-priority TODOs" were already
implemented — and (b) contained a few claims that the prompt itself flagged for
verification. Each recommendation was sorted into **already-done**, **adopted**,
or **rejected (with rationale)**.

## Decision log

| Recommendation (from the research) | Decision | Rationale |
|---|---|---|
| COCO→YOLO adapter for real labelled data | **Already done** | `coco.py` + `convert_coco.py` (+ tests). A dataset-specific fetch script is cosmetic. |
| ONNX export / INT8 / TensorRT | **Already done / documented** | `export_onnx.py` exports + benchmarks; INT8/TensorRT need the target device, so documented honestly, not faked. |
| Temporal smoothing to cut false alarms | **Already done** | `temporal.py` (−70% transient FP on real video). |
| Tests for adapters/metrics; CI matrix | **Already done** | `test_coco.py`, per-class eval; CI already runs 3.11/3.12. |
| COCO-style mAP@[.5:.95] | **Adopted** | Added `evaluate.coco_map` + test; surfaced in `compare_methods`. Standard detection metric. |
| Pinned lockfile, pre-commit, lint, CI badge | **Adopted** | `requirements-lock.txt`, `ruff` config (clean), `.pre-commit-config.yaml`, a CI lint job, README badges. Real reproducibility value. |
| Train on more **diverse real backgrounds** (domain coverage) | **Adopted** | Built a multi-clip composited dataset (bag1+bag2) and re-trained the seg model to test whether it fixes the out-of-distribution regression (see `reports/results/`). This is the runnable part of "sim-to-real". |
| Use SeaClear/TrashCan marine-debris datasets as a **net-damage proxy** ("treat debris as damage") | **Rejected** | Debris on the seabed is not a hole in a net; this would teach the model to find plastic, not damage. The COCO adapter exists for *real labelled net damage*, not as a debris proxy. (Debris data is fine only as an optional transfer/pretraining or a real-image smoke test, clearly labelled as such.) |
| Specific dataset sizes/licences (e.g. "SeaClear 8610 imgs, CC BY 4.0", TrashCan licence) | **Flagged, not adopted as fact** | These are exactly the "(likely)" claims the prompt told the model to verify. Not repeated as fact here; must be checked against the actual dataset pages before any use. |
| Full MLOps stack (Prometheus/Grafana, drift dashboards, GPU CI, SLO alerting) | **Rejected (for now)** | Production-theatre for a prototype with no real labelled data. The honest gate is data, not dashboards. Documented as a real-deployment next step instead. |
| Self-supervised (DINO/MAE) pretraining on unlabelled SOLAQUA | **Deferred (documented)** | Genuinely promising, but full SSL pretraining is impractical on the CPU-only dev box. The executable proxy — training on diverse real backgrounds — was done instead; SSL is recorded as the next step once GPU/real data is available. |

## The one non-negotiable

Every report eventually asked for "production-level performance". The honest
answer is unchanged and stated throughout the repo: **all damage is synthetic
(one generator), so no validated real-world damage-detection performance is
claimed.** What *was* improved is real and measurable: out-of-distribution
robustness via diverse-background training, reproducibility, and metrics. The
single missing ingredient remains **real labelled net-damage footage**, and the
pipeline is the drop-in slot for it.

## ScaleAQ interview talking points (grounded in this repo)

- **"How would you approach net-damage detection?"** — Inspect/understand the
  data first; start with an explainable classical baseline to gauge difficulty;
  move to a supervised model once labels exist; evaluate adversarially and design
  for human-in-the-loop. This repo *is* that workflow, end to end.
- **"What's the hard part?"** — Not the model — the **data and the sim-to-real
  gap**. On real undamaged SOLAQUA net the naive heuristic false-alarms on ~76% of
  frames; the supervised model only looks great because its damage is synthetic.
  I built an **adversarial evaluation** specifically to expose that, and it even
  caught one of my own models regressing out-of-distribution.
- **"Show me you don't overclaim."** — The whole repo refuses to report
  real-world accuracy it can't prove, and labels every number as proxy. That
  honesty is the point: a model trusted in the water without validation is a fish-
  escape risk.
- **"What would you need from us?"** — A few hundred **labelled real-damage
  frames** across sites/seasons/cameras, a labelling guideline, and an agreed
  false-positive/false-negative trade-off. Then `train_yolo.py` +
  `compare_methods.py` + the adversarial suite turn every proxy number into a real
  evaluation, unchanged.
- **Relevant background** — prior YOLOv8 object-detection pipeline work
  (data→train→ONNX→eval) and an ML-systems mindset (latency/constraints, failure
  tracking) map directly onto on-ROV deployment and honest evaluation.

## Pointers
- Full technical write-up: [`SCALEAQ_PROTOTYPE_REPORT.md`](SCALEAQ_PROTOTYPE_REPORT.md)
- Result tables/figures: [`results/`](results/)
- The research prompt itself: [`DEEP_RESEARCH_PROMPT.md`](DEEP_RESEARCH_PROMPT.md)
