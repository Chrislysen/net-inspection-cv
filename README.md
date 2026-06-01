# net-inspection-cv

A prototype computer-vision toolkit for **detecting possible damage (holes,
tears, abnormal regions) in aquaculture net imagery**. It is built as an
*exploration framework*, not a finished product: it ingests images/video,
preprocesses underwater footage, runs both a classical OpenCV baseline and a
trainable YOLOv8 baseline, evaluates them honestly when labels exist, and
visualises predictions and failure cases.

> **Status: prototype.** The repository ships with a small *synthetic* dataset
> so the full pipeline runs out of the box, **and** an integration with the real
> **SOLAQUA** ROV dataset (SINTEF). **Synthetic results only verify that the code
> path works.** SOLAQUA is real footage but of **undamaged** nets with **no
> damage labels**, so it is used for real-image preprocessing, false-positive /
> robustness analysis, and anomaly detection — **not** for measuring damage
> detection accuracy. Validated damage-detection numbers still require real,
> *labelled* footage (see
> [What ScaleAQ would need to provide](#what-scaleaq-would-need-to-provide-to-make-this-real)).

---

## Example outputs

| Synthetic demo (prediction vs. ground truth) | Synthetic failure case (merged adjacent defects) |
|---|---|
| ![synthetic overlay](docs/images/synthetic_overlay.jpg) | ![synthetic failure](docs/images/synthetic_failure_match.jpg) |
| Orange = prediction, green = ground truth. | Orange = TP, **blue = missed** GT (a hole + adjacent tear merged into one box). |

| Classical baseline on **real** SOLAQUA net (all detections are false alarms) | Anomaly model heatmap on the same frame |
|---|---|
| ![real false positives](docs/images/real_false_positives.jpg) | ![real anomaly heatmap](docs/images/real_anomaly_heatmap.jpg) |
| The nets are **undamaged**; the heuristic still fires on oblique/dark cells. | Anomaly score (left: boxes, right: heatmap) concentrates on biofouling & lighting, not damage. |

**YOLOv8 trained on synthetic damage composited onto real net, evaluated on a held-out clip:**

| Ground truth (green) vs. YOLO prediction (orange) — cross-clip | Classical (left) vs. YOLO (right) on the same frame |
|---|---|
| ![yolo cross-clip](docs/images/yolo_crossclip.jpg) | ![classical vs yolo](docs/images/classical_vs_yolo.jpg) |

| PatchCore (foundation, label-free): boxes \| anomaly heatmap | Temporal: raw (left) vs. persistence-confirmed (right) |
|---|---|
| ![patchcore](docs/images/patchcore_heatmap.jpg) | ![temporal](docs/images/temporal_raw_vs_confirmed.jpg) |

**Method comparison** (synthetic damage on real backgrounds; IoU 0.30, class-agnostic):

| Method | Labels? | In-clip F1 | Cross-clip F1 (held-out clip) |
|---|---|---|---|
| Hand-crafted anomaly (Mahalanobis) | no | 0.12 | 0.05 |
| Classical (OpenCV heuristic) | no | 0.50 | 0.67 |
| **PatchCore (foundation-model, label-free)** | no | **0.78** | — |
| **YOLOv8 (supervised)** | yes | **0.97** | **0.98** |

**Adversarial "is it cheating?" check** — the trained YOLO on **real undamaged
net** (no damage present, so any detection is a false alarm):

| Frame set | False-positive frame rate |
|---|---|
| bag1 (the backgrounds it trained on) | **0%** |
| bag2 (same site, other clip) | **0%** |
| different day | **1%** |

It almost never fires on undamaged net (incl. its own training backgrounds) and
holds F1≈0.97 across in-clip/cross-clip/different-day — ruling out the cheapest
"keying on background/artifacts" failure. **Temporal confirmation** removes a
further **70%** of transient false alarms on real undamaged video.

> **Read these honestly.** Synthetic metrics only verify the pipeline. SOLAQUA
> frames are real but **undamaged/unlabelled** (false-alarm & anomaly behaviour
> only). The YOLO numbers are on **synthetic damage composited on real
> backgrounds** — the *damage appearance* comes from one generator (same in
> train and test) and the cross-clip set is the same site/camera, so this shows
> "learns this damage model and transfers across clips/backgrounds", **not**
> "detects real damage." Real labelled damage is still required to claim
> real-world performance. Full detail in the [report](reports/SCALEAQ_PROTOTYPE_REPORT.md).

---

## The problem

Fish-farm nets develop holes, tears, deformations and abnormal regions over
time. Detecting them early matters: undetected damage risks fish escape, with
operational, financial and environmental consequences. Today this is largely a
manual review task — an operator or ROV pilot watches underwater camera footage
and judges whether something looks wrong.

The question this prototype explores is: **can computer vision flag suspicious
regions in net footage to support (not replace) human inspection?**

### Why it is hard

This is not a clean image-classification problem. A real system must cope with:

- **Underwater optics** — light attenuation, colour casts, turbidity, backscatter.
- **Variable conditions** — lighting, camera distance/angle, net deformation, motion blur.
- **Confounders** — marine growth/biofouling, ropes, shadows, fish occlusion, background structure.
- **Data scarcity & imbalance** — damage is rare; most frames are intact net.
- **Ambiguous labels** — "damage" is partly a judgement call; annotators disagree.
- **Asymmetric error costs** — false positives waste operator time; false negatives can miss real damage. The acceptable trade-off is a *business* decision, not a default.

These are exactly the reasons the prototype favours **explainable baselines,
careful evaluation, and visual failure analysis** over a single black-box model.

---

## Proposed workflow

The repo is organised around the workflow I would actually follow on the job:

1. **Inspect & understand the data** — counts, sizes, class balance, label sanity (`prepare_data.py`).
2. **Define labels** — agree damage classes and a labelling guideline with domain experts (`configs/`).
3. **Classical baseline first** — a fast, explainable OpenCV method to gauge how separable damage is and to surface data difficulty (`run_classical_baseline.py`).
4. **Train a detection/segmentation baseline** — YOLOv8 / YOLOv8-seg once labelled data exists (`train_yolo.py`).
5. **Evaluate false positives / false negatives** — precision/recall/F1/AP, image-level decisions, confidence sweeps, per-image failure overlays (`evaluate.py`).
6. **Produce a prototype + recommendations** — a report covering assumptions, results, limitations and next steps (`reports/`).

---

## Three compared approaches

| | Classical baseline | Anomaly baseline | ML baseline |
|---|---|---|---|
| **Module** | `classical_baseline.py` | `anomaly.py` | `model_baseline.py` (YOLOv8) |
| **Idea** | Net = regular mesh; damage breaks it. **Darkness** cue + **low-edge-density** cue + a **texture gate** to reject dark-but-textured net. | Learn "normal net" patch statistics; flag Mahalanobis outliers. | Learn damage appearance from labelled examples. |
| **Needs labels?** | No | No (only normal frames) | Yes (YOLO det/seg) |
| **Pros** | Fast, transparent, no training data. Triage / difficulty probe. | No damage labels needed; localises "unusual net". | Learns appearance; far higher ceiling (F1≈0.97 here). |
| **Cons** | Brittle; tops out ~F1 0.5 on real backgrounds. | Flags fouling/lighting too; weak box localiser. | Needs representative labelled data; less interpretable. |

The classical and anomaly methods are **screening / difficulty probes**; the ML
path is the route to real performance once labelled data exists. All three share
one prediction schema and are compared with the same metrics (`compare_methods.py`).

---

## Repository layout

```
net-inspection-cv/
  README.md                      this file
  pyproject.toml                 packaging + optional deps (cv / ml / dev)
  requirements.txt               pinned-ish runtime deps
  configs/
    baseline.yaml                classical + preprocessing + eval params
    yolo_dataset.yaml            YOLO dataset config (placeholder)
  src/netinspect/
    data.py                      dataset discovery, YOLO label parsing, summaries
    preprocess.py                CLAHE, white balance, denoise, resize
    classical_baseline.py        explainable OpenCV anomaly baseline
    model_baseline.py            YOLOv8 detect/segment wrapper (graceful if missing)
    anomaly.py                   normal-net anomaly model (patch Mahalanobis)
    patchcore.py                 foundation-model anomaly (pretrained-CNN PatchCore)
    temporal.py                  IoU tracker — confirm detections that persist
    compose.py                   composite photorealistic damage + hard negatives onto REAL frames
    inference.py                 unified facade over all methods (used everywhere)
    evaluate.py                  detection / segmentation / image-level metrics
    visualize.py                 overlays, comparisons, galleries
    video.py                     video frame extraction
    solaqua.py                   SOLAQUA client + ROS-bag camera & sonar extraction
    synthetic.py                 placeholder data generator (testing only)
    utils.py                     IO, geometry, optional-dependency handling
  scripts/                       CLI entry points (see below)
  streamlit_app.py               interactive viewer (streamlit run streamlit_app.py)
  .github/workflows/ci.yml       CI: run tests on push/PR
  tests/                         pytest: data loading + metrics
  reports/SCALEAQ_PROTOTYPE_REPORT.md
  data/  outputs/  runs/         data, predictions, visualisations, training runs
```

---

## Installation

Python 3.10+ (developed and verified on 3.14). Core data/metric code needs only
NumPy/Pandas/Pillow; OpenCV and Ultralytics are **optional** and the code
degrades gracefully without them.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1            # Windows; use source .venv/bin/activate on Linux/Mac
pip install -e ".[cv,ml,data,serve,dev]"  # cv=OpenCV; ml=ultralytics/torch; data=rosbags; serve=FastAPI/Streamlit
```

If you only want the classical baseline + evaluation: `pip install -e ".[cv,dev]"`.
Extras are modular: `data` (rosbags, for SOLAQUA `.bag`), `ml` (YOLO), `serve`
(API + viewer) are each optional and degrade gracefully when absent.
If `ultralytics` is unavailable, the YOLO scripts print clear guidance instead of crashing.

---

## Quick start — the demo path (synthetic data)

This runs the whole pipeline end-to-end on generated placeholder data:

```powershell
# 1. Generate synthetic net images with injected "damage" (+ YOLO labels)
python scripts/make_demo_data.py --out data/sample --n 10

# 2. Inspect the dataset (counts, sizes, class balance, label errors)
python scripts/prepare_data.py --images data/sample/images --labels data/sample/labels --out data/processed

# 3. Run the classical baseline -> predictions + overlays
python scripts/run_classical_baseline.py --images data/sample/images --out outputs/classical --config configs/baseline.yaml

# 4. Evaluate against ground truth -> metrics + failure overlays
python scripts/evaluate.py --preds outputs/classical/preds.json --images data/sample/images --labels data/sample/labels --out outputs/eval

# 5. Collect report assets (gallery, failure cases, metrics.json) in one command
python scripts/make_report_assets.py
```

> **Reminder:** the synthetic demo only verifies that the pipeline works.
> It does **not** prove real aquaculture performance.

### Running the ML (YOLO) path

```powershell
# Build a YOLO train/val split (also writes a ready-to-use dataset.yaml)
python scripts/prepare_data.py --images data/sample/images --labels data/sample/labels --out data/processed --yolo-split

# Train (needs ultralytics). On real data use more epochs / a larger model.
python scripts/train_yolo.py --data data/processed/yolo/dataset.yaml --epochs 50

# Inference with trained weights, then evaluate the same way as the classical path
python scripts/run_inference.py --images data/processed/images --weights runs/detect/train/weights/best.pt --out outputs/predictions
python scripts/evaluate.py --preds outputs/predictions/preds.json --images data/processed/images --labels data/processed/labels --out outputs/eval
```

### Real data — SOLAQUA (SINTEF)

[SOLAQUA](https://data.sintef.no/product/dp-7141fcd5-0fb8-4be3-b9ce-e5f7f5bb4a58)
([paper](https://arxiv.org/abs/2504.01790), CC BY-SA 4.0) is real ROV footage of
operational Norwegian net pens. The camera data is inside ROS `.bag` files and
the nets are **undamaged with no damage labels**, so we use it honestly: real
preprocessing, false-positive analysis, and anomaly detection.

```powershell
# List files (sizes), then download the smallest video bag and extract ~40 frames
python scripts/fetch_solaqua.py --list
python scripts/fetch_solaqua.py --smallest-video --frames-out data/processed/solaqua_frames

# (a) Robustness / false-positive analysis: nets are undamaged, so every
#     classical-baseline detection is a false alarm. This quantifies that.
python scripts/run_real_analysis.py --images data/processed/solaqua_frames --out outputs/real_analysis --config configs/baseline.yaml

# (b) Anomaly detection: learn "normal net", flag deviations (heatmaps + boxes)
python scripts/train_anomaly.py --images data/processed/solaqua_frames --out outputs/anomaly/model
python scripts/run_anomaly.py --images data/processed/solaqua_frames --model outputs/anomaly/model --out outputs/anomaly
```

Observed on 38 extracted frames (real, undamaged net): the classical baseline
raises a false alarm on **~76% of frames** (≈3 detections/frame), and the
anomaly model's heatmaps light up on **biofouling and lighting hotspots** — both
honest illustrations of why a naive heuristic is insufficient and why a learned
model plus a calibrated FP/FN threshold are needed. These numbers describe
false-alarm behaviour on undamaged net; they are **not** damage-detection
metrics (there is no damage to detect here).

### Train a realistic detector (synthetic damage on real backgrounds) + compare

When real labelled damage isn't available, we composite plausible damage onto
**real** SOLAQUA net frames (real texture/biofouling/lighting; labelled
holes/tears) to get a believable training/eval set — clearly not real damage.

```powershell
# Composite damage onto real frames -> YOLO train/val/test (disjoint backgrounds)
python scripts/make_real_dataset.py --frames data/processed/solaqua_frames_dense --out data/processed/real_composite

# Train YOLOv8 on real backgrounds, then compare all 3 methods on the test split
python scripts/train_yolo.py --data data/processed/real_composite/dataset.yaml --epochs 60 --imgsz 480
python scripts/compare_methods.py --images data/processed/real_composite/images/test \
    --labels data/processed/real_composite/labels/test --config configs/baseline.yaml \
    --anomaly-model outputs/anomaly/model --yolo-weights runs/detect/train/weights/best.pt \
    --out outputs/comparison
```

Measured: **YOLO F1 ≈ 0.97** vs classical **0.50** vs anomaly **0.12**, holding up
on a held-out clip (F1 ≈ 0.98) — see the caveats above and in the report.

### Stronger methods + rigorous evaluation

```powershell
# Foundation-model anomaly (label-free, F1 ~0.78): pretrained-CNN PatchCore
python scripts/train_patchcore.py --images data/processed/solaqua_frames_dense --out models/patchcore_normal_net

# Adversarial "is it cheating?" suite: FP on REAL undamaged net + different-day + FROC
python scripts/adversarial_eval.py --yolo-weights models/yolo_damage_v1.pt --out reports/results/adversarial_yolo

# Temporal confirmation on a CONTIGUOUS sequence: removes ~70% of transient false alarms
python scripts/fetch_solaqua.py --bag <clip>.bag --frames-out data/processed/solaqua_seq --every-n 1 --max-frames 120
python scripts/run_temporal.py --method classical --source data/processed/solaqua_seq --out outputs/temporal --config configs/baseline.yaml
```

Committed result tables: [`reports/results/`](reports/results/) — 4-method
comparison, adversarial eval, temporal.

### Tuning classical false positives

The classical baseline's `max_region_density_ratio` (texture gate) trades recall
for fewer false alarms on real net. Lowering it from the untuned setting cut
false detections on undamaged frames **~46–75%** (config `configs/baseline.yaml`).

### Serve it (HTTP API) and explore it (Streamlit)

Trained prototype models are committed under `models/` (see `models/NOTICE.md` for
provenance/licensing), so the YOLO and anomaly paths run out of the box:

```powershell
# FastAPI inference service using the committed models
python scripts/serve.py --anomaly-model models/anomaly_normal_net --yolo-weights models/yolo_damage_v1.pt
# curl -F file=@frame.jpg "http://localhost:8000/predict?method=yolo"

# Or containerised (serves YOLO + anomaly + classical):
docker build -t net-inspection-cv . ; docker run -p 8000:8000 net-inspection-cv

# Interactive viewer: browse frames, switch methods, live thresholds
streamlit run streamlit_app.py

# Unified batch/video/bag inference for any method
python scripts/infer.py --method yolo --yolo-weights models/yolo_damage_v1.pt \
    --source data/processed/solaqua_frames --out outputs/infer
```

Measured results are committed under [`reports/results/`](reports/results/)
(in-clip and cross-clip method comparisons, real-frame false-positive analysis).

### Multi-modal: multibeam sonar

```powershell
python scripts/fetch_solaqua.py --bag data/raw/solaqua/<clip>.bag --sonar-out data/processed/sonar
```

Sonar (SonoptixECHO) decodes to a 512×512 intensity image — a complementary
modality that sees through turbidity. Experimental; not RGB damage detection.

### Working from video

```powershell
python scripts/extract_frames.py --video data/raw/video.mp4 --out data/processed/frames --fps 2
```

---

## What is implemented vs. placeholder

**Implemented and working:**

- Image/video ingestion, YOLO detection **and** segmentation label parsing, dataset summaries with invalid-label reporting.
- Underwater preprocessing (gray-world white balance, CLAHE, denoise, resize).
- Classical baseline producing boxes + scores + overlays + JSON/CSV.
- YOLOv8 detect/segment training & inference wrapper (graceful when ultralytics is absent).
- Evaluation: IoU-matched precision/recall/F1, VOC-style AP, confidence sweep, image-level "contains damage?", mask IoU, explicit FP/FN lists, failure overlays.
- Visualisation: prediction/GT overlays, colour-coded match overlays, markdown gallery.
- **Real-data ingestion (SOLAQUA)**: public-API client, resumable download, ROS `.bag` camera **and multibeam-sonar** extraction.
- **Two anomaly baselines**: hand-crafted patch-Mahalanobis, and a **foundation-model PatchCore** (pretrained CNN) reaching F1 0.78 label-free (6.5× the hand-crafted one).
- **Realistic training data**: photorealistic damage (seamless blend, frayed fibres) + **hard negatives** composited on real backgrounds (`compose.py`).
- **Four-method comparison** + a **measured improvement story** (classical FP reduction; PatchCore; YOLO F1≈0.97).
- **Adversarial "is it cheating?" evaluation**: false positives on real undamaged net, different-day generalization, FROC curve.
- **Temporal reasoning**: persistence tracking that removes ~70% of transient false alarms on real video.
- **Segmentation** (YOLOv8-seg) for masks; **industrial layer**: unified facade, FastAPI service, Streamlit viewer, batch/video/bag runner, CI.
- Tests (31) for data, metrics, anomaly, compositing, inference, temporal, and PatchCore IO.

**Placeholder / synthetic (clearly marked):**

- `data/sample/` and the `synthetic.py` generator are **procedural placeholders**. Any metric computed on them measures the pipeline, not real-world skill.
- `configs/yolo_dataset.yaml` points at the demo split; classes are a starting point to be agreed with domain experts.
- The classical baseline parameters in `configs/baseline.yaml` are tuned for the synthetic data and **will need re-tuning (or replacing with the ML model) on real footage**.

---

## What ScaleAQ would need to provide to make this real

- **Representative footage** — images/video from real inspections across sites, seasons, depths, lighting and camera setups (ideally ScaleAQ camera / Vision system or ROV streams).
- **Real damage examples** — actual holes/tears/abnormal regions, plus plenty of *normal* net under varied conditions.
- **Metadata where available** — camera type, site, depth, date/time, lighting, environment.
- **Labelling guideline + domain feedback** — what counts as "damage", class definitions, edge cases.
- **An agreed error trade-off** — acceptable false-positive vs false-negative rates for the intended use.
- **The deployment target** — real-time on-ROV, offline review, operator decision support, alerting, or integration with existing software (e.g. the Vision platform via its open API).

See [`reports/SCALEAQ_PROTOTYPE_REPORT.md`](reports/SCALEAQ_PROTOTYPE_REPORT.md)
for the full write-up: assumptions, approach, results, failure modes and next steps.

---

## Tests

```powershell
pytest -q
```

## Data attribution

This prototype can download and process the **SOLAQUA** dataset:

> Ohrem, S. J., Haugaløkken, B. O. A., et al. *SOLAQUA: SINTEF Ocean Large
> Aquaculture Robotics Dataset.* SINTEF Ocean. Data:
> https://data.sintef.no/product/dp-7141fcd5-0fb8-4be3-b9ce-e5f7f5bb4a58 —
> arXiv:2504.01790. Licensed **CC BY-SA 4.0**.

SOLAQUA data is not redistributed in this repository; it is downloaded on demand
via `scripts/fetch_solaqua.py`. Any reuse must comply with CC BY-SA 4.0
(attribution + share-alike).

## License

MIT for this prototype's code. Downloaded SOLAQUA data remains under CC BY-SA 4.0.
