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
| The nets are **undamaged**; the heuristic still fires on oblique/dark cells — ~76% false-alarm frame rate. | Anomaly score (left: boxes, right: heatmap) concentrates on biofouling & lighting, not damage. |

> These images illustrate behaviour, not validated performance. Synthetic
> metrics only verify the pipeline; SOLAQUA frames are real but **undamaged and
> unlabelled**, so they show false-alarm/anomaly behaviour, not damage-detection
> accuracy. See the [report](reports/SCALEAQ_PROTOTYPE_REPORT.md).

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

## Two compared approaches

| | Classical baseline | ML baseline |
|---|---|---|
| **Module** | `classical_baseline.py` | `model_baseline.py` (Ultralytics YOLOv8) |
| **Idea** | Net = regular mesh texture; damage breaks it. Combine an **absolute-darkness** cue (see-through holes) with a **low-edge-density** cue (mesh locally missing). | Learn "damage" appearance from labelled examples. |
| **Needs labels?** | No (unsupervised heuristic) | Yes (YOLO det/seg format) |
| **Pros** | Fast, transparent, no training data. Good for sanity-checking and data triage. | Learns real appearance; far better ceiling on messy data. |
| **Cons** | Brittle; fooled by shadows/biofouling; many false positives on real footage. | Needs representative labelled data; less interpretable. |

The classical baseline is a **sanity-check and difficulty probe**, not a final
detector. The ML path is the route to real performance once data is available.

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
    evaluate.py                  detection / segmentation / image-level metrics
    visualize.py                 overlays, comparisons, galleries
    video.py                     video frame extraction
    solaqua.py                   SOLAQUA client + ROS-bag frame extraction
    synthetic.py                 placeholder data generator (testing only)
    utils.py                     IO, geometry, optional-dependency handling
  scripts/                       CLI entry points (see below)
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
pip install -e ".[cv,ml,data,dev]"      # cv = OpenCV/matplotlib/skimage; ml = ultralytics/torch; data = rosbags (SOLAQUA)
```

If you only want the classical baseline + evaluation: `pip install -e ".[cv,dev]"`.
The `data` extra (`rosbags`) is only needed to read SOLAQUA `.bag` files.
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
- **Real-data ingestion (SOLAQUA)**: public-API client, resumable download, ROS `.bag` frame extraction.
- **Anomaly-detection baseline**: patch-feature Mahalanobis model with anomaly heatmaps; trains on normal net, flags deviations.
- **Real-frame false-positive analysis** on SOLAQUA.
- Tests for data loading, metrics, and the anomaly model; one-command report-asset generation.

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
