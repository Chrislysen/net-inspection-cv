# net-inspection-cv

[![CI](https://github.com/Chrislysen/net-inspection-cv/actions/workflows/ci.yml/badge.svg)](https://github.com/Chrislysen/net-inspection-cv/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%E2%80%933.14-blue)
![Tests](https://img.shields.io/badge/tests-189%20passing-brightgreen)
![Lint](https://img.shields.io/badge/lint-ruff-261230)
![License](https://img.shields.io/badge/license-MIT-green)

A research-grade computer-vision toolkit for **detecting possible damage (holes,
tears, abnormal regions) in aquaculture fish-farm net imagery**, built as an
honest *exploration framework*. It ingests images / video / ROS bags, preprocesses
underwater footage, and compares **five detection approaches** — a classical
OpenCV heuristic, a hand-crafted anomaly model, a **foundation-model anomaly
detector (PatchCore)**, a **supervised YOLOv8** detector/segmenter, and a
**det∧seg agreement ensemble** — with a rigorous, adversarial evaluation, temporal
video reasoning, an interactive web console, ONNX export, real-data (SINTEF
**SOLAQUA**) ingestion, and CI. Its defining feature is **intellectual honesty**:
it never claims real-world damage-detection performance it cannot prove, and it
reports the experiments that *failed* alongside the ones that worked.

<!-- Keywords (for search/research): aquaculture, fish farm, net pen, net damage,
hole/tear detection, underwater computer vision, ROV inspection, SOLAQUA, SINTEF,
YOLOv8, YOLOv8-seg, PatchCore, anomaly detection, OpenCV baseline, ROS bag,
rosbags, synthetic-to-real, domain gap, adversarial evaluation, FROC, temporal
tracking, ONNX, FastAPI, Streamlit, net pen inspection, escape prevention. -->

## At a glance

| | |
|---|---|
| **Problem** | Flag holes/tears/abnormal regions in underwater fish-farm net footage to support (not replace) human inspection. |
| **Methods compared** | Classical OpenCV heuristic · hand-crafted patch-Mahalanobis anomaly · **PatchCore** (pretrained-CNN anomaly, label-free; supervised-ResNet vs self-supervised-DINOv2 backbone ablation) · **YOLOv8** detection + **YOLOv8-seg** (supervised) · **det∧seg agreement ensemble**. |
| **Data** | Synthetic demo (pipeline test) · **SOLAQUA** real ROV footage of *undamaged* nets (SINTEF, CC BY-SA 4.0) · **synthetic damage composited onto real net frames** for trainable, labelled, comparable data. |
| **Key result (proxy)** | Damage-localisation F1 on the composite test set: anomaly **0.12** → classical **0.50** → PatchCore **0.78** (label-free) → YOLOv8 **0.97**. |
| **Honesty check** | On real *undamaged* net the detector fires on **0%** of two same-day clips, **31%** of a third, and **1%** on a different day. An earlier version of this table reported only the two clean clips and the different day — the 31% clip was missing from the evaluation set, so the headline understated false alarms. Corrected, and the corrected result is the more useful one: **between-clip spread on a single day (0→31%) is ~30× the day effect (1%)**, so the scene, not the day, is what these models are sensitive to. Looking at the frames shows what the scene effect *is*: the detector fires on the **thin bright mooring cords** rigged around calibration markers, not on the mesh. |
| **Robustness work** | Caught a seg-model out-of-distribution regression (31% → **18%** via multi-clip training; stronger augmentation **failed** at 22%). A 200-frame re-check — which **corrected my own 1%→11% sampling artifact** — shows an honest **precision/recall trade-off** on the held-out day: a bigger **YOLOv8s-seg (A100)** = best recall **0.98** at 11% false alarms; the **det∧seg ensemble** = **0%** false alarms but ~0.57 recall. Plus an **OOD gate** (defers 100% of different-day frames to human review) and a *failed* test-time normalisation. Full ledger: [report §5.7–5.8](reports/PROTOTYPE_REPORT.md). |
| **Temporal** | Persistence tracking removes **~70%** of transient false alarms on real undamaged video. |
| **The hard limit** | All numbers are on **synthetic damage** (one generator). **No validated real-world damage-detection performance is claimed** — that needs real *labelled* net-damage footage. The repo is the drop-in slot for it. |
| **Beyond vision** | **ROV telemetry** from all 5 SOLAQUA sensor bags (net standoff, DVL, depth, temperature, thrust) joined to frames on the bag clock · a per-pass **inspection-validity report** · **site planning** from the Fiskeridirektoratet register × MET Norway ocean forecast · a **DuckDB reporting layer** over every artifact. |
| **Grounded assistant** | Tool-calling Q&A over the real artifacts, guarded by a machine-readable **evidence ledger** + a deterministic post-check. Backend is swappable (Claude API or local Ollama), so the guardrail is **measured**, not asserted: boundary disclosure **100%** on two local 14B models, tool grounding 50–75%. |
| **Engineering** | Unified inference facade · FastAPI service + **interactive web console** · Streamlit viewer · batch/video/bag runner · COCO→YOLO adapter · ONNX export + benchmark · **189 passing tests** · GitHub Actions CI · committed models. |
| **Stack** | Python 3.11–3.14 · OpenCV · PyTorch/torchvision · Ultralytics YOLOv8 · scikit-learn · rosbags · FastAPI · NumPy/Pandas. |

**Where to look:** code in [`src/netinspect/`](src/netinspect/), CLIs in
[`scripts/`](scripts/), the full write-up in
[`reports/PROTOTYPE_REPORT.md`](reports/PROTOTYPE_REPORT.md),
result tables/figures in [`reports/results/`](reports/results/), models in
[`models/`](models/), and the web UI in [`web/`](web/). For the reasoning trail:
the [research decision log](reports/RESEARCH_SYNTHESIS.md) (what external research
was adopted vs rejected and why) and a code-grounded
[technical defense Q&A](reports/INTERVIEW_DEFENSE.md). For deployment:
[model card](reports/MODEL_CARD.md), [deployment runbook + SLOs](reports/DEPLOYMENT.md),
[data-collection protocol](reports/DATA_COLLECTION.md), and the honest
[production-readiness scorecard](reports/PRODUCTION_READINESS.md).

> **Status: prototype.** The repository ships with a small *synthetic* dataset
> so the full pipeline runs out of the box, **and** an integration with the real
> **SOLAQUA** ROV dataset (SINTEF). **Synthetic results only verify that the code
> path works.** SOLAQUA is real footage but of **undamaged** nets with **no
> damage labels**, so it is used for real-image preprocessing, false-positive /
> robustness analysis, and anomaly detection — **not** for measuring damage
> detection accuracy. Validated damage-detection numbers still require real,
> *labelled* footage (see
> [What a farm operator would need to provide](#what-a-farm-operator-would-need-to-provide-to-make-this-real)).

---

## The headline result, in three pictures

### 1. What the detector actually fires on

![what the detector fires on](docs/images/what_the_detector_fires_on.jpg)

Two clips, **same day, same site, same camera, near-identical standoff** — and
**both show undamaged net**, so every box is a false positive. The top clip
produces a 31% false-alarm rate; the bottom produces zero.

The boxes do not land on the calibration markers. They land on the **thin bright
mooring cords** rigged around them — elongated high-contrast structures that
resemble the synthetic tears the detector was trained on. The bottom clip also
carries hardware (floats, smaller markers) further away and fires nothing, so the
trigger is not "equipment in frame" but a particular shape at a particular scale.

Reproduce: `python scripts/make_failure_figure.py`

### 2. The measurement behind it

![what drives false alarms](docs/images/what_drives_false_alarms.png)

638 real frames, 4 clips, 2 days. A **pre-registered hypothesis** — that the
long-running "different-day" gap was an operating-envelope violation driven by
standoff distance — was tested and **rejected**: the clip spanning the widest
standoff range (0.19–1.31 m) produces zero false alarms, and standoff shows no
within-clip association on the held-out day (all *p* > 0.67).

| | |
|---|---|
| Between-clip spread, **one day**, identical standoff | **0% → 31%** |
| Different-day effect | 1% |
| Strongest per-frame correlate | capture sharpness, **r = +0.60** (*p* ≈ 1e-62) |
| Intra-cluster correlation (frames within a clip) | **ICC ≈ 0.31** |
| Effective sample size | **≈ 13**, not 638 |
| Naive per-frame CI vs clip-clustered CI | [0.08, 0.12] vs **[0.00, 0.25]** |

The last three rows matter: frames within a clip are strongly correlated, so a
per-frame confidence interval overstates how well any of this generalises to a
*new net*. Reproduce: `python scripts/analyze_operating_envelope.py`

### 3. Deciding when to fly at all

![site planning](docs/images/site_planning.png)

Detection models answer "is there damage in this frame". An operator also needs
"is it worth flying today, and was the footage captured under conditions where
that answer means anything". This joins the **Fiskeridirektoratet** cod-locality
register to the **MET Norway** ocean forecast — both open, neither needing a key
— to rate inspection windows per site, and shows the size-dependent cod thermal
optimum against today's sea temperature.

Reproduce: `python scripts/site_planner.py --operator ODE --forecast`

---

## Interactive console

A localhost **inspection console** (FastAPI + a custom web UI) ties it all
together and is built to be **self-explanatory**: each frame source carries a
one-line "what am I looking at", each detector a plain-language explainer, and an
**OOD-gate toggle** shows a live *in-distribution / review-needed* badge per frame
(the deploy-time safety net). Browse real SOLAQUA frames, switch detector
(classical / anomaly / PatchCore / YOLO / **ensemble**), adjust confidence live,
and compare methods — with a detection table and latency. `python scripts/serve.py`
then open `http://127.0.0.1:8000`.

![web console](docs/images/webapp_console.png)

*(Shown: YOLO flagging 4 damage regions on composited-on-real net. On real
**undamaged** frames it correctly shows zero — consistent with the adversarial eval.)*

## Grounded assistant — and measuring whether the guardrail holds

A tool-calling assistant answers operational questions over the repo's real
artifacts (inspection results, ROV telemetry, the evidence ledger) and cites the
file behind every number. Its defining behaviour is what it *won't* do: asked
"how accurate is this on real damage?", it states that recall on real damage has
never been measured rather than reaching for the synthetic-proxy F1.

That is enforced in two independent layers — a machine-readable **evidence
ledger** compiled into the system prompt, and a **deterministic post-check** that
flags an answer touching unvalidated capability without the caveat. The second
layer is plain string matching precisely so it holds when the model doesn't, and
is testable with no API key.

```bash
python scripts/ask.py --backend ollama "How accurate is this on real damage?"
python scripts/eval_assistant.py --backend ollama --model qwen3:14b
```

**The backend is swappable** (Anthropic API or a local Ollama model) with
identical tool schemas and an identical guardrail, so the same 12-case
adversarial suite turns a design claim into a measurement:

| backend / model | boundary disclosure | tool grounding | overall |
|---|---|---|---|
| ollama / qwen2.5:14b-instruct (local) | **5/5 (100%)** | 9/12 (75%) | 75% |
| ollama / qwen3:14b (local) | **5/5 (100%)** | 6/12 (50%) | 50% |

The two properties **diverge**, which is the interesting part. The *safety*
property — refusing to answer past the evidence — is carried by the system
prompt and held at **100% on both local models**. The *provenance* property —
verifying by tool call rather than answering from the prompt — did not: every
single remaining failure is the same mode, a correct answer produced without
checking. Right answer, wrong process, and worth knowing before trusting it in
an operational loop.

Two of those "failures" were originally **mine, not the models'**: a
`must_mention` check on the bare substring `"not"` scored a correct
"the hypothesis has been rejected" as a miss, and the missing-data marker list
lacked `"not known"`. Both were corrected and the saved answers re-scored with
`--rescore`, which calls no model — fixing a brittle check and quietly re-running
the models would have reported different numbers for the wrong reason.

## Example outputs

| Synthetic demo (prediction vs. ground truth) | Synthetic failure case (merged adjacent defects) |
|---|---|
| ![synthetic overlay](docs/images/synthetic_overlay.jpg) | ![synthetic failure](docs/images/synthetic_failure_match.jpg) |
| Orange = prediction, green = ground truth. | Orange = TP, **blue = missed** GT (a hole + adjacent tear merged into one box). |

| Classical baseline on **real** SOLAQUA net (all detections are false alarms) | Anomaly model heatmap on the same frame |
|---|---|
| ![real false positives](docs/images/real_false_positives.jpg) | ![real anomaly heatmap](docs/images/real_anomaly_heatmap.jpg) |
| The nets are **undamaged**; the heuristic still fires on oblique/dark cells. | Anomaly score (left: boxes, right: heatmap) concentrates on biofouling & lighting, not damage. |

**YOLOv8 detecting damage on real net (live `scripts/infer.py` output, confidences shown):**

![yolo detection](docs/images/yolo_detection_example.jpg)

**Live stream (`scripts/live_inspect.py`) — real-time overlay with temporal confirmation + OOD gate:**

![live inspect](docs/images/live_inspect_example.jpg)

*Status bar shows method · confirmed-defect count · OOD verdict. Here the ensemble
boxes a damage tear (0.96) and correctly ignores the instrument housing, while the
**OOD gate flags this different-day frame for review** — the deploy-time safety net.*

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

| Frame set | det v1 | seg v3 | seg-gpu |
|---|---|---|---|
| bag1 — the backgrounds it trained on (38 frames) | **0%** | 0% | 0% |
| bag2 — same site, other clip (120) | **0%** | 2.5% | 0% |
| **bag3 — same DAY, third clip (199)** | **31%** | 12% | 2.5% |
| different day (200) | **1%** | 18% | 11% |

**Read the third row before the fourth.** An earlier version of this table omitted
bag3 entirely — the evaluation set happened to contain the two clean same-day clips
and not the dirty one, so "0% on its own training backgrounds" was true of the
subset and false of the day. That is a sampling artifact of my own making, the same
class as the 1%→11% one recorded in §5.7, and it is the third such correction in
this project. All four undamaged clips on disk are now evaluated
(`scripts/adversarial_eval.py`).

The corrected numbers say something more interesting than the original claim did.
bag1, bag2 and bag3 are the **same day, same site, same camera**, flown at
near-identical commanded standoff (0.65 / 0.60 / 0.60 m — verified against the ROV
telemetry in `scripts/extract_telemetry.py`). Two of them produce zero false alarms
and one produces 31%. **The between-clip spread on one day is roughly thirty times
the different-day effect.** Whatever these models are sensitive to, it is a property
of the scene, not of the calendar — which is why the pre-registered hypothesis that
standoff distance explained the different-day gap was tested and **rejected**
(`scripts/analyze_operating_envelope.py`).

It also means the per-frame confidence intervals in this repo are too narrow.
Frames within a clip are strongly correlated (ICC ≈ 0.31 for the detector), so 638
frames carry roughly **13 frames' worth of independent information**. The honest
interval on the detector's overall false-alarm rate is the clip-clustered
**[0.00, 0.25]**, not the naive per-frame [0.08, 0.12].

**Temporal confirmation** removes a further **70%** of transient false alarms on
real undamaged video.

**Closing the different-day gap — a precision/recall trade-off (measured on 200
frames).** The nano segmenter fires on 18% of *different-day* undamaged frames (vs
the detector's 1%). Two honest options, opposite ends of the operating curve:

| Held-out different day (200 frames) | det v1 | seg v3 | **ensemble (det∧seg)** | **seg-gpu (A100)** |
|---|---|---|---|---|
| Undamaged false-positive rate | 1% | 18% | **0%** | 11% |
| Damage recall (F1) | 0.56 | 0.93 | 0.57 | **0.98** |

The **ensemble** (detector proposes, segmenter confirms) drives false alarms to
**0%** and adds masks, but inherits the detector's caution (~0.57 recall — it only
confirms damage the detector already found). The bigger **`seg-gpu`** catches
**0.98** of the damage at an 11% false-alarm cost. Pick by which error is more
expensive — a *missed defect* or a *false alarm*. (Recall here is on synthetic
damage and a small split, so it's a noisy proxy.)

![ensemble false-positive suppression](docs/images/ensemble_fp_suppression.jpg)

*Real different-day undamaged net. **Left:** `seg v3` false-alarms on an instrument
housing. **Right:** the det∧seg ensemble stays clean — same frame, no retraining.*

> **Read these honestly.** Synthetic metrics only verify the pipeline. SOLAQUA
> frames are real but **undamaged/unlabelled** (false-alarm & anomaly behaviour
> only). The YOLO numbers are on **synthetic damage composited on real
> backgrounds** — the *damage appearance* comes from one generator (same in
> train and test) and the cross-clip set is the same site/camera, so this shows
> "learns this damage model and transfers across clips/backgrounds", **not**
> "detects real damage." Real labelled damage is still required to claim
> real-world performance. Full detail in the [report](reports/PROTOTYPE_REPORT.md).

## Results (figures)

Generated from the committed result JSONs by `python scripts/make_plots.py`
(figures in [`reports/results/plots/`](reports/results/plots/)).

| Label-free → supervised ladder | Detection metrics by method |
|---|---|
| ![f1 ladder](reports/results/plots/method_f1_ladder.png) | ![metrics](reports/results/plots/method_metrics.png) |

| Adversarial: FP on REAL undamaged net (det v1 vs seg v2/v3/v4) | FROC operating curve (det v1 vs seg v2/v3/v4) |
|---|---|
| ![fp undamaged](reports/results/plots/fp_on_undamaged.png) | ![froc](reports/results/plots/froc.png) |

| Ensemble recovers det-v1 robustness + keeps masks | OOD gate defers the unfamiliar frames to a human |
|---|---|
| ![ensemble](reports/results/plots/ensemble_comparison.png) | ![ood gate](reports/results/plots/ood_gate.png) |

| Temporal confirmation removes ~70% of transient false alarms |
|---|
| ![temporal](reports/results/plots/temporal_reduction.png) |

The robustness story, end to end: the first segmentation model (`seg v2`) regressed
badly out-of-distribution (31% different-day false alarms) — a failure the
evaluation **caught**. Multi-clip training (`seg v3`) cut that to **18%**; a
**stronger-augmentation** follow-up (`seg v4`) **failed** to improve it (22% — kept
as an honest negative); and a bigger **YOLOv8s-seg on a GPU** (`seg-gpu`) reached
the **best recall (0.98)** at an 11% false-alarm cost. The honest conclusion is a
**precision/recall trade-off**, not a single winner: the **det v1 / ensemble** end
(0–1% false alarms, lower recall) vs the **seg-gpu** end (0.98 recall, 11% false
alarms) — chosen by which error costs more. (One flattering 1% number along the way
turned out to be a small-sample artifact, re-checked on 200 frames and corrected.)
Separately, a label-free **backbone ablation** for the anomaly detector
(`reports/results/ssl_dino/`): off-the-shelf **DINOv2** features are competitive and
cleaner than ImageNet-ResNet18 but not a free OOD win; and **domain SSL** — a
**SimCLR ResNet18 I pretrained from scratch on the unlabelled SOLAQUA frames**
(`ssl_pretrain.py`) — *underperforms* both (AUROC 0.80/0.61 vs 0.98/0.82 and
1.00/0.93), an honest reminder that self-supervised learning needs scale (508 frames
isn't it; the GPU path to the full video is built in).

## Usage cases

**1. Screen inspection video for suspicious regions (offline review).**
```powershell
python scripts/infer.py --method yolo --yolo-weights models/yolo_damage_v1.pt --source clip.mp4 --out outputs/review --every-n 5
# fewer false alarms via temporal confirmation on a contiguous sequence:
python scripts/run_temporal.py --method yolo --yolo-weights models/yolo_damage_v1.pt --source data/processed/frames --out outputs/review_temporal
```

**1b. Live camera / ROV feed (real-time).** `live_inspect.py` runs any method on an
**RTSP/HTTP IP camera, a USB webcam, or a video file**, confirms defects over time
(one alert per *persisting* defect, not per frame), and flags out-of-distribution
frames for human review:
```powershell
# RTSP/ROV stream, ensemble detector, live window + event log:
python scripts/live_inspect.py --source rtsp://CAMERA_IP/stream --method ensemble \
    --yolo-weights models/yolo_damage_v1.pt --seg-weights models/yolo_damage_seg_v3.pt \
    --patchcore-model models/patchcore_normal_net --display --out outputs/live
# USB webcam 0:  --source 0     # any video file:  --source clip.mp4
```
Emits `damage_confirmed` events (`events.jsonl`) with bbox, score and an `ood_review`
flag. For a ROS-publishing ROV, swap the source for an `rclpy`/`rospy` image
subscriber and call the same facade per message (see `solaqua.py` for the pattern).

**2. Only have *normal*-net footage, want to flag anything unusual (label-free).**
```powershell
python scripts/train_patchcore.py --images data/processed/normal_frames --out models/patchcore_mynet
python scripts/run_anomaly.py --images data/processed/new_frames --model models/patchcore_mynet --out outputs/anomaly   # heatmaps + boxes
```

**3. Have labelled damage → train and rank approaches.**
```powershell
python scripts/prepare_data.py --images data/raw/images --labels data/raw/labels --out data/processed --yolo-split
python scripts/train_yolo.py --data data/processed/yolo/dataset.yaml --epochs 60
python scripts/compare_methods.py --images <test_imgs> --labels <test_lbls> --yolo-weights runs/detect/train/weights/best.pt --out outputs/comparison
```

**4. Decide whether to trust a model (before deploying).**
```powershell
python scripts/adversarial_eval.py --yolo-weights models/yolo_damage_v1.pt --out reports/results/adversarial   # FP on undamaged + FROC
# Combine the robust detector with the segmenter (det proposes, seg confirms):
python scripts/eval_ensemble.py --det models/yolo_damage_v1.pt --seg models/yolo_damage_seg_v3.pt --out reports/results/ensemble
# Out-of-distribution gate — flag unfamiliar frames for human review:
python scripts/eval_ood_gate.py --patchcore-model models/patchcore_normal_net --out reports/results/ood_gate

# Subtle-damage stress test (--hard = small, low-contrast damage) + sensitivity dial:
python scripts/make_real_dataset.py --frames data/processed/solaqua_diffday --out data/processed/hard_composite --hard
python scripts/eval_sensitivity.py --yolo-weights models/yolo_damage_seg_gpu.pt \
    --composited data/processed/hard_composite/images/test data/processed/hard_composite/labels/test \
    --undamaged data/processed/solaqua_diffday --out reports/results/sensitivity_hard   # recall vs false-alarms by conf
python scripts/make_plots.py        # turn results into figures
```

**5. Serve / explore.**
```powershell
python scripts/serve.py --yolo-weights models/yolo_damage_v1.pt --patchcore-model models/patchcore_normal_net   # HTTP API
streamlit run streamlit_app.py      # interactive viewer
```

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
    dino_backbone.py             self-supervised DINOv2 backbone for PatchCore (SSL-vs-supervised ablation)
    ssl_pretrain.py              from-scratch SimCLR — domain-pretrain a ResNet18 on unlabelled SOLAQUA
    ensemble.py                  det-proposes / seg-confirms agreement ensemble (det-v1 robustness + masks)
    ood_gate.py                  out-of-distribution gate — defer unfamiliar frames to human review
    temporal.py                  IoU tracker — confirm detections that persist
    compose.py                   composite photorealistic damage + hard negatives onto REAL frames
    inference.py                 unified facade over all methods (used everywhere)
    evaluate.py                  detection / segmentation / image-level metrics
    visualize.py                 overlays, comparisons, galleries
    video.py                     video frame extraction
    solaqua.py                   SOLAQUA client + ROS-bag camera & sonar extraction
    coco.py                      COCO -> YOLO adapter (real labelled data drop-in)
    synthetic.py                 placeholder data generator (testing only)
    utils.py                     IO, geometry, optional-dependency handling
  scripts/                       CLI entry points (see below)
  web/                           interactive console (index.html / style.css / app.js)
  streamlit_app.py               alternative interactive viewer
  models/                        committed prototype models (.pt/.npz/.onnx) + NOTICE
  .github/workflows/ci.yml       CI: run tests on push/PR
  tests/                         pytest: data loading + metrics
  reports/PROTOTYPE_REPORT.md
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

**GPU training (Colab).** Heavy runs (multi-clip, bigger `yolov8s/m-seg`, longer
schedules) are slow on CPU. [`notebooks/colab_train_gpu.ipynb`](notebooks/colab_train_gpu.ipynb)
runs the identical pipeline on a Colab GPU — clone → pull SOLAQUA → rebuild the
composited dataset → train → held-out adversarial eval → download weights — in
minutes, and sketches the GPU-only **SSL-pretraining-on-SOLAQUA** next step.

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

# Self-supervised vs supervised features (same detector, only the backbone differs):
python scripts/train_patchcore.py --images data/processed/solaqua_frames_dense --backbone dinov2_vits14 --out models/patchcore_dino_vits14
python scripts/compare_anomaly_backbones.py --models resnet18=models/patchcore_resnet18 dinov2=models/patchcore_dino_vits14 --out reports/results/ssl_dino

# Domain SSL: SimCLR-pretrain a ResNet18 on the UNLABELLED SOLAQUA frames (GPU=minutes),
# then use that backbone in PatchCore vs ImageNet-supervised vs off-the-shelf DINOv2:
python scripts/pretrain_ssl.py --frames data/processed/solaqua_frames_dense data/processed/solaqua_bag2 data/processed/solaqua_bag3 --epochs 400 --batch 256 --device cuda --out models/ssl_resnet18_solaqua.pt
python scripts/train_patchcore.py --images data/processed/solaqua_frames_dense --backbone-weights models/ssl_resnet18_solaqua.pt --out models/patchcore_ssl_solaqua

# Adversarial "is it cheating?" suite: FP on REAL undamaged net + different-day + FROC
python scripts/adversarial_eval.py --yolo-weights models/yolo_damage_v1.pt --out reports/results/adversarial_yolo

# Temporal confirmation on a CONTIGUOUS sequence: removes ~70% of transient false alarms
python scripts/fetch_solaqua.py --bag <clip>.bag --frames-out data/processed/solaqua_seq --every-n 1 --max-frames 120
python scripts/run_temporal.py --method classical --source data/processed/solaqua_seq --out outputs/temporal --config configs/baseline.yaml
```

Committed result tables: [`reports/results/`](reports/results/) — 4-method
comparison, adversarial eval, temporal.

### Ingesting real labelled data (COCO) + deployment export

```powershell
# Real labels usually arrive as COCO -> convert to YOLO (the drop-in slot for
# an operator's eventual labelled damage). Also handles SeaClear/TrashCan etc.
python scripts/convert_coco.py --coco anns.json --images imgs --out data/processed/real --single-class
python scripts/train_yolo.py --data data/processed/real/dataset.yaml --epochs 60

# Export to ONNX (portable; TensorRT/INT8 on-device) + latency benchmark
python scripts/export_onnx.py --weights models/yolo_damage_v1.pt --imgsz 480
```

> **Honest note on public datasets.** SeaClear / TrashCan / Trash-ICRA19 are
> marine *debris* (plastic, animals, plants) — useful as a real-image smoke test
> of the COCO adapter and for transfer/pretraining, **not** a net-damage proxy.
> Verify each dataset's licence before downloading. The COCO adapter exists so
> that *real labelled net-damage data* drops straight in when it's available.
>
> **Deployment.** ONNX export works and is benchmarked
> ([`reports/results/onnx_benchmark.json`](reports/results/onnx_benchmark.json));
> on this CPU dev box ONNX Runtime wasn't faster than PyTorch — the real speedup
> is **TensorRT (FP16/INT8) on the target device** (e.g. Jetson), which is
> documented rather than run here (needs the device + calibration data).

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
- **Segmentation** (YOLOv8-seg) for masks; **COCO ingestion adapter** (real labelled data drops in); **ONNX export + latency benchmark**.
- **Interactive web console** (FastAPI + custom UI) with self-explanatory source/method descriptions and a live **OOD-gate badge**; Streamlit viewer; batch/video/bag runner; **`live_inspect.py`** real-time RTSP/USB/ROV streaming with temporal confirmation + per-defect alerts; per-class evaluation; CI.
- **Production-shaped serving:** path-traversal-safe, upload validation, structured logging + request IDs, `/health` `/ready` `/metrics`, no-leak error handling.
- **Torch-free ONNX inference** (`onnx_infer.py`, parity-verified) + a **streaming pipeline** (`stream_inspect.py`) that emits one confirmed-damage alert per new track.
- **Ops artifacts:** model card, deployment/SLO runbook, data-collection protocol, and a self-critical [production-readiness scorecard](reports/PRODUCTION_READINESS.md).
- Tests (61) for data, metrics, anomaly, compositing, inference, temporal, PatchCore, COCO, per-class, **service security/integration**, **ONNX**, the **DINOv2 backbone**, and **SimCLR pretraining**.

**Placeholder / synthetic (clearly marked):**

- `data/sample/` and the `synthetic.py` generator are **procedural placeholders**. Any metric computed on them measures the pipeline, not real-world skill.
- `configs/yolo_dataset.yaml` points at the demo split; classes are a starting point to be agreed with domain experts.
- The classical baseline parameters in `configs/baseline.yaml` are tuned for the synthetic data and **will need re-tuning (or replacing with the ML model) on real footage**.

---

## What a farm operator would need to provide to make this real

- **Representative footage** — images/video from real inspections across sites, seasons, depths, lighting and camera setups (ideally the operator's own inspection-camera or ROV streams).
- **Real damage examples** — actual holes/tears/abnormal regions, plus plenty of *normal* net under varied conditions.
- **Metadata where available** — camera type, site, depth, date/time, lighting, environment.
- **Labelling guideline + domain feedback** — what counts as "damage", class definitions, edge cases.
- **An agreed error trade-off** — acceptable false-positive vs false-negative rates for the intended use.
- **The deployment target** — real-time on-ROV, offline review, operator decision support, alerting, or integration with existing inspection software via its API.

See [`reports/PROTOTYPE_REPORT.md`](reports/PROTOTYPE_REPORT.md)
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
