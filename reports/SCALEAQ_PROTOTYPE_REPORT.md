# Aquaculture Net Inspection — Computer Vision Prototype

**A technical write-up of `net-inspection-cv`.**

This report describes how I would approach the image-analysis task for net
inspection: understanding the data, building baselines, evaluating them
honestly, analysing failures, and recommending next steps. It is deliberately
grounded — it states what is implemented, what is assumed, and what would be
needed to turn the prototype into something useful on real footage.

> **Headline caveat.** This report contains two kinds of results, both clearly
> labelled and **neither** a validated damage-detection score:
>
> 1. **Synthetic data** (`src/netinspect/synthetic.py`) — verifies the pipeline
>    runs end to end. Says nothing about real-world skill.
> 2. **Real data — SOLAQUA** (SINTEF ROV footage, CC BY-SA 4.0) — real
>    underwater net imagery, but of **undamaged** nets with **no damage labels**.
>    Used for real-image preprocessing, **false-positive / robustness analysis**,
>    and **anomaly detection**. Because there is no damage and no ground truth,
>    it **cannot** measure damage-detection precision/recall.
>
> Validated damage-detection numbers require real, *labelled* footage containing
> actual damage. That is the main thing still needed from ScaleAQ.

---

## 1. Problem understanding

Fish-farm nets develop **holes, tears, deformations and abnormal regions**.
Missing them has real consequences: fish escape (financial, regulatory and
environmental cost), so inspection is a routine, safety-relevant operation.
Today, inspection largely means a human watching underwater camera or ROV
footage and deciding whether something looks wrong.

The opportunity is **decision support**: a vision system that flags suspicious
regions for a human to confirm. The realistic near-term goal is **not** a fully
autonomous "the net is broken" classifier, but a tool that reduces how much
footage a person must watch and draws attention to likely problems — while
keeping a human in the loop.

This is consistent with the published research direction: ROV-mounted cameras
feeding deep-learning detectors/segmenters of net defects (holes, biofouling,
vegetation, debris) under difficult underwater conditions.¹⁻⁴ Reported accuracies
above 90% exist, but on *custom datasets under controlled-ish conditions* — which
underlines that **the data, not the model, is the hard part**.

### 1.1 Operational context & stakes (why honesty is the design driver)

This is a **safety-relevant** application, and that shapes every design choice:

- **The cost of errors is asymmetric.** A false negative (missed damage) can lead
  to a **fish escape** — with financial loss, environmental impact, and regulatory
  reporting obligations. A false positive wastes operator/ROV time. The operating
  point must therefore be chosen *with stakeholders* from the FROC/PR curve, not
  defaulted — almost certainly biased toward recall with **human-in-the-loop**
  confirmation. (Norwegian marine fish-farm installations are governed by the
  **NS 9415** standard; this prototype is decision *support* for inspection, not an
  autonomous authority on net integrity.)
- **Evaluation must be by site/condition, not random split.** Underwater
  appearance varies with turbidity, light, depth, biofouling and camera. A random
  split leaks near-duplicate frames and flatters the model; held-out *sites/days*
  are the honest test. This repo already evaluates cross-clip and **different-day**
  for exactly this reason (§5.5–5.7).
- **Integration target matters.** ScaleAQ already runs camera systems and the
  *Vision* platform with open APIs; a real capability would plug in there (offline
  review first, then alerting), not as a standalone tool. The serving layer here
  (`serve.py`, unified `inference.py`) is shaped for that hand-off, not as a product.
- **Why this drives the whole project:** because a confidently-wrong "the net is
  fine" is the most expensive failure mode, the work prioritises *not overclaiming*
  and *adversarially testing its own outputs* over headline metrics.

---

## 2. Data assumptions

Because real data was not available for this prototype, I made explicit,
conservative assumptions and built the pipeline so real data drops in cleanly:

- **Input** is images and/or video frames from underwater cameras or ROVs.
- **Labels**, when they exist, are in standard **YOLO format** — detection
  (`class xc yc w h`) or segmentation (`class x1 y1 … xn yn`), both supported.
- **Classes** start as `damage, hole, tear, abnormal_region` (placeholder — to
  be agreed with domain experts and a labelling guideline).
- **Damage is rare**: most frames are intact net, so class imbalance and the
  cost asymmetry between false positives and false negatives are central, not
  incidental.
- Images may be **unlabelled**; the pipeline must still ingest, preprocess,
  predict and produce qualitative output, and must say clearly when no
  quantitative evaluation is possible.

---

## 3. Proposed approaches

I deliberately compare a **classical** and an **ML** approach, because on a new
problem with little data the cheap explainable method tells you how hard the
problem is and whether the ML route is worth it.

### 3.1 Classical / OpenCV baseline (implemented)

**Insight:** a net is a *regular mesh texture*. Damage breaks that regularity:

- **Holes** appear as locally **dark, low-texture** blobs (you see through to
  open water), so the mesh edges vanish.
- **Tears** are dark, elongated disruptions of the mesh.

The baseline combines two transparent cues and is fully explainable:

1. **Absolute-darkness cue** — luminance below `mean − k·σ` on the
   white-balanced (pre-CLAHE) image. CLAHE is deliberately *not* used for this
   cue because it equalises local contrast and destroys the absolute darkness
   that distinguishes a see-through hole from mid-toned mesh.
2. **Low-edge-density cue** — regions where local Canny edge density falls below
   a fraction of the image-wide median, i.e. the mesh pattern is locally absent.

The cues are combined with **OR** (they catch different damage types: thin tears
via darkness, compact holes via low density), cleaned with morphology, extracted
as contours, and filtered by area, solidity and per-contour texture. A
transparent heuristic score (darkness + size + emptiness) ranks candidates.

This is a **sanity-check and difficulty probe**, not a production detector.

### 3.2 ML baseline — YOLOv8 detection / segmentation (implemented, runnable)

A thin wrapper around **Ultralytics YOLOv8** (`detect` or `segment`). It accepts
the standard YOLO dataset format, trains, runs inference, and emits the **same
prediction JSON schema** as the classical path so both are evaluated
identically. It degrades gracefully: if `ultralytics` is not installed, the
scripts print actionable guidance instead of crashing.

YOLOv8-**seg** is the more promising long-term option because damage is
irregular and pixel-level masks capture extent better than boxes — but it needs
mask labels.

### 3.3 Anomaly detection (implemented)

If only *normal*-net images are available (the likely early situation — and
exactly what SOLAQUA provides), a **one-class / anomaly detector** is the natural
third approach. Implemented here as a deliberately simple, explainable
PaDiM-style model (`src/netinspect/anomaly.py`):

1. split each frame into a patch grid;
2. describe each patch with Lab colour stats, local contrast and edge/texture
   density (intact mesh is regular and textured);
3. fit a single multivariate Gaussian to patch features from normal frames
   (standardised, covariance-shrunk for a stable inverse);
4. score new patches by Mahalanobis distance → an anomaly heatmap + candidate
   regions, with the threshold calibrated from the training-distance percentile.

This flags *deviation from normal net*, not validated damage — on real footage it
also flags fish, biofouling and lighting. A learned deep backbone or autoencoder
is the next step; this is the honest baseline to beat.

---

## 4. Implemented prototype

A working, modular, CLI-driven repository (see the README for the layout and
commands). End to end:

```
ingest (synthetic | images | video frames | SOLAQUA ROS bags)
   → preprocess (white balance, CLAHE, denoise, resize)
   → classical baseline  ─┐
   → YOLOv8 baseline      ─┤→ unified predictions JSON
   → anomaly model        ─┘   (normal-net deviation, heatmaps)
   → evaluate (det / seg / image-level, FP & FN, sweeps)
   → real-frame false-positive analysis (SOLAQUA)
   → visualise (overlays, match overlays, failure cases, heatmaps, galleries)
   → report assets
```

An **industrial-shaped serving layer** wraps these: a unified inference facade
(`inference.py`), a **FastAPI** service with an **interactive ROV web console**
(`serve.py` + `web/`), a **Streamlit** viewer, a unified **batch/video/bag**
runner with run manifests (`infer.py`), and **CI**. A **COCO→YOLO adapter**
(`coco.py`) is the drop-in slot for real labelled data (which usually arrives as
COCO); public debris datasets (SeaClear etc.) are usable through it as a
real-image smoke test and for transfer, but **not** as a net-damage proxy.
**ONNX export** (`export_onnx.py`) plus a latency benchmark provide the
deployment hand-off; on this CPU box ONNX Runtime was not faster than PyTorch, so
the real speed-up (**TensorRT FP16/INT8 on the ROV's device**) is documented
rather than run (it needs the target hardware and calibration frames).

Verified on this machine (Python 3.14, OpenCV 4.x, Ultralytics 8.x, torch 2.x
CPU, rosbags): synthetic demo runs end to end; **26/26 unit tests pass**; the
**real SOLAQUA pipeline runs** (downloaded two bags totalling ~2.1 GB, extracted
camera + multibeam-sonar frames); a **YOLOv8 trained for 30+ epochs** on
synthetic-damage-on-real-frames reaches **F1 ≈ 0.97** in-clip and **≈ 0.98**
cross-clip; the FastAPI service and Streamlit viewer both boot and serve
predictions.

---

## 5. Evaluation method

Evaluation adapts to the labels available:

- **Detection** — greedy IoU matching (default IoU 0.30, class-agnostic for a
  localisation-focused prototype): precision, recall, F1, VOC-style **AP**
  (101-point), a **confidence sweep**, and explicit **false-positive /
  false-negative lists** with per-image match overlays.
- **Segmentation** — mask IoU (utilities in `evaluate.py` / `utils.py`).
- **Image-level** — the operationally important "does this frame contain damage
  at all?" decision (precision/recall/accuracy).
- **No labels** — the evaluator says so honestly and produces a qualitative
  prediction gallery instead of inventing numbers.

**Class-agnostic** matching is a deliberate choice: for a first prototype the
meaningful question is "did we localise the damage", not "did we name the
subtype correctly".

### 5.1 Results on synthetic data (pipeline verification only)

Dataset: 10 generated images (7 with damage, 3 clean), 11 ground-truth boxes,
plus **unlabelled distractors** (soft shadows, biofouling-like speckle) added on
purpose to stress the baseline.

| Metric (classical baseline) | Value |
|---|---|
| Precision | **1.00** |
| Recall | **0.91** |
| F1 | **0.95** |
| AP (IoU 0.30) | **0.90** |
| Image-level accuracy | **1.00** (7/7) |
| False positives | 0 |
| False negatives | 1 |

**How to read this:** the near-perfect score is *expected and uninformative
about the real world*. The synthetic damage is trivially dark against a clean
mid-toned mesh, so a darkness cue separates it almost perfectly. The single miss
(below) is the interesting part. On real footage — shadows, biofouling,
turbidity, fish — the very same cues will produce **many false positives and
misses**. This result confirms the *plumbing*, not the *capability*.

### 5.2 Failure case (real, from the demo)

The one false negative is illustrative: a **hole and an adjacent tear** were
merged by the morphological closing into a single detected region. That
detection matched one ground-truth box, so the second annotation was scored as a
**miss**. The match overlay is saved automatically to
`reports/assets/failures/` and `outputs/eval/failures/`.

This is a genuine, generalisable weakness of contour-based methods: **adjacent
defects merge**, and a single box cannot represent two. It is one concrete
argument for moving to **segmentation** (mask per region) on real data.

### 5.3 Real-data validation on SOLAQUA (no labels — false-positive & anomaly behaviour)

To go beyond synthetic data, the prototype downloads and processes real ROV
footage from **SOLAQUA** (`scripts/fetch_solaqua.py` → public API → ROS `.bag`).
For this report I used the smallest video bag (`2024-08-22_14-06-43_video.bag`,
~916 MB, RGB topic `/image/compressed_image/data`) and extracted **38 frames** of
**undamaged** net (green mesh, biofouling, a rope/seam, turbid green water).

Because the nets are undamaged and unlabelled, there is **no recall to measure**.
What we *can* measure is how a naive detector behaves on real intact net — i.e.
its **false-alarm rate**:

| Classical baseline on 38 real undamaged frames | Value |
|---|---|
| Total detections (all are false alarms) | **114** |
| Mean detections per frame | **3.0** |
| Frames with ≥1 false alarm | **29 / 38 (76%)** |

**Interpretation.** The baseline fires on darker see-through cells (net viewed
obliquely), biofouling and lighting variation. On real footage the hand-tuned
darkness/texture cues that scored ~1.0 on synthetic data produce a **76%
false-alarm frame rate** on *undamaged* net. This is the honest, important
result: it quantifies why the classical method is a triage/sanity tool, not a
detector, and why a learned model **and an agreed FP/FN operating point** are
required. Overlays are saved to `outputs/real_analysis/`.

**Anomaly model.** Trained on the 38 normal frames (`scripts/train_anomaly.py`),
the patch-Mahalanobis model's heatmaps concentrate on **biofouling clumps and
lighting hotspots** — the genuinely out-of-distribution content — while the
regular mesh scores low. At a 99th-percentile training threshold it flags ~half
the frames somewhere. Since all frames are undamaged, these flags are again
*candidate false alarms*; the value is that the method **localises "unusual
net"** without any labels, which is the right primitive for a label-scarce start.
The honest caveat stands: it cannot yet distinguish damage from heavy fouling —
that separation needs labelled examples. Heatmaps are in `outputs/anomaly/`.

### 5.4 Reducing classical false positives (measurable improvement on real net)

The 76% real-frame false-alarm rate (§5.3) was the concrete weakness, so I fixed
it as a measurable engineering task. The cause was a cue that let *intact-but-dark*
net cells through. The fix is a **texture gate**: a real opening lacks mesh, so
its internal edge density is well below the image median (measured: holes ≈0.64×,
net ≈0.99× median), whereas a dark net patch keeps its fibre grid. Applying this
gate always (not bypassing it for dark regions) gives a tunable recall/false-alarm
trade-off on real net:

| `max_region_density_ratio` | False detections on undamaged frames | Recall on composited damage |
|---|---|---|
| untuned (prior) | 114 | — |
| 0.82 (default) | 62  (**−46%**) | 0.64 |
| 0.70 | 28  (**−75%**) | 0.51 |

An honest ceiling emerges: a hand-crafted heuristic tops out around **F1 ≈ 0.5**
on realistic backgrounds. That is precisely why a learned model is needed.

### 5.5 A trainable detector on realistic proxy data + three-method comparison

To compare methods *quantitatively* without real damage labels, I composited
plausible damage onto **real** SOLAQUA frames (`compose.py`): real background
(texture, biofouling, lighting), synthetic but labelled holes/tears. Train/val/
test use **disjoint real backgrounds**; a second, separately-downloaded clip
(bag2) is a held-out cross-clip test. A YOLOv8n was trained on this set and all
three methods evaluated with the shared metrics (IoU 0.30, class-agnostic):

| Method | In-clip F1 | In-clip AP | Cross-clip F1 (held-out clip) |
|---|---|---|---|
| Classical (OpenCV) | 0.50 | 0.55 | 0.67 |
| Anomaly (patch-Mahalanobis) | 0.12 | 0.02 | 0.05 |
| **YOLOv8** | **0.97** | **0.97** | **0.98** |

**What this shows (honestly):**
* The **ML approach is decisively the right path** — YOLO beats the heuristic by
  ~2× F1 and, importantly, **does not collapse on a held-out clip**.
* The **anomaly model is a poor box-localiser** (it is an image-level *screen*,
  not a detector) — useful as a label-free triage, not as the detector.

**What this does NOT show (the caveats that matter):**
* The damage *appearance* comes from **one synthetic generator**, identical in
  train and test. So YOLO's score partly reflects that the damage model is
  in-distribution — it measures "learns this damage model on real backgrounds and
  transfers across clips", **not** "detects real damage".
* bag2 is the **same site, day and camera** (a different traversal), so cross-clip
  ≠ cross-site. True generalisation needs different sites/seasons/cameras.
* Therefore **no real-world damage-detection reliability is claimed.** These are
  strong *relative* results on a realistic proxy, and the exact pipeline trains on
  real labels unchanged — the moment ScaleAQ provides labelled damage, this
  becomes a real evaluation.

### 5.6 Is this "industrial-grade"? An honest answer

The user asked for something reliable enough to be an industrial product. The
*engineering* here is built to that standard: unified inference facade, FastAPI
service, batch/video/bag runner with run manifests, an interactive viewer, tests
and CI, graceful dependency degradation, and a real-data ingestion path. But
**industrial reliability of damage detection is proven by validation on real
damage, which this project does not have.** The responsible claim is: *a
production-shaped prototype with a strong, honestly-evaluated proxy result and a
drop-in slot for real labelled data* — not a certified detector. Shipping it as
"reliable" without that validation would be the one thing this project refuses to
do.

### 5.7 Stronger methods and adversarial rigor

Several upgrades push the prototype well past a first pass, each measured honestly —
a foundation-model anomaly detector, an adversarial "is it cheating?" suite,
temporal confirmation, a caught-and-fixed segmentation regression, a stronger-
augmentation experiment that *failed*, a detector∧segmenter agreement ensemble that
resolves it, and a self-supervised-vs-supervised feature ablation.

**Foundation-model anomaly (PatchCore).** Replacing the 6-feature hand-crafted
model with patch embeddings from a pretrained ResNet (memory bank + nearest-
neighbour distance) lifts label-free anomaly localisation from **F1 0.12 → 0.78**
on the composite test (threshold picked on val, reported on test). Updated
label-free → supervised ladder on the same test set:

| Method | Labels? | F1 |
|---|---|---|
| Hand-crafted anomaly | no | 0.12 |
| Classical heuristic | no | 0.50 |
| PatchCore (foundation) | no | **0.78** |
| YOLOv8 | yes | **0.97** |

**Adversarial "is it cheating?" evaluation.** A high F1 on synthetic-on-real is
only trustworthy if the model isn't keying on compositing artifacts. The tests
that try to break that:

* *False positives on REAL undamaged net* (no damage → every fire is a false
  alarm): **0%** frame-FP on bag1 (its own training backgrounds), **0%** on bag2,
  **1%** on a different day. A model cheating on background/artifacts would light
  up here; it does not.
* *Generalisation*: damage recall holds at **F1 ≈ 0.97** across in-clip,
  cross-clip and different-day backgrounds.
* *FROC* (different-day): **0 false positives per undamaged frame at conf ≥ 0.4**
  with recall ≈ 0.97 — a clean operating point.

This rules out the cheapest cheating and characterises the operating curve. It
still does **not** prove real-damage performance — the damage appearance is
synthetic — but it materially strengthens the proxy result. Hard negatives
(dark-but-textured distractors) are baked into the training data so the detector
must use damage cues (uniform see-through fill + frayed rim), not just "is dark".

**Temporal confirmation.** Real damage is static on the net; clutter (fish,
glints) is transient. A lightweight IoU tracker that confirms only detections
persisting over ≥3 frames removes **~70%** of transient false alarms on real
undamaged video (30 → 9 detections over 120 contiguous frames), at a few frames'
latency — exactly the post-processing an operator-facing tool needs.

**Segmentation, and a finding the eval caught.** A YOLOv8n-**seg** model trained
on the harder photorealistic data (masks; hard negatives) gets box F1 0.95 and
mean mask IoU 0.66 in-distribution. But the adversarial suite **caught a
regression**: this model fires on **31%** of *different-day* undamaged frames
(vs **1%** for the simpler detector) and its different-day recall is lower
(F1 0.77 vs 0.97). "More realistic + harder" did **not** mean "more robust" —
the subtler, seamless-blended damage produced a fuzzier concept, and the model
(trained for fewer epochs) generalises worse out-of-distribution. The honest
conclusion: the simpler `yolo_damage_v1` remains the more robust *detector*; the
seg model adds masks but needs more epochs / more diverse damage appearance to
match it. This is the evaluation doing its job — preventing the overclaim that
the newer, fancier model is strictly better.

**Closing the loop — the fix, measured.** The diagnosis (single-clip backgrounds
+ subtle damage → fuzzy, non-robust concept) predicts a fix: train on *diverse*
backgrounds. So `yolo_damage_seg_v3` was trained on a **multi-clip** dataset
(bag1 + bag2 backgrounds, per-instance appearance jitter), holding out the
different-day clip. Result on that held-out day:

| Different-day (held out) | seg v2 (1 clip) | **seg v3 (multi-clip)** | det v1 |
|---|---|---|---|
| Undamaged false-positive rate | 31% | **18%** | 1% |
| Damage recall (F1) | 0.77 | **0.91** | 0.98 |
| FROC: recall at ~0 FP/frame | 0.45 | **0.93** | 0.97 |

Multi-clip training **cut OOD false positives 31% → 18%, lifted held-out recall
0.77 → 0.91**, and made the operating curve usable (at conf 0.7: ~0 FP/frame with
0.93 recall, where v2 collapsed to 0.45). The diagnosis was correct and the fix
is real — **but partial**: it still does not match the box detector's 1% OOD
false-positive rate. Honest takeaway: diverse data closed most of the gap;
segmenting subtle (synthetic) damage remains harder to generalise, and `det v1`
stays the most robust *detector*. This is the senior loop in full — *found a
regression in my own model, diagnosed it, fixed it with diverse data, and
re-measured* — reported with its residual gap intact rather than rounded up.

**Pushing further — a hypothesis that did *not* pan out (`seg v4`).** The residual
gap (v3's 18% different-day FP vs det v1's 1%) invites an obvious next lever:
since only two training clips exist and the third day is held out, *simulate*
day-to-day variation with **strong photometric augmentation** (hue/brightness
jitter ≈3–4× the YOLO default, plus rotation/perspective) and add more
pure-negative frames (26% vs 15%) on a larger, multi-clip set. Hypothesis:
lighting-invariant features + more "this is just net" examples should cut the OOD
false positives. Trained to convergence (`yolo_damage_seg_v4`, 80 epochs), it did
**not**:

| Different-day (held out) | seg v2 | seg v3 | **seg v4 (+strong aug)** | det v1 |
|---|---|---|---|---|
| Undamaged false-positive rate | 31% | **18%** | 22% | 1% |
| Damage recall (F1) | 0.77 | **0.91** | 0.87 | 0.98 |
| FROC: recall at ~0 FP/frame | 0.45 | 0.93 | 0.93 | 0.97 |

v4 segments *in-distribution* damage better than v3 (mask mAP50 0.967 vs ≈0.94)
but generalises no better out-of-distribution — slightly worse at the default
threshold, tied on the FROC operating curve. The honest reading: augmenting only
two source clips cannot manufacture genuine day-to-day diversity (water colour,
turbidity, sun angle), and pushing the jitter harder made the in-distribution fit
sharper without transferring. Closing the residual gap needs a real *third*
training day (deliberately held out here for an honest OOD test) or a less fuzzy
damage signal — not more augmentation. **`seg v3` remains the best segmenter and
`det v1` the most robust detector.** I am reporting this as a negative result
rather than quietly dropping the run — the experiment that fails is part of the
record.

**What actually fixed it — capacity + real data (`seg-gpu`, on an A100).** The
nano models were small (3.3 M params). Scaling to **YOLOv8s-seg** (11.8 M) and
training on **three real clips** on a GPU produced a standalone segmenter that, on
the held-out different day, fires on just **1%** of undamaged frames — matching the
box detector — with recall F1 **0.958** and masks:

| Held-out different day | seg v3 (nano) | seg v4 (nano +aug) | **seg-gpu (YOLOv8s)** | det v1 |
|---|---|---|---|---|
| Undamaged false-positive rate | 18% | 22% | **1%** | 1% |
| Damage recall (F1) | 0.91 | 0.87 | **0.958** | 0.976 |
| FROC: ~0 FP/frame at conf | 0.7 | 0.8 | **0.6** | 0.7 |

The lesson is specific and honest: the segmentation OOD gap was substantially a
**model-capacity** limitation of the nano, not an inherent flaw of segmentation —
and it was closed with *more capacity on the same real clips*, not with
augmentation tricks. (The damage is still synthetic, so this is a strong proxy
result, not validated real-damage performance; and more real *days/sites* would
push it further.) This makes `seg-gpu` the best standalone segmenter; the ensemble
below remains a zero-retraining alternative.

**The practical fix — an agreement ensemble (best of both, measured).** The
det-vs-seg tension has a clean resolution that needs *no* retraining: let the
robust detector **propose** and the segmenter **confirm** (keep a `det v1` box only
if `seg v3` also fires on it, box IoU ≥ 0.3), in `src/netinspect/ensemble.py`. A
region flagged by two independently-trained models is far less likely to be a
shared spurious cue. Measured head-to-head (`scripts/eval_ensemble.py`):

| Held-out different day | det v1 | seg v3 | **ensemble (det∧seg)** |
|---|---|---|---|
| Undamaged false-positive rate | 1% | 18% | **1%** |
| Damage recall (F1) | 0.976 | 0.912 | **0.976** |

The ensemble keeps the detector's **1%** out-of-distribution false-positive rate
*and* its **0.976** recall — recovering the robustness the standalone segmenter
lost — while contributing segmentation masks wherever the two models agree. That is
the deployable answer to "masks *or* robustness?": **both**, at the cost of a second
forward pass. It is wired into the inference facade and the web console as its own
method.

![ensemble false-positive suppression](../docs/images/ensemble_fp_suppression.jpg)

*Held-out different day. Left: `seg v3` raises a false alarm on an instrument
housing; right: the det∧seg ensemble stays clean — same frame, no retraining.
On actual (composited) damage all three still fire — see
`docs/images/ensemble_on_damage.jpg`.*

**Self-supervised features — a label-free probe of the deferred SSL idea.** A
documented next step is self-supervised pretraining on the unlabelled SOLAQUA
frames. Full pretraining is GPU-bound, but one executable slice of the question
runs on CPU: *do off-the-shelf self-supervised features transfer better than
supervised ones for our one-class anomaly detector?* The PatchCore pipeline's
only learned component is its **backbone**, so I swapped it — ImageNet-supervised
**ResNet18** vs self-supervised **DINOv2 ViT-S/14** (`dino_backbone.py`) — holding
everything else fixed, and measured the fair, threshold-free metric: image-level
**AUROC** for separating damaged from undamaged frames.

| Backbone (same detector) | in-clip AUROC | different-day AUROC | undamaged FP frame-rate (in-clip) |
|---|---|---|---|
| ResNet18 (ImageNet-supervised) | 0.98 | **0.99** | 68% |
| DINOv2 ViT-S/14 (self-supervised) | **1.00** | 0.96 | **32%** |

The honest read: **as a frame-level screen both are strong (AUROC 0.96–1.00)** —
the anomaly route *can* flag damaged frames even though it draws poor boxes (an
image-level screen, not a localiser; tight boxes need the calibrated threshold of
the PatchCore F1-0.78 result above). Self-supervised features are **competitive
and visibly cleaner** — DINOv2 is perfect in-clip and fires on roughly half as
many undamaged frames — but they do **not** beat supervised features on the
hardest different-day AUROC. And the real bottleneck is **threshold transfer, not
the backbone**: at the uncalibrated 2×-median threshold *both* backbones fire on
100% of different-day undamaged frames even though their scores still rank
damaged frames above undamaged ones (high AUROC). So off-the-shelf SSL is **not a
free lunch** here; the gains to chase are (a) SSL pretraining *on SOLAQUA* itself
(domain-matched, the still-deferred GPU step) and (b) a domain-adaptive,
calibrated operating threshold. These are image-level scores on synthetic damage
and undamaged net — they characterise behaviour, not real-damage accuracy.
Reproduce: `scripts/compare_anomaly_backbones.py` (results in
`reports/results/ssl_dino/`).

### 5.8 Closing the out-of-distribution gap — what worked, what didn't, what's left

Pulling the robustness thread together, here is the honest ledger of *every*
lever tried or scoped to close the different-day gap, so the judgement is visible:

| Lever | Status | Effect on different-day false alarms |
|---|---|---|
| Multi-clip training (`seg v3`) | **Done** | 31% → **18%** (most of the regression) |
| Stronger photometric augmentation (`seg v4`) | **Done — failed** | 18% → 22% (no help; reported, not buried) |
| **Bigger model + 3 real clips (`seg-gpu`, YOLOv8s, A100)** | **Done** | 18% → **1%** standalone (recall F1 0.958, *with* masks) — capacity + real diversity did what augmentation couldn't |
| **det∧seg agreement ensemble** | **Done** | 18% → **1%** (matches the detector; keeps masks, no retraining) |
| Confidence-threshold operating point (FROC) | **Done** | `seg v3` at conf ≥0.7 reaches ~0 FP/undamaged frame (recall ~0.93) |
| Temporal confirmation | **Done** (−70% transient FP on real video) | persistence over ≥3 frames removes flicker false alarms |
| Real *multi-day* training | **Data-limited (but see `seg-gpu`)** | SOLAQUA's public feature has only **two days** (3 clips on 08‑22, 2 on 08‑20); 08‑20 is the held-out test, so a genuine third *day* does not exist to add. Notably, `seg-gpu` trained on the 3 *same-day* clips already generalised to the held-out different day at 1% FP — so much of the gap was **model capacity**, not only day-diversity. More real days/sites would still help and are required for validated real-damage performance. |
| Hard-negative mining | **Scoped** | the compositor already injects unlabelled hard negatives; the next step is mining the *real* frames the model false-alarms on (e.g. instrument housings) and adding them — a cheap, targeted loop once a labelling pass exists. |
| Domain normalisation (gray-world WB + CLAHE) | **Done — failed at test time** | measured (`scripts/eval_domain_norm.py`): normalising frames only at inference *hurt* (det v1 different-day FP **0% → 28%**, recall 0.98 → 0.86; seg neutral) — a train/test mismatch. To exploit it the normalisation must be in *both* training and inference (a retrain). Reported, not assumed. |
| Self-supervised pretraining *on SOLAQUA* | **Deferred (GPU)** | off-the-shelf DINOv2 transfer was probed (§5.7); domain pretraining on the unlabelled frames is the GPU-bound next step. |
| Uncertainty / OOD gating | **Done** | `src/netinspect/ood_gate.py` — a PatchCore-score gate calibrated on in-distribution frames flags **6%** of training-clip frames, **0%** of same-site/other-clip, and **100%** of the different DAY for human review (`scripts/eval_ood_gate.py`). A not-yet-certified detector can then run safely: auto-handle familiar frames, defer exactly the unfamiliar ones. |

**The deployable answer today** is the ensemble plus a calibrated high-confidence
operating point and temporal confirmation — together they hold the detector's ~1%
different-day false-alarm rate *with* masks, no new data required. **The honest
ceiling is unchanged:** every number here is on synthetic damage or undamaged net.
"Industry-ready" is gated on **real labelled damage across sites/seasons** and an
agreed false-positive/false-negative operating point — a *data-and-process*
threshold no modelling trick crosses. The engineering is built to receive that
data the moment it exists.

---

## 6. Expected failure modes (on real data)

The classical baseline will, predictably:

- **False-positive** on shadows, dark biofouling, ropes, frame edges, fish, and
  low-light/turbid regions — anything dark or texture-poor looks like damage.
- **False-negative** on damage that is *not* darker than its surroundings
  (bright backscatter behind a hole, partially occluded tears, distant/oblique
  views), and on adjacent defects that merge.
- **Be sensitive to scale and parameters** — the thresholds in
  `configs/baseline.yaml` are tuned to the synthetic data and will not transfer.

The ML baseline removes the hand-tuned cues but introduces its own risks:
**distribution shift** across sites/seasons/cameras, **label noise and
ambiguity**, and **class imbalance** (very few damage examples). These are
data-and-process problems, which is why the recommendations below focus there.

---

## 7. What ScaleAQ would need to provide to make this real

- **Representative footage** across sites, seasons, depths, lighting and camera
  setups — ideally from ScaleAQ camera / Vision systems or ROV streams.
- **Real damage examples** (holes/tears/abnormal regions) **and** abundant
  *normal*-net footage under varied conditions.
- **Metadata** where available: camera type, site, depth, date/time, lighting,
  environment — essential for honest held-out evaluation by site/condition.
- **A labelling guideline + domain feedback**: what counts as damage, class
  definitions, edge cases, and whether boxes or masks are wanted.
- **An agreed error trade-off**: acceptable false-positive vs false-negative
  rates for the intended use (these drive the operating threshold, not a default).
- **The deployment target**: real-time on-ROV, offline review, operator decision
  support, alerting, or integration with existing software (e.g. the Vision
  platform via its open API).

> **What SOLAQUA already covers vs. the gap.** SOLAQUA supplies the *normal*-net,
> varied-condition footage (and is enough to build/validate ingestion,
> preprocessing, false-positive analysis and the anomaly model). The remaining
> gap is the decisive one: **labelled real *damage*** and an agreed error
> trade-off. With even a modest set of labelled damage frames, the YOLO path and
> a quantitative evaluation in this repo become immediately usable.

---

## 8. What a successful summer project would deliver

A realistic summer scope is *not* a finished production model. A strong outcome:

1. A **dataset-understanding report** on real footage (quality, balance, gaps).
2. A **labelling guideline** agreed with domain experts, and a first labelled set.
3. The **classical baseline** retuned on real data as a difficulty probe.
4. A trained **YOLOv8 detection (and/or seg) baseline** with honest evaluation.
5. An **evaluation protocol** with held-out sites/conditions and a clear
   FP/FN trade-off.
6. A **failure-case analysis** and a **recommendation**: detection vs
   segmentation vs anomaly detection for this data.
7. A documented **integration path** and next steps.

This prototype already provides the scaffolding for items 3–7.

---

## 9. Next steps for production

1. **Collect & label** representative data; write and iterate the labelling guide.
2. **Start with a baseline model** (YOLOv8n/s detection), establish honest metrics.
3. **Evaluate on held-out sites/conditions**, not a random split — generalisation
   across environments is the real test.
4. **Define the acceptable FP/FN trade-off** with stakeholders; pick the operating
   point from the confidence sweep accordingly.
5. **Consider segmentation and anomaly detection** as the data dictates (masks for
   irregular damage; one-class methods when mostly normal data is available).
6. **Integrate into the inspection workflow** — offline review first, then
   alerting/real-time if justified — with **human-in-the-loop** confirmation.
7. **Monitor in production** (drift, new failure modes, periodic re-labelling),
   echoing an ML-systems mindset: do not trust a model because it worked once.

---

## 10. Honesty statement

- Pure-synthetic metrics prove only that the code path works.
- SOLAQUA-only results describe **false-alarm / anomaly behaviour on *undamaged*
  net**; not damage-detection accuracy.
- The YOLO F1≈0.97 is on **synthetic damage composited on real backgrounds**,
  with the damage appearance from one generator (in-distribution in train and
  test) and a cross-clip (not cross-site) held-out set. It is a strong *relative*
  result on realistic proxy data — **not** validated real-damage performance.
- **No real-world damage-detection reliability is claimed anywhere.** The
  engineering is production-shaped; the *detector* is not certified — that
  requires validation on real labelled damage.
- The classical baseline is a tunable screen (real-net ceiling ~F1 0.5); the
  anomaly model flags deviation-from-normal, not confirmed damage.
- Generators are clearly marked; SOLAQUA data is downloaded on demand under
  CC BY-SA 4.0 and not redistributed.

---

### References

1. *Evaluating Deep Learning Assisted Automated Aquaculture Net Pens Inspection Using ROV* — arXiv:2308.13826.
2. *Active vision-based real-time aquaculture net pens inspection using ROV* — Scientific Reports (Nature), 2025.
3. *Inspection Operations and Hole Detection in Fish Net Cages through a Hybrid Underwater Intervention System Using Deep Learning* — JMSE 12(1):80, MDPI.
4. *AquaChat: An LLM-Guided ROV Framework for Adaptive Inspection of Aquaculture Net Pens* — arXiv:2507.16841.
5. **SOLAQUA: SINTEF Ocean Large Aquaculture Robotics Dataset** — arXiv:2504.01790; data at https://data.sintef.no/product/dp-7141fcd5-0fb8-4be3-b9ce-e5f7f5bb4a58 (CC BY-SA 4.0). *Used directly by this prototype for real-data ingestion, false-positive analysis and anomaly detection.*

*(References 1–4 are background context for the problem framing; this prototype
does not reproduce or depend on them. Reference 5 is used directly.)*
