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

Verified on this machine (Python 3.14, OpenCV 4.x, Ultralytics 8.x, torch 2.x
CPU, rosbags): the synthetic demo runs end to end, **21/21 unit tests pass**, a
YOLO training + inference smoke test completes (1 epoch — only to prove the path,
metrics meaningless), and the **real SOLAQUA pipeline runs** (downloaded a 916 MB
bag, extracted 38 frames, ran false-positive analysis and the anomaly model).

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

- Synthetic metrics prove only that the code path works.
- SOLAQUA results describe **false-alarm / anomaly behaviour on *undamaged*
  net**; they are **not** damage-detection accuracy (no damage, no labels).
- No real-world damage-detection accuracy is claimed anywhere.
- The classical baseline is a sanity-check, not a validated detector (76%
  false-alarm frame rate on real undamaged net).
- The anomaly model flags deviation-from-normal, not confirmed damage.
- The ML (YOLO) path is runnable but untrained on real *labelled* damage data.
- The synthetic generator is clearly marked; SOLAQUA data is downloaded on
  demand under CC BY-SA 4.0 and not redistributed.

---

### References

1. *Evaluating Deep Learning Assisted Automated Aquaculture Net Pens Inspection Using ROV* — arXiv:2308.13826.
2. *Active vision-based real-time aquaculture net pens inspection using ROV* — Scientific Reports (Nature), 2025.
3. *Inspection Operations and Hole Detection in Fish Net Cages through a Hybrid Underwater Intervention System Using Deep Learning* — JMSE 12(1):80, MDPI.
4. *AquaChat: An LLM-Guided ROV Framework for Adaptive Inspection of Aquaculture Net Pens* — arXiv:2507.16841.
5. **SOLAQUA: SINTEF Ocean Large Aquaculture Robotics Dataset** — arXiv:2504.01790; data at https://data.sintef.no/product/dp-7141fcd5-0fb8-4be3-b9ce-e5f7f5bb4a58 (CC BY-SA 4.0). *Used directly by this prototype for real-data ingestion, false-positive analysis and anomaly detection.*

*(References 1–4 are background context for the problem framing; this prototype
does not reproduce or depend on them. Reference 5 is used directly.)*
