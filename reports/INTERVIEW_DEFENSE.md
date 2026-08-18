# Defend this project — technical Q&A (code-grounded)

A study sheet of the questions a sharp aquaculture-technology engineer is most likely to ask,
with precise answers tied to the actual implementation. If you can derive each of
these from scratch, the project is a strong asset; if you can't, learn it before
the interview — the honesty narrative collapses the moment depth is faked.

Every claim below traces to a file in this repo. The single non-negotiable: all
damage is **synthetic**, so no number here is validated real-world performance.

---

### 1. Why does the classical baseline false-alarm so much on real net? (`classical_baseline.py`)
It combines two cheap cues: an **absolute-darkness** cue (luminance below
`mean − k·σ` on the white-balanced, pre-CLAHE image) and a **low-edge-density**
cue (Canny edge density below a fraction of the image-wide median). On real net,
*intact* mesh seen at an oblique angle or in shadow is genuinely darker than
average, so the darkness cue fires — there's no semantic notion of "damage". I cut
the false-alarm rate with a **texture gate**: a real opening has near-zero internal
edge density, while a dark *net* patch still has its fibre grid. Measured on the
composite set, holes sit at ~0.64× the median edge density vs ~0.99× for net, so
rejecting candidates above `max_region_density_ratio` of the median removes the
intact-dark-net false positives. Lowering that gate trades recall for fewer false
alarms (e.g. 0.82 → −46% false detections, 0.70 → −75%, at the cost of recall).
**Honest caveat:** I tried an FFT-periodicity gate too; it did *not* separate holes
from net on real data (it actually scored holes *higher*), so I disabled it — a
small example of measuring before trusting an idea.

### 2. How does PatchCore work, and why does it beat the hand-crafted anomaly model? (`patchcore.py`)
The hand-crafted model (`anomaly.py`) describes each patch with ~6 numbers (Lab
colour stats, contrast, edge density) and fits one Gaussian → Mahalanobis distance.
It localises damage poorly (F1 0.12) because 6 hand-picked features can't capture
"what intact net looks like". **PatchCore** instead takes patch embeddings from a
**pretrained ResNet18** (concatenated `layer2`+`layer3` feature maps, with a 3×3
neighbourhood pool for locality), builds a **memory bank** of normal-net patch
embeddings (coreset-subsampled to keep it small), and scores a test patch by its
**nearest-neighbour distance** to that bank — far from every normal patch ⇒
anomalous. ImageNet features encode texture/structure far richer than 6 scalars,
so F1 jumps to **0.78** label-free. **Two honest points:** (a) the threshold is
picked on a *validation* split and reported on *test* (no test-tuning); (b) it's a
novelty detector — train it on too few / too narrow normal frames and it flags
*background* novelty, not damage (I hit exactly this bug and fixed it by training
on backgrounds matched to the test distribution).

### 3. Why isn't the YOLO F1 0.97 real performance?
Because the damage is **composited by one generator**, identical in distribution
across train and test. The model can be partly keying on the *compositing
signature* (uniform see-through fill, frayed-edge style) rather than real damage
cues, and real holes look different (bright backscatter, partial occlusion, fish
behind). The cross-clip/different-day sets vary the **background**, not the damage
distribution, and they're the same site/camera. So the only defensible claim is:
*"it learns this damage model on real backgrounds and transfers across
backgrounds"* — **not** *"it detects real damage."* Validating that needs real
labelled damage; the pipeline trains on it unchanged.

### 4. How does the adversarial evaluation rule out cheating, and what did it find? (`adversarial_eval.py`)
Real undamaged net has *no* dark see-through holes, so a damage detector should
**rarely fire** on it. The test runs the trained model on real undamaged frames —
including the very backgrounds it trained on — and counts detections (every one is
a false alarm). The det model fired on **0%** of bag1/bag2 frames and **1%**
different-day, holding F1≈0.97; a model cheating on background/artifacts would light
up here. The FROC curve (FP-per-undamaged-frame vs recall across confidence) shows
**0 FP/frame at conf ≥ 0.4 with recall ~0.97** — a clean operating point. Crucially,
the same suite **caught a regression in my own seg model** (31% different-day false
positives), which I diagnosed and fixed rather than hid (Q6).

### 5. Why IoU 0.30 and class-agnostic matching? (`evaluate.py`)
For a first prototype the meaningful question is *"did we localise the damage"*, not
*"did we name the subtype"* — so matching is class-agnostic. IoU 0.30 is lenient
because damage is small and irregular and a tight box isn't the point yet; I also
report **COCO mAP@[.5:.95]** (`coco_map`) so the strict-localisation number is
visible too, not hidden behind a lenient threshold.

### 6. Why did the segmentation model regress out-of-distribution, and how did you fix it?
The seg v2 model was trained on *harder, more photorealistic* damage (seamless-
blended, subtler) for *fewer* epochs on a *single* background clip. Result: a
fuzzier "damage" concept that, on a **different day**, fired on 31% of undamaged
frames — worse than the simpler det v1 (1%). Diagnosis: subtle damage + low
background diversity → the model latched onto cues that also appear in different-day
net. Fix: retrain on a **multi-clip dataset** (bag1 + bag2 backgrounds) with
per-instance appearance jitter, so the model sees diverse real backgrounds and can't
rely on clip-specific cues. The held-out **different-day** clip measures whether it
worked. *(Result: see `reports/results/` — reported honestly whether or not it fully
closed the gap.)* The meta-point: the evaluation **found** the regression; the fix is
**measured**, not asserted.

### 7. ONNX wasn't faster than PyTorch — so what did you optimise? (`export_onnx.py`)
Nothing, on this CPU dev box — and I say so. ONNX Runtime (184 ms) was *slower* than
Ultralytics' optimised PyTorch predict (107 ms) here. ONNX export is the **portable
hand-off**: the real speed-up is **TensorRT with FP16/INT8 on the deployment device**
(e.g. a Jetson on the ROV), which needs the target hardware and INT8 calibration
frames — so it's **documented, not faked**. Claiming an on-device number I didn't
measure would be exactly the dishonesty this project avoids.

### 8. The false-positive / false-negative trade-off — how would you set it?
Asymmetric costs: a **false negative** (missed damage) can mean a **fish escape** —
financial, environmental, and regulatory (Norwegian fish-farm structures fall under
**NS 9415**; escapes are reportable). A **false positive** wastes operator/ROV time.
So you don't pick a default threshold — you pick the operating point **with
stakeholders** from the FROC/PR curve given agreed costs, likely biasing toward
recall with a **human-in-the-loop** confirming flags, plus **temporal persistence**
(Q9) to suppress nuisance alarms without dropping real ones.

### 9. How does temporal confirmation reduce false alarms, and what's the cost? (`temporal.py`)
Real damage is **static relative to the net**; clutter (a passing fish, a light
glint) is transient. A lightweight IoU tracker associates detections frame-to-frame
and only **confirms** a track seen in ≥`min_hits` frames — removing flicker. Measured:
**−70%** transient false alarms on real undamaged video. Cost: a few frames of
**latency** before a detection is confirmed, which is fine for inspection review.

### 10. When would you use each of the four methods?
**Classical** — fast, explainable triage / data-difficulty probe, no training data.
**Hand-crafted anomaly** — weak; mostly a baseline to beat. **PatchCore** — label-free
screening when you have lots of *normal* footage and no damage labels (the realistic
early situation). **YOLOv8** — the performance path once labelled data exists. They
share one prediction schema and one metric harness (`compare_methods.py`), so the
comparison is apples-to-apples.

### 11. What's the single biggest weakness, and what would you do first with real access?
The whole thing rests on **synthetic damage** — the sim-to-real gap is **unmeasured**
because there's no real labelled damage. So the first thing I'd do is **not** train a
bigger model — it'd be **data**: collect and label a few hundred real damage frames
across sites/seasons/cameras, write a labelling guideline with domain experts, and
**evaluate by held-out site, not random split**. Then the existing
`train_yolo.py` + `compare_methods.py` + adversarial suite turn every proxy number
into a real one, unchanged.

### 12. Convince me this isn't just a fast pile of features.
Fair challenge. The depth is in the *evaluation and the decisions*, not the feature
count: the texture-gate derivation (Q1), the PatchCore calibration-on-val discipline
(Q2), the adversarial suite that **falsified my own model** (Q4/Q6), and the
`RESEARCH_SYNTHESIS.md` decision log where I **rejected** plausible-but-wrong advice
(debris-as-damage, unverified licences). Any one module I can take three "why"s deep;
the breadth exists to make the *comparison* fair, not to pad a feature list.

---

**One-sentence pitch:** *"I compared four ways to detect net damage on real SINTEF
footage, then spent most of my effort trying to prove my own best model was cheating
— because in aquaculture a false 'net is fine' is a fish-escape risk, and I'd rather
ship honesty than a number."*
