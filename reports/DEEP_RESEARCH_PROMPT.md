# Deep-research prompt — optimizing `net-inspection-cv`

Paste the block below into ChatGPT (deep research / o-series with browsing). It is
self-contained: the GitHub repo is private, so the prompt carries its own summary.

---

````text
ROLE
You are a senior computer-vision research engineer and aquaculture-technology
domain expert. You are rigorous, cite primary sources, and never overclaim.

OBJECTIVE
Produce an evidence-based, prioritized roadmap to evolve the existing prototype
described below into the strongest possible applied-ML project for aquaculture
fish-farm NET-DAMAGE detection — optimizing for BOTH:
 (a) a credible engineering path toward a real, deployable system, and
 (b) a standout portfolio piece for a ScaleAQ summer-student ML/computer-vision
     role (ScaleAQ is a Norwegian aquaculture-technology company: cameras, sensors,
     ROVs, feeding/monitoring, and the "Vision" software platform).
The repository is PRIVATE — rely on the summary below plus your own research.

CONTEXT — WHAT ALREADY EXISTS (a working prototype)
- Goal: flag holes / tears / abnormal regions in underwater fish-farm net footage
  to SUPPORT (not replace) human inspection.
- Four detection approaches, compared with shared metrics:
  1) classical OpenCV heuristic (darkness + low-edge-density + FFT/texture gates);
  2) hand-crafted patch-Mahalanobis anomaly model;
  3) PatchCore (pretrained ResNet18 features + memory bank) — label-free anomaly;
  4) supervised YOLOv8 detection + YOLOv8-seg (masks).
- Data:
  * synthetic net images (pipeline test only);
  * SOLAQUA (SINTEF Ocean, arXiv:2504.01790, CC BY-SA 4.0): REAL ROV footage of
    operational salmon pens, but nets are UNDAMAGED and UNLABELLED; ROS .bag with
    mono+stereo camera and multibeam sonar; ingested via the pure-Python `rosbags`;
  * "synthetic damage composited onto REAL net frames" (seamless blending, frayed
    fibres, + UNLABELLED hard negatives) to get trainable, labelled, comparable data.
- Headline PROXY results (synthetic damage on real backgrounds, IoU 0.30):
  damage-localisation F1 = anomaly 0.12 -> classical 0.50 -> PatchCore 0.78
  (label-free) -> YOLOv8 0.97. Holds ~0.97 in-clip / cross-clip / different-day.
- Adversarial "is it cheating?" evaluation: the trained YOLO fires on 0% of REAL
  UNDAMAGED frames (incl. its own training backgrounds), 1% on a different day;
  FROC shows 0 false positives/frame at conf>=0.4 with recall ~0.97. The eval also
  CAUGHT a regression: the YOLOv8-seg model (harder/photorealistic data, fewer
  epochs) fires on 31% of different-day undamaged frames — i.e. "fancier" was less
  robust out-of-distribution.
- Temporal reasoning (IoU tracker, confirm detections persisting >=3 frames) removes
  ~70% of transient false alarms on real undamaged video.
- Engineering: unified inference facade; FastAPI + interactive web console; Streamlit
  viewer; batch/video/bag runner; COCO->YOLO adapter; ONNX export + latency benchmark
  (ONNX Runtime was NOT faster than PyTorch on CPU here); 35 tests; GitHub Actions CI;
  committed models. Stack: Python 3.11-3.14, OpenCV, PyTorch/torchvision, Ultralytics
  YOLOv8, scikit-learn, rosbags, FastAPI.
- THE HARD LIMIT (non-negotiable honesty): ALL damage is SYNTHETIC (one generator).
  NO validated real-world damage-detection performance is claimed. The single
  missing ingredient is REAL LABELLED net-damage footage.

RESEARCH QUESTIONS (answer each with evidence + citations)

1. DOMAIN & REQUIREMENTS
   - How is fish-farm net inspection actually done today (divers vs ROV vs fixed
     cameras)? What damage types/sizes matter operationally, and what are the
     escape-risk and regulatory drivers (e.g. Norwegian standard NS 9415, fish-escape
     reporting)? What false-positive/false-negative trade-offs do operators accept?
   - What are ScaleAQ's actual products, camera/ROV/sensor systems, and the "Vision"
     platform + open APIs, and where would a net-damage CV capability plausibly fit
     their roadmap? What would most impress their engineers in a summer-student?
   - What does REAL net damage look like underwater (vs synthetic dark holes) under
     biofouling, turbidity, lighting, distance, and net deformation?

2. DATA (the #1 gap — be exhaustive and verify)
   - Find and tabulate EXISTING datasets of NET / cage / aquaculture-structure
     DEFECTS (not just marine debris): name, size, classes, modality, access URL,
     and VERIFIED licence. Investigate the lines of work around: improved Mask R-CNN
     net-damage detection; Akram et al. multi-scale net-defect segmentation (2023)
     and ROV net-pen inspection benchmarks (2025); any "NDv1/NDv2", Khalifa
     University, or LABUST net datasets; SeaClear; TrashCan; Trash-ICRA19. For each,
     state whether you could actually access it and its true licence (flag
     "unverified" if unsure — do NOT assume).
   - How have researchers obtained and LABELLED net-damage data? Class taxonomies,
     labelling guidelines, inter-annotator agreement, tooling (CVAT etc.).
   - Best practice for SYNTHETIC / sim-to-real underwater-defect data: domain
     randomization, cut-paste augmentation, GAN/diffusion defect synthesis,
     Blender/Unreal net rendering, physically-based underwater image simulation.

3. METHODS / STATE OF THE ART (2023-2026, with aquaculture/underwater evidence)
   - Best detectors/segmenters for SMALL, IRREGULAR, SPARSE underwater defects:
     YOLO variants vs RT-DETR vs Mask R-CNN improvements vs SAM/SAM2-based pipelines.
   - Anomaly-detection SOTA beyond PatchCore for "normal net with biofouling
     variation": EfficientAD, PaDiM, FastFlow, Reverse Distillation, DRAEM, WinCLIP /
     zero-shot. Which best tolerate benign variation (fouling, lighting)?
   - Foundation models: DINOv2 features for anomaly; SAM/SAM2 for net-region or
     defect segmentation; CLIP-based zero-shot. Are these worth it here?
   - Net-region segmentation / ROI masking to suppress non-net false positives
     (fish, ropes, background). Multi-modal camera + multibeam-sonar fusion.

4. SIM-TO-REAL & SELF-SUPERVISION
   - Concrete techniques to close and MEASURE the synthetic-to-real gap.
   - How to exploit abundant UNLABELLED real footage (SOLAQUA-style): self-supervised
     pretraining (DINO/MAE) on real net frames, then fine-tune. Evidence it helps.

5. EVALUATION RIGOR
   - Best practice for evaluating under DOMAIN SHIFT (by site/season/camera, not
     random splits); FROC vs PR curves; operating-point selection under asymmetric
     costs; confidence calibration; how to report proxy results without overclaiming.

6. DEPLOYMENT & MLOPS
   - On-ROV / edge inference (e.g. NVIDIA Jetson Orin): TensorRT FP16/INT8, realistic
     latency budgets, INT8 calibration, streaming-video pipelines. Cite real numbers.
   - Integration patterns with an existing camera/vision platform via open APIs.
   - MLOps: data versioning, model registry, drift monitoring, active learning /
     human-in-the-loop, periodic re-labelling.

7. POSITIONING & "WHAT NOT TO DO"
   - Given a summer-student timeframe, what is the highest-leverage sequence? What is
     OVER-ENGINEERING for this context? What are the operational/ethical risks
     (false negatives -> fish escape; false positives -> wasted operator time)?

CONSTRAINTS / ETHOS FOR YOUR RESEARCH
 - Cite PRIMARY sources (papers, dataset pages, vendor docs). Prefer 2023-2026.
 - VERIFY dataset access and licences; explicitly flag anything you could not verify.
 - Clearly separate EVIDENCE-BACKED recommendations from speculation.
 - Prioritise by IMPACT x EFFORT; call out overkill.
 - Preserve the project's core honesty: never recommend claiming real-world
   performance without real labelled damage data.

DELIVERABLE FORMAT
 A structured report with:
  1) Executive summary (the 5 highest-leverage moves);
  2) Domain & ScaleAQ needs;
  3) Data strategy — a TABLE of net-defect datasets (name | size | classes | modality
     | access | VERIFIED licence | notes) + a synthesis/sim-to-real plan;
  4) Methods roadmap — ranked SOTA options with evidence and trade-offs;
  5) Sim-to-real & self-supervision plan;
  6) Evaluation protocol (splits, metrics, operating point);
  7) Deployment & MLOps plan (with real latency/throughput figures where available);
  8) PRIORITISED ACTION PLAN — table: action | why it matters | impact | effort |
     evidence/citation;
  9) "What NOT to do" / over-engineering and risk section;
 10) Open questions and the data ScaleAQ would need to provide.
 Use inline citations throughout and end with a numbered Sources list.
````

---

**Tip:** if your tool allows giving the agent repo access, also grant read access to
the private repo (as the owner) so it can verify the summary above against the code,
the report (`reports/SCALEAQ_PROTOTYPE_REPORT.md`), and the result tables
(`reports/results/`). Otherwise the embedded summary is sufficient.
