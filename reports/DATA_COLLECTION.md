# Data-collection & labelling protocol (what closes the gap)

The single thing standing between this prototype and a real system is **labelled
real net-damage footage**. This is a concrete protocol ScaleAQ could follow to
provide it — designed so the data drops straight into the existing pipeline.

## What to collect
- **Real damage examples:** holes, tears, frayed/abraded sections, deformation,
  and "abnormal regions". Aim for variety in size (small pinhole → large rip),
  depth, and severity. Even **a few hundred** labelled damage frames is enough to
  turn every proxy number in this repo into a real one.
- **Abundant normal net** under varied conditions (already have SOLAQUA; more helps
  the anomaly models and the negative class).
- **Hard negatives:** heavy biofouling, vegetation, shadows, fish occlusion, ropes —
  the things that *look* like damage but aren't. These are what drive false alarms.

## Cover the real variation (so evaluation is honest)
Spread collection across, and **record as metadata**, the axes that cause domain
shift: **site**, **season/date**, **camera/ROV type**, **depth**, **lighting**,
**turbidity**, and **net type/mesh**. This metadata is essential because the model
must be **evaluated by held-out site/condition, not a random split** (a random
split leaks near-duplicate frames and flatters the model).

## Labelling guideline (define with domain experts)
- **Classes:** start simple — a single `damage` class is the most robust first
  target; add `hole / tear / abnormal_region` subtypes only if operationally needed.
- **Geometry:** boxes are enough to start; **polygons/masks** are better for
  irregular damage (the seg path is ready for them).
- **Edge cases to decide up front:** Is heavy biofouling "damage"? Partial/occluded
  holes? Minimum size that matters? Document the call and keep it consistent.
- **Quality:** double-label a subset and measure inter-annotator agreement; ambiguous
  "damage" is itself a finding to feed back to the operating-point discussion.
- **Tooling:** CVAT / Roboflow / Label Studio all export COCO.

## Hand-off → drops straight in
Export labels as **COCO JSON** (the usual annotation-tool output). Then:
```bash
python scripts/convert_coco.py --coco annotations.json --images imgs --out data/processed/real --single-class
python scripts/prepare_data.py  --images data/processed/real/images --labels data/processed/real/labels --out data/processed/real --yolo-split
python scripts/train_yolo.py    --data data/processed/real/yolo/dataset.yaml --epochs 60
python scripts/compare_methods.py --images <held-out-site/images> --labels <held-out-site/labels> \
       --yolo-weights runs/detect/train/weights/best.pt --out outputs/real_eval
python scripts/adversarial_eval.py --yolo-weights runs/detect/train/weights/best.pt --out reports/results/real_adversarial
```
No code changes needed — the COCO adapter, training, comparison, and adversarial
evaluation are already built for exactly this moment.

## Then, and only then
Report performance **by held-out site**, set the false-positive/false-negative
operating point with stakeholders, and update `MODEL_CARD.md` /
`PRODUCTION_READINESS.md` with real numbers. That is the step that moves the
product from ~3/10 toward deployable.
