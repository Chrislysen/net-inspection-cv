"""Render 'in action' comparison images for the README/report.

Shows the ensemble win visually on the held-out different *day*:
* an UNDAMAGED frame where the segmentation model raises a false alarm but the
  det-gated ensemble stays clean (false-positive suppression);
* a DAMAGED (composited) frame where det, seg and ensemble all localise the damage.

Outputs to docs/images/. Needs the det + seg weights and the SOLAQUA frames.
"""
from __future__ import annotations

import argparse

import _common  # noqa: F401
import numpy as np

from netinspect.ensemble import EnsembleConfig, combine
from netinspect.model_baseline import YoloConfig, load_model, predict_image
from netinspect.utils import ensure_dir, get_logger, list_images, read_image
from netinspect.visualize import overlay_boxes, side_by_side

LOGGER = get_logger()


def _banner(img: np.ndarray, text: str) -> np.ndarray:
    cv2 = __import__("cv2")
    h, w = img.shape[:2]
    bar = np.full((28, w, 3), 22, np.uint8)
    cv2.putText(bar, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    return np.concatenate([bar, img], axis=0)


def _preds(model, img, conf):
    return [b for b in predict_image(model, img, YoloConfig(conf=0.01, imgsz=480)) if b.score >= conf]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--det", default="models/yolo_damage_v1.pt")
    ap.add_argument("--seg", default="models/yolo_damage_seg_v3.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="docs/images")
    args = ap.parse_args()

    det_model, seg_model = load_model(args.det), load_model(args.seg)
    cfg = EnsembleConfig(det_conf=args.conf, seg_conf=args.conf, mode="agree")
    out = ensure_dir(args.out)

    # 1) Undamaged different-day frame where seg fires but the ensemble does not.
    chosen = None
    for p in list_images("data/processed/solaqua_diffday"):
        img = read_image(p)
        det = _preds(det_model, img, args.conf)
        seg = _preds(seg_model, img, args.conf)
        ens = combine(det, seg, cfg)
        if len(seg) > 0 and len(ens) == 0:        # seg false alarm, ensemble clean
            chosen = (img, seg, ens)
            LOGGER.info("FP-suppression example: %s (seg %d -> ensemble %d)",
                        p.name, len(seg), len(ens))
            break
    if chosen:
        img, seg, ens = chosen
        left = _banner(overlay_boxes(img, preds=seg), f"seg v3: {len(seg)} FALSE alarm(s) on undamaged net")
        right = _banner(overlay_boxes(img, preds=ens), f"ensemble (det^seg): {len(ens)} - clean")
        from netinspect.utils import write_image
        write_image(out / "ensemble_fp_suppression.jpg", side_by_side(left, right))
        print(f"Wrote {out/'ensemble_fp_suppression.jpg'}")
    else:
        LOGGER.warning("No seg-fires-but-ensemble-clean frame found; skipping FP example.")

    # 2) Damaged different-day frame: det vs seg vs ensemble all localise.
    imgs = list_images("data/processed/diffday_composite/images/test")
    for p in imgs:
        img = read_image(p)
        det = _preds(det_model, img, args.conf)
        seg = _preds(seg_model, img, args.conf)
        ens = combine(det, seg, cfg)
        if det and seg and ens:
            from netinspect.utils import write_image
            panel = side_by_side(
                _banner(overlay_boxes(img, preds=det), f"det v1: {len(det)}"),
                side_by_side(_banner(overlay_boxes(img, preds=seg), f"seg v3: {len(seg)}"),
                             _banner(overlay_boxes(img, preds=ens), f"ensemble: {len(ens)}")))
            write_image(out / "ensemble_on_damage.jpg", panel)
            print(f"Wrote {out/'ensemble_on_damage.jpg'}")
            break


if __name__ == "__main__":
    main()
