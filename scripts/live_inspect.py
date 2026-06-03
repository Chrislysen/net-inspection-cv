"""Live net inspection on a camera / RTSP / ROV stream (real-time).

Points the unified inference facade at a *live* source — an RTSP/HTTP IP camera,
a USB webcam, or a video file — runs a detector frame by frame, confirms defects
over time (so an operator gets one alert per *persisting* defect, not per frame),
and optionally flags **out-of-distribution** frames for human review.

    # RTSP / IP camera (e.g. an ROV feed), ensemble detector, show a window:
    python scripts/live_inspect.py --source rtsp://CAMERA_IP/stream \\
        --method ensemble --yolo-weights models/yolo_damage_v1.pt \\
        --seg-weights models/yolo_damage_seg_v3.pt --display

    # USB webcam 0, YOLOv8s-seg, with the OOD gate + event log:
    python scripts/live_inspect.py --source 0 --method yolo \\
        --yolo-weights models/yolo_damage_seg_gpu.pt \\
        --patchcore-model models/patchcore_normal_net --out outputs/live

Honesty: the shipped models learned *synthetic* damage, so on a new camera/site
treat detections as **human-review triage**, not verified alarms — the OOD gate
will usually flag a new domain. Run in shadow mode first; fine-tune on real labels
before alerting (see the README).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import _common  # noqa: F401

from netinspect.classical_baseline import ClassicalConfig
from netinspect.inference import NetInspector
from netinspect.temporal import TemporalConfig, Tracker
from netinspect.utils import ensure_dir, get_logger, optional_import

LOGGER = get_logger()


def _open(source: str):
    cv2 = optional_import("cv2")
    if cv2 is None:
        raise RuntimeError("Live inspection needs OpenCV. Install `.[cv]`.")
    src = int(source) if source.isdigit() else source     # USB index or URL/path
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source} "
                           "(check the RTSP URL / camera index / file path)")
    return cv2, cap


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="RTSP/HTTP URL | USB index (0,1,..) | video file")
    ap.add_argument("--method", default="yolo",
                    choices=["classical", "anomaly", "patchcore", "yolo", "ensemble"])
    ap.add_argument("--yolo-weights", default="models/yolo_damage_v1.pt")
    ap.add_argument("--seg-weights", default="models/yolo_damage_seg_v3.pt",
                    help="enables method=ensemble")
    ap.add_argument("--patchcore-model", default=None,
                    help="if set, runs the OOD gate (flag unfamiliar frames for review)")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--min-hits", type=int, default=3, help="frames a defect must persist to confirm")
    ap.add_argument("--max-age", type=int, default=5)
    ap.add_argument("--ood-every", type=int, default=15, help="run the OOD gate every N frames")
    ap.add_argument("--every-n", type=int, default=1, help="process 1 of every N frames")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--display", action="store_true", help="show a live window (press q to quit)")
    ap.add_argument("--out", default=None, help="dir for events.jsonl + annotated frames")
    args = ap.parse_args()

    def _exists(p):
        return p if p and (Path(p).exists() or Path(str(p) + ".npz").exists()) else None

    insp = NetInspector(classical_cfg=ClassicalConfig(),
                        patchcore_model_path=_exists(args.patchcore_model),
                        yolo_weights=_exists(args.yolo_weights),
                        seg_weights=_exists(args.seg_weights))
    if args.method not in insp.available_methods():
        print(f"Method '{args.method}' unavailable ({insp.available_methods()}). "
              "Check the weights paths."); return

    # OOD gate: reuse the PatchCore model's own calibrated threshold.
    gate = None
    if _exists(args.patchcore_model):
        from netinspect.ood_gate import OODGate
        from netinspect.patchcore import PatchCoreModel
        pc = PatchCoreModel.load(args.patchcore_model)
        gate = OODGate(threshold=pc.threshold)

    tracker = Tracker(TemporalConfig(min_hits=args.min_hits, max_age=args.max_age))
    out_dir = ensure_dir(args.out) if args.out else None
    events = open(out_dir / "events.jsonl", "w", encoding="utf-8") if out_dir else None

    cv2, cap = _open(args.source)
    from netinspect.visualize import overlay_boxes
    LOGGER.info("Live inspection: source=%s method=%s (q to quit)", args.source, args.method)
    seen_confirmed: set[int] = set()
    idx = processed = 0
    ood_flag = False
    t0 = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                LOGGER.info("Stream ended / dropped."); break
            idx += 1
            if (idx - 1) % args.every_n:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = insp.predict(rgb, method=args.method, conf=args.conf)
            confirmed = tracker.update(res.boxes)

            # OOD gate (throttled): flag frames unlike the training net.
            if gate is not None and processed % args.ood_every == 0:
                score = OODGate.frame_score(rgb, pc)  # type: ignore[name-defined]
                ood_flag = gate.flag(score)

            # Emit one alert per NEWLY confirmed defect track.
            for tid, box in tracker.confirmed_tracks():
                if tid not in seen_confirmed:
                    seen_confirmed.add(tid)
                    evt = {"event": "damage_confirmed", "track": tid, "frame": idx,
                           "score": round(box.score, 3),
                           "bbox": [int(box.x1), int(box.y1), int(box.x2), int(box.y2)],
                           "ood_review": ood_flag, "ts": round(time.time(), 2)}
                    LOGGER.info("ALERT %s", evt)
                    if events:
                        events.write(json.dumps(evt) + "\n"); events.flush()

            processed += 1
            if out_dir or args.display:
                vis = overlay_boxes(rgb, preds=confirmed or res.boxes)[..., ::-1].copy()
                tag = f"{args.method} | {len(confirmed)} confirmed | "
                tag += "OOD: review" if ood_flag else "in-distribution"
                cv2.putText(vis, tag, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (40, 120, 255) if ood_flag else (80, 220, 120), 2, cv2.LINE_AA)
                if out_dir:
                    cv2.imwrite(str(out_dir / f"frame_{idx:06d}.jpg"), vis)
                if args.display:
                    cv2.imshow("net-inspection (live)", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            if args.max_frames and processed >= args.max_frames:
                break
    finally:
        cap.release()
        if args.display:
            cv2.destroyAllWindows()
        if events:
            events.close()

    fps = processed / max(1e-6, time.perf_counter() - t0)
    print(f"\nProcessed {processed} frames at ~{fps:.1f} FPS; "
          f"{len(seen_confirmed)} confirmed defect alert(s).")
    if out_dir:
        print(f"Events + annotated frames in {out_dir}")
    print("Reminder: synthetic-trained models -> human-review triage; "
          "fine-tune on real labels before alerting.")


if __name__ == "__main__":
    main()
