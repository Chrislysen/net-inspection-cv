"""Live net inspection on a camera, RTSP/ROV stream, or video file (headless or windowed).

A thin CLI over :mod:`netinspect.live`, which the browser console
(``scripts/serve.py``) drives too — so what you see here and what you see in the
web UI are produced by the same code. Detections are confirmed over time, so you
get one alert per *persisting* defect rather than one per frame, and unfamiliar
frames can be flagged for human review instead of silently scored.

    # RTSP / IP camera (e.g. an ROV feed), ensemble detector, show a window:
    python scripts/live_inspect.py --source rtsp://CAMERA_IP/stream \\
        --method ensemble --yolo-weights models/yolo_damage_v1.pt \\
        --seg-weights models/yolo_damage_seg_v3.pt --display

    # USB webcam 0, YOLOv8s-seg, with the OOD gate + an event log:
    python scripts/live_inspect.py --source 0 --method yolo \\
        --yolo-weights models/yolo_damage_seg_gpu.pt \\
        --patchcore-model models/patchcore_normal_net --out outputs/live

    # A recorded clip, looping, as a stand-in camera:
    python scripts/live_inspect.py --source clip.mp4 --loop --display

Prefer the browser? ``python scripts/serve.py`` then open the Live tab.

Honesty: the shipped models learned *synthetic* damage, so on a new camera or
site treat detections as **human-review triage**, not verified alarms — the OOD
gate will usually flag a new domain, because it is one. Run in shadow mode and
fine-tune on real labels before anything alerts.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import _common  # noqa: F401

from netinspect.classical_baseline import ClassicalConfig
from netinspect.inference import NetInspector
from netinspect.live import LiveInspector, LiveSource, describe_source
from netinspect.utils import ensure_dir, get_logger

LOGGER = get_logger()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True,
                    help="RTSP/HTTP URL | USB index (0,1,..) | video file")
    ap.add_argument("--method", default="yolo",
                    choices=["classical", "anomaly", "patchcore", "yolo", "ensemble"])
    ap.add_argument("--yolo-weights", default="models/yolo_damage_v1.pt")
    ap.add_argument("--seg-weights", default="models/yolo_damage_seg_v3.pt",
                    help="enables method=ensemble")
    ap.add_argument("--patchcore-model", default=None,
                    help="if set, runs the OOD gate (flag unfamiliar frames for review)")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--min-hits", type=int, default=3,
                    help="frames a defect must persist before it is confirmed")
    ap.add_argument("--max-age", type=int, default=5)
    ap.add_argument("--ood-every", type=int, default=15,
                    help="run the OOD gate every N processed frames")
    ap.add_argument("--every-n", type=int, default=1, help="process 1 of every N frames")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--loop", action="store_true", help="loop a video file forever")
    ap.add_argument("--display", action="store_true",
                    help="show a live window (press q to quit)")
    ap.add_argument("--out", default=None, help="dir for events.jsonl + annotated frames")
    args = ap.parse_args()

    def _exists(p):
        return p if p and (Path(p).exists() or Path(str(p) + ".npz").exists()) else None

    insp = NetInspector(classical_cfg=ClassicalConfig(),
                        patchcore_model_path=_exists(args.patchcore_model),
                        yolo_weights=_exists(args.yolo_weights),
                        seg_weights=_exists(args.seg_weights))
    if args.method not in insp.available_methods():
        raise SystemExit(f"Method {args.method!r} unavailable "
                         f"({insp.available_methods()}). Check the weights paths.")

    ood_model = None
    if _exists(args.patchcore_model):
        from netinspect.patchcore import PatchCoreModel
        ood_model = PatchCoreModel.load(args.patchcore_model)

    live = LiveInspector(insp, method=args.method, conf=args.conf,
                         min_hits=args.min_hits, max_age=args.max_age,
                         ood_model=ood_model, ood_every=args.ood_every,
                         draw=bool(args.out or args.display))

    out_dir = ensure_dir(args.out) if args.out else None
    events = open(out_dir / "events.jsonl", "w", encoding="utf-8") if out_dir else None

    cv2 = None
    if args.display or out_dir:
        from netinspect.utils import require
        cv2 = require("cv2", hint="pip install -e '.[cv]'")

    LOGGER.info("Live inspection: %s (%s), method=%s%s",
                args.source, describe_source(args.source), args.method,
                " — press q to quit" if args.display else "")

    seen: set[int] = set()
    processed = 0
    t0 = time.perf_counter()
    try:
        with LiveSource(args.source, loop_files=args.loop) as source:
            for frame in live.run(source, max_frames=args.max_frames,
                                  every_n=args.every_n):
                processed += 1

                # One alert per newly confirmed track, not per frame.
                for tid, box in live.tracker.confirmed_tracks():
                    if tid in seen:
                        continue
                    seen.add(tid)
                    evt = {"event": "damage_confirmed", "track": tid,
                           "frame": frame.index, "score": round(box.score, 3),
                           "bbox": [int(box.x1), int(box.y1), int(box.x2), int(box.y2)],
                           "ood_review": bool(frame.ood), "ts": round(frame.timestamp, 2)}
                    LOGGER.info("ALERT %s", evt)
                    if events:
                        events.write(json.dumps(evt) + "\n")
                        events.flush()

                if cv2 is not None:
                    vis = frame.image[..., ::-1].copy()          # RGB -> BGR
                    tag = (f"{frame.method} | {len(frame.confirmed)} confirmed | "
                           + ("OOD: review" if frame.ood else "in-distribution"))
                    cv2.putText(vis, tag, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (40, 120, 255) if frame.ood else (80, 220, 120),
                                2, cv2.LINE_AA)
                    if out_dir:
                        cv2.imwrite(str(out_dir / f"frame_{frame.index:06d}.jpg"), vis)
                    if args.display:
                        cv2.imshow("net-inspection (live)", vis)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
    except KeyboardInterrupt:
        LOGGER.info("Interrupted.")
    finally:
        if args.display and cv2 is not None:
            cv2.destroyAllWindows()
        if events:
            events.close()

    fps = processed / max(1e-6, time.perf_counter() - t0)
    print(f"\nProcessed {processed} frames at ~{fps:.1f} FPS; "
          f"{len(seen)} confirmed defect alert(s).")
    if out_dir:
        print(f"Events + annotated frames in {out_dir}")
    print("Reminder: synthetic-trained models -> human-review triage; "
          "fine-tune on real labels before alerting.")


if __name__ == "__main__":
    main()
