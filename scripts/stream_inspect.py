"""Streaming inspection pipeline — the operational shape of a net-inspection feed.

Consumes a *stream* (video file, ROS `.bag`, or a frame directory in order), runs
detection + **temporal confirmation**, and emits structured **events** as JSONL:
one ``damage_confirmed`` alert per *newly* confirmed track (so an operator gets
one alert per real defect, not one per frame), plus periodic ``heartbeat`` events
with throughput. This is how you'd wire the model into an inspection workflow.

It can run the **torch-free ONNX** detector (`--onnx`) — the deployable path — or
any in-process method.

Examples
--------
    python scripts/stream_inspect.py --onnx models/yolo_damage_v1.onnx --source clip.mp4 --out outputs/stream
    python scripts/stream_inspect.py --method classical --source data/processed/solaqua_seq --out outputs/stream
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
from netinspect.utils import (
    VIDEO_EXTENSIONS,
    ensure_dir,
    get_logger,
    list_images,
    optional_import,
    read_image,
    write_json,
)

LOGGER = get_logger()


def _iter_stream(source: Path, every_n: int, max_frames: int | None):
    """Yield (index, name, rgb) from a frame dir, a video, or a ROS bag."""
    if source.is_dir():
        for i, p in enumerate(list_images(source)):
            yield i, p.stem, read_image(p)
        return
    suffix = source.suffix.lower()
    if suffix == ".bag":
        import tempfile

        from netinspect.solaqua import extract_bag_frames
        tmp = Path(tempfile.mkdtemp())
        for i, p in enumerate(extract_bag_frames(source, tmp, every_n=every_n, max_frames=max_frames)):
            yield i, p.stem, read_image(p)
        return
    if suffix in VIDEO_EXTENSIONS:
        cv2 = optional_import("cv2")
        cap = cv2.VideoCapture(str(source))
        idx = saved = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % every_n == 0:
                yield saved, f"{source.stem}_{idx:06d}", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                saved += 1
                if max_frames and saved >= max_frames:
                    break
            idx += 1
        cap.release()
        return
    raise ValueError(f"Unsupported source: {source}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="video | .bag | frame directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--onnx", default=None, help="ONNX weights -> torch-free detector")
    ap.add_argument("--method", default="classical", choices=["classical", "yolo", "patchcore"])
    ap.add_argument("--yolo-weights", default=None)
    ap.add_argument("--patchcore-model", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--every-n", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--min-hits", type=int, default=3, help="frames to confirm a detection")
    ap.add_argument("--heartbeat", type=int, default=50, help="emit a heartbeat every N frames")
    args = ap.parse_args()

    # Detector: torch-free ONNX, or an in-process method via the facade.
    if args.onnx:
        from netinspect.onnx_infer import OnnxDetector
        onnx = OnnxDetector(args.onnx, conf=args.conf)
        def detect(img):
            return onnx.predict(img)
        method_label = "onnx-yolo"
    else:
        ccfg = ClassicalConfig()
        if args.config:
            params = _common.load_yaml(args.config).get("classical", {})
            ccfg = ClassicalConfig(**{k: v for k, v in params.items() if k in ClassicalConfig().__dict__})
        insp = NetInspector(classical_cfg=ccfg, patchcore_model_path=args.patchcore_model,
                            yolo_weights=args.yolo_weights)
        if args.method not in insp.available_methods():
            print(f"Method '{args.method}' unavailable ({insp.available_methods()}).")
            return
        def detect(img):
            return insp.predict(img, method=args.method, conf=args.conf).boxes
        method_label = args.method

    tracker = Tracker(TemporalConfig(min_hits=args.min_hits, conf=args.conf))
    out = ensure_dir(args.out)
    events_path = out / "events.jsonl"
    seen_tracks: set[int] = set()
    n = alerts = raw_total = 0
    latencies = []
    t_start = time.perf_counter()

    with events_path.open("w", encoding="utf-8") as ef:
        for idx, name, img in _iter_stream(Path(args.source), args.every_n, args.max_frames):
            t0 = time.perf_counter()
            raw = detect(img)
            tracker.update(raw)
            dt = (time.perf_counter() - t0) * 1000
            latencies.append(dt)
            raw_total += len(raw)
            n += 1

            for tid, box in tracker.confirmed_tracks():
                if tid not in seen_tracks:        # NEW confirmed damage -> one alert
                    seen_tracks.add(tid)
                    alerts += 1
                    ef.write(json.dumps({
                        "event": "damage_confirmed", "track_id": tid, "frame_index": idx,
                        "frame": name, "score": round(box.score, 3),
                        "bbox": [int(box.x1), int(box.y1), int(box.x2), int(box.y2)],
                        "latency_ms": round(dt, 1),
                    }) + "\n")
            if args.heartbeat and n % args.heartbeat == 0:
                fps = n / (time.perf_counter() - t_start)
                ef.write(json.dumps({"event": "heartbeat", "frames": n,
                                     "fps": round(fps, 1), "alerts": alerts}) + "\n")

    elapsed = time.perf_counter() - t_start
    summary = {
        "source": str(args.source), "detector": method_label, "frames": n,
        "raw_detections": raw_total, "confirmed_alerts": alerts,
        "mean_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "throughput_fps": round(n / elapsed, 2) if elapsed else 0,
        "min_hits": args.min_hits,
    }
    write_json(summary, out / "stream_summary.json")
    print(f"Streamed {n} frames @ {summary['throughput_fps']} fps "
          f"({summary['mean_latency_ms']} ms/frame).")
    print(f"Raw detections: {raw_total} -> {alerts} confirmed alert(s) "
          f"(temporal min_hits={args.min_hits}).")
    print(f"Events: {events_path}")


if __name__ == "__main__":
    main()
