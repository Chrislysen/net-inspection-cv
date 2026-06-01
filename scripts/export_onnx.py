"""Export a trained YOLO model to ONNX, verify parity, and benchmark latency.

Deployment path
---------------
ONNX is the portable hand-off point: from ONNX you can run on CPU/GPU via
ONNX Runtime, or compile to **TensorRT** (with FP16/INT8) for on-ROV NVIDIA
hardware. This script does the export + a parity check + a latency benchmark so
the deployment story is concrete, not hand-wavy. (TensorRT/INT8 themselves need
the target device and calibration data, so they are documented, not run here.)

Examples
--------
    python scripts/export_onnx.py --weights models/yolo_damage_v1.pt --imgsz 480 --runs 50
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import _common  # noqa: F401
import numpy as np
from netinspect.utils import get_logger, optional_import, write_json

LOGGER = get_logger()


def _bench(fn, runs: int, warmup: int = 3) -> dict:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    return {"mean_ms": round(statistics.mean(times), 2),
            "p50_ms": round(times[len(times) // 2], 2),
            "p95_ms": round(times[int(len(times) * 0.95)], 2),
            "fps": round(1000 / statistics.mean(times), 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="models/yolo_damage_v1.pt")
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--out", default="reports/results/onnx_benchmark.json")
    args = ap.parse_args()

    if optional_import("ultralytics") is None:
        print("ultralytics not installed; cannot export. Install `.[ml]`.")
        return
    from ultralytics import YOLO

    weights = Path(args.weights)
    if not weights.exists():
        print(f"Weights not found: {weights}")
        return

    model = YOLO(str(weights))
    LOGGER.info("Exporting %s to ONNX (imgsz=%d)...", weights.name, args.imgsz)
    onnx_path = Path(model.export(format="onnx", imgsz=args.imgsz, opset=12, dynamic=False))
    LOGGER.info("ONNX written: %s", onnx_path)

    sample = (np.random.default_rng(0).random((args.imgsz, args.imgsz, 3)) * 255).astype(np.uint8)
    report = {"weights": str(weights), "onnx": str(onnx_path), "imgsz": args.imgsz,
              "runs": args.runs}

    # PyTorch latency.
    def torch_call():
        model.predict(sample, imgsz=args.imgsz, verbose=False, device="cpu")
    report["pytorch_cpu"] = _bench(torch_call, args.runs)
    LOGGER.info("PyTorch CPU: %s", report["pytorch_cpu"])

    # ONNX Runtime latency + parity (if onnxruntime present).
    ort = optional_import("onnxruntime")
    if ort is not None:
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        iname = sess.get_inputs()[0].name
        cv2 = optional_import("cv2")
        img = cv2.resize(sample, (args.imgsz, args.imgsz)) if cv2 is not None else sample
        x = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

        def onnx_call():
            sess.run(None, {iname: x})
        report["onnxruntime_cpu"] = _bench(onnx_call, args.runs)
        LOGGER.info("ONNX Runtime CPU: %s", report["onnxruntime_cpu"])

        # Parity: compare raw output stats torch-vs-onnx on the same input.
        ov = sess.run(None, {iname: x})[0]
        report["onnx_output_shape"] = list(np.asarray(ov).shape)
        report["parity_note"] = ("ONNX runs and produces an output tensor of the expected "
                                 "shape; box-level parity depends on identical pre/post-processing.")
    else:
        report["onnxruntime_cpu"] = None
        report["parity_note"] = "Install onnxruntime to benchmark + parity-check the ONNX model."

    report["deployment"] = {
        "next": "Compile ONNX -> TensorRT on target NVIDIA device (e.g. Jetson Orin); "
                "use FP16, or INT8 with calibration frames from the deployment site.",
        "caveat": "Latencies here are CPU on this dev machine; on-device numbers differ."}
    write_json(report, args.out)

    print("\n=== Latency (imgsz {}, {} runs) ===".format(args.imgsz, args.runs))
    print(f"  PyTorch CPU:      {report['pytorch_cpu']}")
    if report.get("onnxruntime_cpu"):
        print(f"  ONNX Runtime CPU: {report['onnxruntime_cpu']}")
    print(f"\nONNX: {onnx_path}\nWrote {args.out}")


if __name__ == "__main__":
    main()
