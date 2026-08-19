# Deployment runbook

How the prototype is run and served, and the concrete path toward an operational
deployment. **Caveat:** the detector is not validated on real damage — this
describes the *engineering* deployment, not authorisation to rely on its output.

## 1. Run locally
```bash
pip install -e ".[cv,ml,serve,export]"
python scripts/serve.py            # console + API at http://127.0.0.1:8000
```
Health/readiness/metrics: `GET /api/health`, `/api/ready`, `/api/metrics`.

## 2. Container
```bash
docker build -t netinspect .
docker run -p 8000:8000 -e NETINSPECT_API_KEY="$(openssl rand -hex 24)" netinspect
```
**The key is not optional.** `CMD` binds `0.0.0.0` — the only useful bind inside a
container — and the service refuses a non-loopback bind without authentication,
so without it the container exits at startup rather than coming up open.

For a build with no AGPL in it at all:
```bash
docker build --build-arg EXTRAS=cv,permissive,serve -t netinspect-permissive .
docker run --rm netinspect-permissive netinspect sbom --fail-on copyleft   # exit 0
```
Note that variant ships without ultralytics, so point it at a permissive
checkpoint (`--permissive-weights`) rather than the YOLO weights, or `/api/ready`
will correctly report `yolo` as configured-but-unavailable.

The image installs the package + committed models and runs the FastAPI service
(`scripts/serve.py`). Health checks should target `/api/health` for **liveness**
and `/api/ready` for **readiness** — the latter returns 503, naming the methods,
when anything the deployment was configured for failed to resolve. Both stay open
without a key so probes keep working.

## 3. Torch-free / edge inference
The deployable inference path needs **no PyTorch**:
```bash
python scripts/export_onnx.py --weights models/yolo_damage_v1.pt --imgsz 480   # once
python scripts/stream_inspect.py --onnx models/yolo_damage_v1.onnx --source <video|bag|dir> --out outputs/stream
```
`onnx_infer.OnnxDetector` runs with only `onnxruntime + numpy + cv2`, verified to
match the torch path. **On-device acceleration (next step, needs the hardware):**
compile the ONNX to **TensorRT** on the target (e.g. Jetson Orin) with **FP16** or
**INT8** (INT8 needs ~100–500 calibration frames from the deployment site). Measure
on-device latency there — the CPU numbers in this repo are dev-box only.

## 4. Streaming / operational integration
`stream_inspect.py` is the operational shape: it consumes a stream, applies
**temporal confirmation** (a defect must persist ≥`min_hits` frames), and emits
JSONL **events** — one `damage_confirmed` alert per *new* track, plus `heartbeat`
records with throughput. Wire the event sink to the operator's inspection workflow /
Vision platform (offline review first, alerting later), not as a standalone tool.

## 5. SLOs (illustrative — set real targets with stakeholders)
- **Latency:** ≥95% of frames processed < *T* ms on the target device (measure on
  TensorRT, not CPU). The CPU dev box does ~7 fps (~110 ms/frame).
- **Throughput:** sustain the camera/ROV frame rate after `--every-n` sampling.
- **False-alarm budget:** < *X* confirmed false alerts per hour of *undamaged* net
  (tune `min_hits` + confidence from the FROC curve at the agreed operating point).
- **Recall floor:** an agreed minimum on a labelled validation set (requires real data).

## 6. Monitoring & failover
- **Monitor:** `/api/metrics` (latency, inference counts, errors, uptime); log
  per-request id + latency; sample flagged frames for human audit; watch for
  **drift** (sudden spike/collapse in alert rate ⇒ lighting/condition change).
- **Failover:** the service degrades gracefully — if a heavy model is absent the
  facade still serves the classical method (`/api/ready` reflects this). On a
  device, fall back to the **CPU ONNX** path if GPU/TensorRT fails; restart on OOM.
- **Human-in-the-loop is mandatory:** the model assists; a person confirms any
  flagged defect before action.

## 7. What is explicitly NOT production-ready
**Authentication is implemented** and is mandatory for any non-loopback bind —
`NETINSPECT_API_KEY`, enforced on every `/api` and `/predict` route except the two
probe endpoints, with the service refusing to start otherwise. It is listed here
no longer.

Still genuinely missing: **rate limiting and TLS** (both delegated to the reverse
proxy in `deploy/docker-compose.yml`), **multi-worker scaling** (metrics are
in-process counters and averages, so they neither aggregate across workers nor
give percentiles), a real metrics backend and dashboards, automated
retraining/registry, and — above all — **validated real-damage performance**.
See `PRODUCTION_READINESS.md`.
