"""FastAPI inference service + interactive web console for net inspection.

Production-shaped serving layer:
* path-traversal-safe frame access; upload size/type validation;
* structured request logging with a request id and latency;
* liveness/readiness/metrics endpoints (`/api/health`, `/api/ready`, `/api/metrics`);
* a global exception handler that never leaks stack traces to clients.

Endpoints
---------
* ``GET  /``                  — the web console (static SPA)
* ``GET  /api/health``        — liveness + methods + sources
* ``GET  /api/ready``         — readiness (models resolvable)
* ``GET  /api/metrics``       — Prometheus-style text metrics
* ``GET  /api/frames``        — list frames in a source directory
* ``GET  /api/image``         — raw frame bytes (basename only)
* ``GET  /api/infer``         — run a method on a server-side frame -> JSON
* ``POST /predict``           — multipart image upload -> JSON detections
* ``POST /predict/overlay``   — multipart image upload -> overlay PNG
* ``POST /api/analyze``       — drop an image -> overlay + detections + OOD, one call
* ``POST /api/live/start``    — open a camera / RTSP / video source and start inferring
* ``GET  /api/live/stream``   — annotated frames as multipart MJPEG
* ``GET  /api/live/status``   — fps, latency, dropped frames, confirmed events
* ``POST /api/live/stop``     — release the source

Run
---
    python scripts/serve.py            # auto-loads committed models in models/
    # open http://127.0.0.1:8000

Then drag an image onto the console, or open the Live tab and enter ``0`` for a
webcam, an ``rtsp://`` URL for an ROV feed, or a path to a video file.

Security
--------
Enforced by :mod:`netinspect.security`, configured entirely from the environment
so no secret ends up in a config file:

* ``NETINSPECT_API_KEY`` — required on every ``/api`` and ``/predict`` route
  except ``/api/health`` and ``/api/ready``, which stay open for load balancers.
  Sent as ``X-API-Key``, ``Authorization: Bearer``, or ``?key=`` (the last lands
  in access logs; it exists so the console can be opened from a link).
* **Without a key the service refuses to bind anything but loopback.** Local
  development still just works; publishing an unauthenticated inference endpoint
  to a network is not something you can do by forgetting.
* ``POST /api/live/start`` is **default-deny**. Camera indices are allowed, plus
  files under ``NETINSPECT_MEDIA_ROOT`` (defaulting to the repo's ``data/``) and
  URLs matching ``NETINSPECT_LIVE_ALLOW``. Private, loopback and link-local
  addresses are refused even when a pattern matches, because ``169.254.169.254``
  is how a stream opener becomes cloud-credential theft.
* Uploads are capped by bytes *and* by pixel count, so a small PNG cannot
  decompress into a gigabyte of RGB.
* Concurrent inferences are bounded; excess requests queue and then 503 rather
  than exhausting memory.
* CORS is same-origin unless ``NETINSPECT_CORS_ORIGINS`` says otherwise.

Still single-process: run it behind a reverse proxy for TLS and rate limiting.

Predictions come from models trained on **synthetic** damage and require human
review; recall on real damage is unmeasured.

Note: no ``from __future__ import annotations`` — FastAPI resolves annotations
at runtime.
"""
import argparse
import base64
import io
import sys
import threading
import time
import uuid
from collections import defaultdict
from pathlib import Path

import _common  # noqa: F401
import numpy as np

from netinspect import __version__
from netinspect.classical_baseline import ClassicalConfig
from netinspect.inference import NetInspector
from netinspect.utils import get_logger, list_images, read_image
from netinspect.visualize import overlay_boxes

LOGGER = get_logger()
REPO = _common.REPO_ROOT
WEB_DIR = REPO / "web"

MAX_UPLOAD_BYTES = 16 * 1024 * 1024          # reject oversized uploads
ALLOWED_UPLOAD_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/webp", "image/tiff"}

FRAME_SOURCES = {
    "SOLAQUA bag1 (real, undamaged)": "data/processed/solaqua_frames",
    "SOLAQUA bag2 (real, undamaged)": "data/processed/solaqua_bag2",
    "SOLAQUA different-day (real)": "data/processed/solaqua_diffday",
    "Composited damage on real net": "data/processed/real_composite/images/test",
    "Contiguous sequence (video)": "data/processed/solaqua_seq",
    "Synthetic demo": "data/sample/images",
}

# One-line "what am I looking at" per source — surfaced in the console.
SOURCE_INFO = {
    "SOLAQUA bag1 (real, undamaged)": "Real ROV footage the models trained on. Undamaged — a detector should fire ~never.",
    "SOLAQUA bag2 (real, undamaged)": "Real, same site, a different clip. Undamaged — tests for false alarms.",
    "SOLAQUA different-day (real)": "Real, a DIFFERENT day (held-out). Undamaged — the honest out-of-distribution test.",
    "Composited damage on real net": "Synthetic damage pasted on real net (labelled). The only frames here that contain 'damage'.",
    "Contiguous sequence (video)": "A continuous clip — use temporal confirmation to drop flicker false alarms.",
    "Synthetic demo": "Procedural placeholder data. Verifies the pipeline only, not real-world skill.",
}


class _Metrics:
    """Tiny in-process metrics store (single-process prototype)."""
    def __init__(self):
        self.requests = defaultdict(int)        # path -> count
        self.errors = 0
        self.inferences = defaultdict(int)       # method -> count
        self.latency_sum_ms = 0.0
        self.latency_count = 0
        self.started = time.time()

    def observe(self, path: str, ms: float):
        self.requests[path] += 1
        self.latency_sum_ms += ms
        self.latency_count += 1

    def prometheus(self) -> str:
        lines = [
            "# HELP netinspect_requests_total HTTP requests by path",
            "# TYPE netinspect_requests_total counter",
        ]
        for p, n in sorted(self.requests.items()):
            lines.append(f'netinspect_requests_total{{path="{p}"}} {n}')
        lines += ["# HELP netinspect_inferences_total Inferences by method",
                  "# TYPE netinspect_inferences_total counter"]
        for m, n in sorted(self.inferences.items()):
            lines.append(f'netinspect_inferences_total{{method="{m}"}} {n}')
        avg = self.latency_sum_ms / self.latency_count if self.latency_count else 0.0
        lines += [
            "# HELP netinspect_request_latency_ms_avg Average request latency",
            "# TYPE netinspect_request_latency_ms_avg gauge",
            f"netinspect_request_latency_ms_avg {avg:.2f}",
            "# HELP netinspect_errors_total Unhandled errors",
            "# TYPE netinspect_errors_total counter",
            f"netinspect_errors_total {self.errors}",
            "# HELP netinspect_uptime_seconds Service uptime",
            "# TYPE netinspect_uptime_seconds gauge",
            f"netinspect_uptime_seconds {time.time() - self.started:.0f}",
        ]
        return "\n".join(lines) + "\n"


def _available_sources():
    return {name: rel for name, rel in FRAME_SOURCES.items() if list_images(REPO / rel)}


def _ood_method(inspector):
    """Cheapest available out-of-distribution signal: anomaly, else patchcore."""
    avail = inspector.available_methods()
    for m in ("anomaly", "patchcore"):
        if m in avail:
            return m
    return None


def _ood_status(inspector, img):
    """Run the OOD gate: is this frame unlike the training net? -> human review."""
    m = _ood_method(inspector)
    if m is None:
        return None
    r = inspector.predict(img, method=m)
    ms, th = r.meta.get("max_score"), r.meta.get("threshold")
    if ms is None or not th:
        return None
    return {"flagged": bool(ms >= th), "score": round(float(ms), 3),
            "threshold": round(float(th), 3), "via": m}


def _png_b64(image: np.ndarray) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_app(inspector: NetInspector, security=None):
    from contextlib import asynccontextmanager

    from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
    from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
    from fastapi.staticfiles import StaticFiles

    # A live session owns a camera handle and two threads, so it must be
    # released on shutdown — otherwise a reload leaves the device locked.
    live_state: dict = {"session": None}

    def _stop_live():
        session = live_state.get("session")
        if session is not None:
            try:
                session.stop()
            finally:
                live_state["session"] = None

    @asynccontextmanager
    async def lifespan(_app):
        yield
        _stop_live()

    from netinspect.security import (
        SecurityConfig,
        SourceRejected,
        apply_decoder_limits,
        check_api_key,
        check_image_size,
        validate_live_source,
    )

    sec = security if security is not None else SecurityConfig.from_env()
    if sec.media_root is None:
        # Default to the repo's own data directory rather than the whole
        # filesystem: the bundled clips keep working out of the box, and
        # "C:/Windows/win.ini" still does not. Override with NETINSPECT_MEDIA_ROOT.
        sec.media_root = (REPO / "data").resolve()
    apply_decoder_limits(sec)

    app = FastAPI(title="net-inspection-cv console", version=__version__,
                  lifespan=lifespan)
    sources = _available_sources()
    metrics = _Metrics()

    # Same-origin by default. A wildcard here plus an unauthenticated API is how
    # any page on the internet gets to drive your inference service.
    if sec.cors_origins:
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(CORSMiddleware, allow_origins=list(sec.cors_origins),
                           allow_credentials=True, allow_methods=["GET", "POST"],
                           allow_headers=["*"])

    # Inference is CPU- and memory-hungry, and nothing else bounds how many run
    # at once. Without this a handful of concurrent requests can page a box to
    # death; with it, they queue.
    inference_slots = threading.BoundedSemaphore(sec.max_concurrency)

    async def _predict_async(img, **kw):
        """Run inference off the event loop.

        The upload routes are `async def`, so anything blocking inside them
        blocks the whole server — not just that request. Model inference takes
        ~100 ms to 1.3 s here and the concurrency semaphore waits up to 30 s, so
        one upload could freeze every other connection, including health checks.
        """
        from starlette.concurrency import run_in_threadpool
        return await run_in_threadpool(lambda: _predict(img, **kw))

    def _predict(img, **kw):
        """Every inference goes through here, so the concurrency cap cannot be
        bypassed by adding a route that forgets it."""
        if not inference_slots.acquire(timeout=30):
            raise HTTPException(503, "Server busy — too many concurrent inferences. "
                                     "Retry shortly.")
        try:
            return inspector.predict(img, **kw)
        finally:
            inference_slots.release()

    @app.middleware("http")
    async def _authenticate(request: Request, call_next):
        path = request.url.path
        # Liveness must stay reachable for a load balancer, and the static
        # console has to load before it can send a key.
        open_paths = path in ("/api/health", "/api/ready") or not path.startswith(("/api", "/predict"))
        if sec.auth_enabled and not open_paths:
            supplied = (request.headers.get("x-api-key")
                        or (request.headers.get("authorization") or "").removeprefix("Bearer ").strip()
                        # Query-param keys land in access logs and browser
                        # history; supported only so the console can be opened
                        # from a link, and documented as the weaker option.
                        or request.query_params.get("key"))
            if not check_api_key(supplied, sec):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)

    @app.middleware("http")
    async def _observability(request: Request, call_next):
        rid = uuid.uuid4().hex[:8]
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:  # pragma: no cover - defensive
            metrics.errors += 1
            LOGGER.exception("rid=%s unhandled error on %s", rid, request.url.path)
            return JSONResponse({"error": "internal error", "request_id": rid}, status_code=500)
        ms = (time.perf_counter() - t0) * 1000
        if request.url.path.startswith("/api") or request.url.path.startswith("/predict"):
            metrics.observe(request.url.path, ms)
            LOGGER.info("rid=%s %s %s -> %d %.0fms", rid, request.method,
                        request.url.path, response.status_code, ms)
        response.headers["X-Request-ID"] = rid
        return response

    async def _read_upload(file: UploadFile) -> np.ndarray:
        if file.content_type and file.content_type not in ALLOWED_UPLOAD_TYPES:
            raise HTTPException(415, f"Unsupported content type: {file.content_type}")
        data = await file.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"Upload exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MB")
        try:
            from PIL import Image
            im = Image.open(io.BytesIO(data))
            # Checked before decoding: a 16k-square PNG is a small file that
            # expands to about a gigabyte of RGB.
            check_image_size(im.width, im.height, sec)
            return np.asarray(im.convert("RGB"))
        except ValueError as exc:
            raise HTTPException(413, str(exc))
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(400, "Could not decode image")

    def _resolve(source: str, name: str) -> Path:
        rel = sources.get(source)
        if rel is None:
            raise HTTPException(404, f"Unknown source: {source}")
        # Path-traversal safe: keep only the basename, then confirm containment.
        safe_name = Path(name).name
        base = (REPO / rel).resolve()
        p = (base / safe_name).resolve()
        if base != p.parent or not p.exists():
            raise HTTPException(404, f"Frame not found: {safe_name}")
        return p

    def _validate_method(method: str):
        if method not in inspector.available_methods():
            raise HTTPException(400, f"Method '{method}' unavailable. "
                                     f"Available: {inspector.available_methods()}")

    @app.get("/api/health")
    def health():
        return {"status": "ok", "methods": inspector.available_methods(),
                "sources": list(sources.keys()),
                "source_info": {k: SOURCE_INFO.get(k, "") for k in sources},
                "ood_gate": _ood_method(inspector) is not None,
                "version": app.version}

    @app.get("/api/ready")
    def ready():
        # Ready if the always-on classical method is usable; report model availability.
        methods = inspector.available_methods()
        return JSONResponse(
            {"ready": "classical" in methods, "methods": methods},
            status_code=200 if "classical" in methods else 503)

    @app.get("/api/version")
    def version():
        """Exactly what this deployment is running.

        The first question during an incident is "which build is that", and a
        version string alone does not answer it — the weights matter more than
        the code. Model digests are included so a report can be tied to the
        artefacts that produced it.
        """
        import hashlib
        import subprocess

        def _commit():
            try:
                return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                      cwd=REPO, capture_output=True, text=True,
                                      timeout=2).stdout.strip() or None
            except Exception:
                return None

        models = {}
        model_dir = REPO / "models"
        if model_dir.exists():
            for f in sorted(model_dir.glob("*.pt")):
                h = hashlib.sha256()
                with open(f, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
                models[f.name] = {"sha256": h.hexdigest()[:16],
                                  "bytes": f.stat().st_size}
        return {"service": app.version, "commit": _commit(),
                "python": sys.version.split()[0],
                "methods": inspector.available_methods(),
                "auth": sec.auth_enabled,
                "models": models}

    @app.get("/api/metrics", response_class=PlainTextResponse)
    def metrics_endpoint():
        return metrics.prometheus()

    @app.get("/api/frames")
    def frames(source: str = Query(...)):
        rel = sources.get(source)
        if rel is None:
            raise HTTPException(404, f"Unknown source: {source}")
        return {"source": source, "frames": [p.name for p in list_images(REPO / rel)]}

    @app.get("/api/image")
    def image(source: str = Query(...), name: str = Query(...)):
        return FileResponse(_resolve(source, name))

    @app.get("/api/infer")
    def infer(source: str = Query(...), name: str = Query(...),
              method: str = Query("yolo"), conf: float = Query(0.25, ge=0.0, le=1.0),
              ood: bool = Query(False)):
        _validate_method(method)
        img = read_image(_resolve(source, name))
        result = _predict(img, method=method, conf=conf)
        metrics.inferences[method] += 1
        vis = result.heatmap if result.heatmap is not None else overlay_boxes(img, preds=result.boxes)
        return JSONResponse({
            "method": method, "frame": Path(name).name, "conf": conf,
            "latency_ms": round(result.elapsed_ms, 1), "count": len(result.boxes),
            "ood": _ood_status(inspector, img) if ood else None,
            "image_size": {"width": img.shape[1], "height": img.shape[0]},
            "detections": [
                {"class": b.class_name, "score": round(b.score, 3),
                 "bbox": [int(b.x1), int(b.y1), int(b.x2), int(b.y2)]}
                for b in result.boxes],
            "overlay": _png_b64(vis), "is_heatmap": result.heatmap is not None,
            "disclaimer": "Prototype: proxy-trained model; human review required.",
        })

    @app.post("/predict")
    async def predict(file: UploadFile = File(...), method: str = Query("classical"),
                      conf: float = Query(0.25, ge=0.0, le=1.0)):
        _validate_method(method)
        img = await _read_upload(file)
        r = await _predict_async(img, method=method, conf=conf)
        metrics.inferences[method] += 1
        payload = r.to_dict()
        payload["disclaimer"] = "Prototype: proxy-trained model; human review required."
        return JSONResponse(payload)

    @app.post("/predict/overlay")
    async def predict_overlay(file: UploadFile = File(...), method: str = Query("classical"),
                              conf: float = Query(0.25, ge=0.0, le=1.0)):
        from PIL import Image
        _validate_method(method)
        img = await _read_upload(file)
        r = await _predict_async(img, method=method, conf=conf)
        metrics.inferences[method] += 1
        vis = r.heatmap if r.heatmap is not None else overlay_boxes(img, preds=r.boxes)
        buf = io.BytesIO(); Image.fromarray(vis).save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")

    # ---------------------------------------------------------------- #
    # Drop-to-analyse: one round trip returns overlay, detections and the
    # OOD verdict together, so the UI never has to stitch two responses.
    # ---------------------------------------------------------------- #
    @app.post("/api/analyze")
    async def analyze(file: UploadFile = File(...), method: str = Query("yolo"),
                      conf: float = Query(0.25, ge=0.0, le=1.0),
                      ood: bool = Query(True)):
        _validate_method(method)
        img = await _read_upload(file)
        r = await _predict_async(img, method=method, conf=conf)
        metrics.inferences[method] += 1
        vis = r.heatmap if r.heatmap is not None else overlay_boxes(img, preds=r.boxes)
        return JSONResponse({
            "filename": Path(file.filename or "upload").name,
            "method": method, "conf": conf,
            "latency_ms": round(r.elapsed_ms, 1), "count": len(r.boxes),
            "ood": _ood_status(inspector, img) if ood else None,
            "image_size": {"width": img.shape[1], "height": img.shape[0]},
            "detections": [
                {"class": b.class_name, "score": round(b.score, 3),
                 "bbox": [int(b.x1), int(b.y1), int(b.x2), int(b.y2)]}
                for b in r.boxes],
            "overlay": _png_b64(vis), "is_heatmap": r.heatmap is not None,
            "disclaimer": ("Prototype: model trained on SYNTHETIC damage. "
                           "Human review required; recall on real damage is unmeasured."),
        })

    # ---------------------------------------------------------------- #
    # Live camera / ROV feed
    #
    # One session at a time, deliberately: a second concurrent stream would
    # contend for the same model and make both slower and neither real-time.
    # ---------------------------------------------------------------- #
    @app.post("/api/live/start")
    def live_start(source: str = Query(..., description="0 for webcam, an RTSP/HTTP URL, or a video path"),
                   method: str = Query("yolo"),
                   conf: float = Query(0.25, ge=0.0, le=1.0),
                   min_hits: int = Query(3, ge=1, le=30),
                   ood: bool = Query(False),
                   loop: bool = Query(True, description="Loop video files"),
                   odometry: bool = Query(False, description="Track along-net travel"),
                   standoff_m: float = Query(0.6, gt=0.0, le=10.0,
                                             description="Declared standoff; sets live scale")):
        from netinspect.live import LiveInspector, LiveSession

        _validate_method(method)
        try:
            source = validate_live_source(source, sec)
        except SourceRejected as exc:
            raise HTTPException(403, str(exc))
        _stop_live()

        ood_model = None
        if ood:
            try:
                ood_model = inspector._patchcore()      # noqa: SLF001 - same package
            except Exception as exc:
                LOGGER.warning("OOD gate unavailable for live: %s", exc)

        live = LiveInspector(inspector, method=method, conf=conf,
                             min_hits=min_hits, ood_model=ood_model,
                             odometry=odometry, standoff_m=standoff_m)
        session = LiveSession(source, live, loop_files=loop)
        try:
            session.start()
        except Exception as exc:
            raise HTTPException(400, str(exc))
        live_state["session"] = session
        return JSONResponse(session.status())

    @app.post("/api/live/stop")
    def live_stop():
        running = live_state.get("session") is not None
        _stop_live()
        return {"stopped": running}

    @app.get("/api/live/status")
    def live_status():
        session = live_state.get("session")
        return JSONResponse(session.status() if session else {"running": False})

    @app.get("/api/live/stream")
    def live_stream(fps: float = Query(12.0, gt=0.0, le=60.0)):
        """Annotated frames as multipart MJPEG — an <img> tag renders it directly.

        MJPEG rather than WebRTC or a websocket because it needs no client-side
        decoding, survives a page refresh, and degrades gracefully: if inference
        is slower than the requested rate, the same annotated frame is re-sent
        rather than a backlog being replayed.
        """
        from fastapi.responses import StreamingResponse
        from PIL import Image

        session = live_state.get("session")
        if session is None:
            raise HTTPException(409, "No live session. POST /api/live/start first.")

        boundary = "netinspectframe"
        interval = 1.0 / fps

        def frames():
            while True:
                s = live_state.get("session")
                if s is None or not s.running:
                    break
                frame = s.latest()
                if frame is not None:
                    buf = io.BytesIO()
                    Image.fromarray(frame.image).save(buf, format="JPEG", quality=80)
                    payload = buf.getvalue()
                    yield (f"--{boundary}\r\nContent-Type: image/jpeg\r\n"
                           f"Content-Length: {len(payload)}\r\n\r\n").encode()
                    yield payload
                    yield b"\r\n"
                time.sleep(interval)

        return StreamingResponse(
            frames(),
            media_type=f"multipart/x-mixed-replace; boundary={boundary}",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    # ---------------------------------------------------------------- #
    # Net model: place a mapped pass on a schematic cage
    #
    # The cage is DECLARED (the operator knows their pen; the footage cannot
    # tell us its radius) while the strip and the sites are MEASURED. The
    # response keeps the two apart in a `provenance` block and the UI renders
    # them differently, so the picture cannot quietly imply we reconstructed a
    # net we did not.
    # ---------------------------------------------------------------- #
    MAPS_DIR = REPO / "reports" / "results" / "inspection_maps"

    def _load_map(clip: str) -> dict:
        safe = Path(clip).name
        p = (MAPS_DIR / f"{safe}_map.json").resolve()
        if MAPS_DIR.resolve() != p.parent or not p.exists():
            raise HTTPException(404, f"No inspection map for {safe!r}. "
                                     "Run scripts/map_inspection.py first.")
        import json
        return json.loads(p.read_text(encoding="utf-8"))

    @app.get("/api/maps")
    def maps():
        if not MAPS_DIR.exists():
            return {"maps": []}
        return {"maps": sorted(p.name[: -len("_map.json")]
                               for p in MAPS_DIR.glob("*_map.json"))}

    @app.get("/api/scene")
    def scene(clip: str = Query(...),
              circumference_m: float = Query(160.0, gt=1.0, le=1000.0),
              cylinder_depth_m: float = Query(15.0, ge=0.0, le=200.0),
              cone_depth_m: float = Query(10.0, ge=0.0, le=200.0),
              start_bearing_deg: float = Query(0.0, ge=-360.0, le=360.0),
              barge_bearing_deg: float = Query(0.0, ge=-360.0, le=360.0),
              clockwise: bool = Query(True)):
        from netinspect.netmodel import PenGeometry, build_scene

        data = _load_map(clip)
        try:
            geom = PenGeometry(circumference_m=circumference_m,
                               cylinder_depth_m=cylinder_depth_m,
                               cone_depth_m=cone_depth_m,
                               start_bearing_deg=start_bearing_deg,
                               barge_bearing_deg=barge_bearing_deg,
                               clockwise=clockwise)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

        out = build_scene(sites=data.get("sites", []), track=data.get("track", []),
                          coverage=data.get("coverage", {}), geom=geom)
        out["clip"] = data.get("clip", clip)
        out["method"] = data.get("method")
        out["crops"] = data.get("site_crops", {})
        out["caveats"] = data.get("caveats", [])
        out["telemetry_check"] = data.get("telemetry_check", {})
        return JSONResponse(out)

    @app.get("/api/scene/crop")
    def scene_crop(clip: str = Query(...), site: int = Query(..., ge=1)):
        """The clearest look at one site — what makes a coordinate judgeable."""
        base = (MAPS_DIR / f"{Path(clip).name}_crops").resolve()
        p = (base / f"site_{int(site)}.jpg").resolve()
        if base != p.parent or not p.exists():
            raise HTTPException(404, "No crop for that site")
        return FileResponse(p, media_type="image/jpeg")

    if WEB_DIR.exists():
        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anomaly-model", default="models/anomaly_normal_net")
    ap.add_argument("--patchcore-model", default="models/patchcore_normal_net")
    ap.add_argument("--yolo-weights", default="models/yolo_damage_v1.pt")
    ap.add_argument("--permissive-weights", default="models/permissive_v1.pt",
                    help="torchvision (BSD-3-Clause) detector — the AGPL-free "
                         "path, selectable in the console as 'permissive'")
    ap.add_argument("--seg-weights", default="models/yolo_damage_seg_v3.pt",
                    help="Segmentation model; enables the det+seg 'ensemble' method")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    def _exists(p):
        return p if p and (Path(p).exists() or Path(str(p) + ".npz").exists()) else None

    inspector = NetInspector(
        classical_cfg=ClassicalConfig(),
        anomaly_model_path=_exists(args.anomaly_model),
        patchcore_model_path=_exists(args.patchcore_model),
        yolo_weights=_exists(args.yolo_weights),
        seg_weights=_exists(args.seg_weights),
        permissive_weights=_exists(args.permissive_weights),
    )
    LOGGER.info("Methods: %s", inspector.available_methods())
    LOGGER.info("Console: http://%s:%d", args.host, args.port)
    import uvicorn

    from netinspect.security import InsecureBinding, SecurityConfig, check_binding
    sec = SecurityConfig.from_env()
    try:
        check_binding(args.host, sec)
    except InsecureBinding as exc:
        raise SystemExit(f"\n{exc}\n")
    LOGGER.info("Security: %s", sec.describe())
    uvicorn.run(build_app(inspector, security=sec), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
