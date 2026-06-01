"""FastAPI inference service + interactive web console for net inspection.

Endpoints
---------
* ``GET  /``                  — the web console (static SPA)
* ``GET  /api/health``        — methods + frame sources available
* ``GET  /api/frames``        — list frames in a source directory
* ``GET  /api/image``         — raw frame bytes
* ``GET  /api/infer``         — run a method on a server-side frame -> JSON
                                (detections + base64 overlay + latency)
* ``POST /predict``           — multipart upload -> JSON detections
* ``POST /predict/overlay``   — multipart upload -> overlay PNG

Run
---
    python scripts/serve.py            # auto-loads committed models in models/
    # open http://127.0.0.1:8000

Prototype serving layer: single process, no auth. Predictions are from
synthetic/proxy-trained models and require human review.

Note: no ``from __future__ import annotations`` — FastAPI resolves annotations
at runtime.
"""
import argparse
import base64
import io
from pathlib import Path

import _common  # noqa: F401
import numpy as np

from netinspect.classical_baseline import ClassicalConfig
from netinspect.inference import NetInspector
from netinspect.utils import get_logger, list_images, read_image
from netinspect.visualize import overlay_boxes

LOGGER = get_logger()
REPO = _common.REPO_ROOT
WEB_DIR = REPO / "web"

# Candidate demo frame sources (only those that exist are exposed).
FRAME_SOURCES = {
    "SOLAQUA bag1 (real, undamaged)": "data/processed/solaqua_frames",
    "SOLAQUA bag2 (real, undamaged)": "data/processed/solaqua_bag2",
    "SOLAQUA different-day (real)": "data/processed/solaqua_diffday",
    "Composited damage on real net": "data/processed/real_composite/images/test",
    "Contiguous sequence (video)": "data/processed/solaqua_seq",
    "Synthetic demo": "data/sample/images",
}


def _available_sources():
    return {name: rel for name, rel in FRAME_SOURCES.items()
            if list_images(REPO / rel)}


def _png_b64(image: np.ndarray) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_app(inspector: NetInspector):
    from fastapi import FastAPI, File, HTTPException, Query, UploadFile
    from fastapi.responses import FileResponse, JSONResponse, Response
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="net-inspection-cv console", version="0.2.0")
    sources = _available_sources()

    def _read_upload(data: bytes) -> np.ndarray:
        from PIL import Image
        return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))

    def _resolve(source: str, name: str) -> Path:
        rel = sources.get(source)
        if rel is None:
            raise HTTPException(404, f"Unknown source: {source}")
        p = REPO / rel / name
        if not p.exists():
            raise HTTPException(404, f"Frame not found: {name}")
        return p

    @app.get("/api/health")
    def health():
        return {"status": "ok", "methods": inspector.available_methods(),
                "sources": list(sources.keys())}

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
              method: str = Query("yolo"), conf: float = Query(0.25)):
        if method not in inspector.available_methods():
            raise HTTPException(400, f"Method '{method}' unavailable.")
        img = read_image(_resolve(source, name))
        result = inspector.predict(img, method=method, conf=conf)
        vis = result.heatmap if result.heatmap is not None else overlay_boxes(img, preds=result.boxes)
        return JSONResponse({
            "method": method, "frame": name, "conf": conf,
            "latency_ms": round(result.elapsed_ms, 1),
            "count": len(result.boxes),
            "image_size": {"width": img.shape[1], "height": img.shape[0]},
            "detections": [
                {"class": b.class_name, "score": round(b.score, 3),
                 "bbox": [int(b.x1), int(b.y1), int(b.x2), int(b.y2)]}
                for b in result.boxes],
            "overlay": _png_b64(vis),
            "is_heatmap": result.heatmap is not None,
        })

    @app.post("/predict")
    async def predict(file: UploadFile = File(...), method: str = Query("classical"),
                      conf: float = Query(0.25)):
        img = _read_upload(await file.read())
        r = inspector.predict(img, method=method, conf=conf)
        return JSONResponse(r.to_dict())

    @app.post("/predict/overlay")
    async def predict_overlay(file: UploadFile = File(...), method: str = Query("classical"),
                              conf: float = Query(0.25)):
        from PIL import Image
        img = _read_upload(await file.read())
        r = inspector.predict(img, method=method, conf=conf)
        vis = r.heatmap if r.heatmap is not None else overlay_boxes(img, preds=r.boxes)
        buf = io.BytesIO(); Image.fromarray(vis).save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")

    # Web console (mounted last so /api/* take precedence).
    if WEB_DIR.exists():
        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anomaly-model", default="models/anomaly_normal_net")
    ap.add_argument("--patchcore-model", default="models/patchcore_normal_net")
    ap.add_argument("--yolo-weights", default="models/yolo_damage_v1.pt")
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
    )
    LOGGER.info("Methods: %s", inspector.available_methods())
    LOGGER.info("Console: http://%s:%d", args.host, args.port)
    import uvicorn
    uvicorn.run(build_app(inspector), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
