"""FastAPI inference service exposing the net-inspection methods over HTTP.

Endpoints
---------
* ``GET  /health``            — liveness + which methods are loaded
* ``GET  /methods``           — available methods
* ``POST /predict``           — multipart image upload -> JSON detections
* ``POST /predict/overlay``   — same, returns an overlay PNG

Run
---
    python scripts/serve.py --yolo-weights runs/detect/train/weights/best.pt \\
        --anomaly-model outputs/anomaly/model --port 8000
    # then: curl -F file=@frame.jpg "http://localhost:8000/predict?method=classical"

This is a thin, honest serving layer for a prototype — single process, no auth,
no batching. It demonstrates the integration shape, not a hardened deployment.

Note: this module deliberately does NOT use ``from __future__ import annotations``
because FastAPI/pydantic must resolve the ``UploadFile`` annotations at runtime.
"""
import argparse
import io

import _common  # noqa: F401
import numpy as np
from netinspect.inference import NetInspector
from netinspect.utils import get_logger
from netinspect.visualize import overlay_boxes

LOGGER = get_logger()


def build_app(inspector: NetInspector):
    from fastapi import FastAPI, File, Query, UploadFile
    from fastapi.responses import JSONResponse, Response

    app = FastAPI(title="net-inspection-cv", version="0.1.0",
                  description="Prototype aquaculture net damage inspection API.")

    def _read_upload(data: bytes) -> np.ndarray:
        from PIL import Image
        return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))

    @app.get("/health")
    def health():
        return {"status": "ok", "methods": inspector.available_methods()}

    @app.get("/methods")
    def methods():
        return {"available": inspector.available_methods()}

    @app.post("/predict")
    async def predict(file: UploadFile = File(...),
                      method: str = Query("classical"),
                      conf: float = Query(0.25)):
        img = _read_upload(await file.read())
        result = inspector.predict(img, method=method, conf=conf)
        payload = result.to_dict()
        payload["image_size"] = {"width": img.shape[1], "height": img.shape[0]}
        payload["disclaimer"] = ("Prototype. Synthetic/proxy-trained models; not "
                                 "validated on real damage. Human review required.")
        return JSONResponse(payload)

    @app.post("/predict/overlay")
    async def predict_overlay(file: UploadFile = File(...),
                              method: str = Query("classical"),
                              conf: float = Query(0.25)):
        from PIL import Image
        img = _read_upload(await file.read())
        result = inspector.predict(img, method=method, conf=conf)
        vis = result.heatmap if result.heatmap is not None else overlay_boxes(img, preds=result.boxes)
        buf = io.BytesIO()
        Image.fromarray(vis).save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anomaly-model", default=None)
    ap.add_argument("--yolo-weights", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    inspector = NetInspector(anomaly_model_path=args.anomaly_model,
                             yolo_weights=args.yolo_weights)
    LOGGER.info("Methods available: %s", inspector.available_methods())
    import uvicorn
    uvicorn.run(build_app(inspector), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
