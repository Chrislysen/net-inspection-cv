"""Integration/smoke tests for the FastAPI service (no heavy models needed)."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

# Import the service module from scripts/.
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
import serve  # noqa: E402

from netinspect.inference import NetInspector  # noqa: E402


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    # classical-only inspector (no model files needed) keeps the test fast & offline.
    app = serve.build_app(NetInspector())
    return TestClient(app)


def test_health_and_ready(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and "classical" in h["methods"]
    # console capabilities surfaced for an intuitive UI
    assert "source_info" in h and "ood_gate" in h
    r = client.get("/api/ready")
    assert r.status_code == 200 and r.json()["ready"] is True
    assert "X-Request-ID" in r.headers


def test_metrics_prometheus(client):
    client.get("/api/health")  # generate some traffic
    txt = client.get("/api/metrics").text
    assert "netinspect_requests_total" in txt
    assert "netinspect_uptime_seconds" in txt


def test_predict_upload_and_disclaimer(client):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(buf, format="JPEG")
    r = client.post("/predict?method=classical&conf=0.3",
                    files={"file": ("f.jpg", buf.getvalue(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["method"] == "classical" and "disclaimer" in body


def test_unsupported_content_type_rejected(client):
    r = client.post("/predict", files={"file": ("f.txt", b"not an image", "text/plain")})
    assert r.status_code == 415


def test_bad_image_rejected(client):
    r = client.post("/predict", files={"file": ("f.jpg", b"garbage", "image/jpeg")})
    assert r.status_code == 400


def test_unknown_method_rejected(client):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(buf, format="PNG")
    r = client.post("/predict?method=nope", files={"file": ("f.png", buf.getvalue(), "image/png")})
    assert r.status_code == 400


def test_path_traversal_blocked(client):
    # A source must exist to reach _resolve; if none are present, the 404 is for the
    # source — either way a traversal name must never return a file outside the dir.
    r = client.get("/api/image", params={"source": "Synthetic demo",
                                         "name": "../../../../pyproject.toml"})
    assert r.status_code == 404
