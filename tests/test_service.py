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


# --------------------------------------------------------------------------- #
# Drop-to-analyse
# --------------------------------------------------------------------------- #
def _png_bytes(w=64, h=48):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def test_analyze_returns_overlay_detections_and_disclaimer_in_one_call(client):
    """The drop zone needs everything from a single round trip."""
    r = client.post("/api/analyze?method=classical&conf=0.25&ood=0",
                    files={"file": ("frame.png", _png_bytes(), "image/png")})
    assert r.status_code == 200
    d = r.json()
    assert d["overlay"].startswith("data:image/png;base64,")
    assert d["image_size"] == {"width": 64, "height": 48}
    assert isinstance(d["detections"], list)
    assert "synthetic" in d["disclaimer"].lower()
    assert d["filename"] == "frame.png"


def test_analyze_rejects_a_non_image(client):
    # method= must be given: the endpoint validates it before touching the
    # upload, and the test inspector only has the classical method.
    r = client.post("/api/analyze?method=classical",
                    files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_analyze_rejects_bytes_that_are_not_a_decodable_image(client):
    r = client.post("/api/analyze?method=classical",
                    files={"file": ("fake.png", b"not really a png", "image/png")})
    assert r.status_code == 400


def test_analyze_rejects_an_unknown_method(client):
    r = client.post("/api/analyze?method=telepathy",
                    files={"file": ("f.png", _png_bytes(), "image/png")})
    assert r.status_code in (400, 422)


def test_analyze_strips_any_path_from_the_filename(client):
    """An uploaded name is attacker-controlled; only the basename is echoed."""
    r = client.post("/api/analyze?method=classical&ood=0",
                    files={"file": ("../../etc/passwd.png", _png_bytes(), "image/png")})
    assert r.status_code == 200
    assert "/" not in r.json()["filename"] and "\\" not in r.json()["filename"]


# --------------------------------------------------------------------------- #
# Live session endpoints
# --------------------------------------------------------------------------- #
def test_live_status_is_safe_before_anything_starts(client):
    assert client.get("/api/live/status").json()["running"] is False


def test_streaming_without_a_session_is_a_conflict_not_a_crash(client):
    assert client.get("/api/live/stream").status_code == 409


def test_starting_an_unopenable_source_returns_400(client, tmp_path):
    r = client.post(f"/api/live/start?source={tmp_path / 'nope.mp4'}&method=classical")
    assert r.status_code == 400
    assert "could not open" in r.json()["detail"].lower()


def test_stopping_when_nothing_runs_is_harmless(client):
    assert client.post("/api/live/stop").json() == {"stopped": False}


def test_live_start_status_and_stop_round_trip(client, tmp_path):
    """Full lifecycle against a real generated video file."""
    cv2 = pytest.importorskip("cv2")
    import time as _time

    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("no mp4v encoder available")
    for i in range(10):
        writer.write(np.full((48, 64, 3), (i * 25) % 255, dtype=np.uint8))
    writer.release()
    if not path.exists() or path.stat().st_size == 0:
        pytest.skip("video file was not produced")

    started = client.post(
        f"/api/live/start?source={path}&method=classical&conf=0.5&min_hits=2&ood=0&loop=true")
    assert started.status_code == 200, started.text
    assert started.json()["running"] is True
    try:
        deadline = _time.time() + 8
        while _time.time() < deadline:
            s = client.get("/api/live/status").json()
            if s.get("frames_inferred", 0) > 1:
                break
            _time.sleep(0.1)
        s = client.get("/api/live/status").json()
        assert s["running"] is True
        assert s["frames_inferred"] >= 1
        assert s["source_kind"] == "video file"
        assert "synthetic" in s["disclaimer"].lower()
    finally:
        assert client.post("/api/live/stop").json() == {"stopped": True}
    assert client.get("/api/live/status").json()["running"] is False
