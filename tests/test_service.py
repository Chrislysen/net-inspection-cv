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


def test_readiness_fails_when_a_configured_model_is_missing(tmp_path):
    """503 must be REACHABLE.

    The old check asked whether "classical" was available. It is hardcoded
    present, so the answer was always yes and the probe could not fail — an
    orchestrator would hold a pod in the load-balancer whose model volume never
    mounted, and every request naming that method would 500.
    """
    from fastapi.testclient import TestClient

    absent = tmp_path / "never_mounted.pt"
    app = serve.build_app(NetInspector(yolo_weights=absent))
    r = TestClient(app).get("/api/ready")

    assert r.status_code == 503, (
        "a deployment configured for yolo whose weights are unreadable reported "
        "itself ready; the readiness probe cannot fail")
    body = r.json()
    assert body["ready"] is False
    assert "yolo" in body["missing"], body
    assert "yolo" in body["detail"], "the response must name what is missing"


def test_readiness_is_green_when_everything_configured_resolved(tmp_path):
    """The converse: no configuration means nothing can be missing."""
    from fastapi.testclient import TestClient

    r = TestClient(serve.build_app(NetInspector())).get("/api/ready")
    assert r.status_code == 200
    assert r.json()["missing"] == []


def test_configured_and_available_are_different_questions(tmp_path):
    """The gap between them is the whole signal."""
    insp = NetInspector(yolo_weights=tmp_path / "gone.pt")
    assert "yolo" in insp.configured_methods(), "it was asked for"
    assert "yolo" not in insp.available_methods(), "it did not resolve"


def test_classical_is_not_advertised_when_opencv_is_broken(monkeypatch):
    """Found by building the container, not by reading the code.

    The image shipped without libGL/libxcb, so `import cv2` failed — and the
    service still listed "classical" as available, because that string was
    hardcoded rather than checked. A request for it 500'd, and readiness stayed
    green, since readiness leans on exactly that method. An orchestrator would
    have kept routing traffic to a container whose detector could not load.
    """
    from netinspect import inference as I

    real = I.optional_import
    monkeypatch.setattr(I, "optional_import",
                        lambda name, *a, **k: None if name == "cv2" else real(name, *a, **k))

    insp = NetInspector()
    assert "classical" not in insp.available_methods(), (
        "classical was advertised with no OpenCV present — the check is a "
        "constant again, and readiness cannot detect a broken image")


def test_readiness_goes_red_when_opencv_is_missing(monkeypatch):
    """The consequence of the above: the probe must actually fail."""
    from fastapi.testclient import TestClient

    from netinspect import inference as I

    real = I.optional_import
    monkeypatch.setattr(I, "optional_import",
                        lambda name, *a, **k: None if name == "cv2" else real(name, *a, **k))

    r = TestClient(serve.build_app(NetInspector())).get("/api/ready")
    assert r.status_code == 503, "a container with no working OpenCV reported ready"
    assert "classical" in r.json()["missing"]


def test_the_mjpeg_stream_does_not_occupy_a_threadpool_worker(client):
    """It must be a coroutine, not a sync generator run in the threadpool.

    Starlette runs a sync streaming generator in the threadpool, so each viewer
    held one worker for the entire life of the stream — sleeping in it between
    frames. AnyIO hands out 40 by default and inference shares that pool, so a
    few dozen open tabs would starve every /predict in the process. Reverting
    this to `def` reintroduces that silently: nothing else fails, the service
    just stops answering under load.
    """
    import inspect as _inspect

    route = next(r for r in client.app.routes
                 if getattr(r, "path", None) == "/api/live/stream")
    assert _inspect.iscoroutinefunction(route.endpoint), (
        "/api/live/stream is a sync def — every viewer will pin a threadpool "
        "worker and starve inference")

    src = _inspect.getsource(route.endpoint)
    assert "time.sleep" not in src, (
        "time.sleep blocks a worker (or the event loop); use anyio.sleep")
    assert "anyio.sleep" in src, "the frame pacing must yield to the event loop"


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


def _client_for_media(root):
    """A client whose live-source policy permits files under `root`.

    Live sources are default-deny, so a test that opens a file has to say where
    files are allowed to come from — the same thing an operator does with
    NETINSPECT_MEDIA_ROOT.
    """
    from fastapi.testclient import TestClient

    from netinspect.security import SecurityConfig
    return TestClient(serve.build_app(
        NetInspector(), security=SecurityConfig(media_root=Path(root).resolve())))


def test_starting_an_unopenable_source_returns_400(tmp_path):
    c = _client_for_media(tmp_path)
    r = c.post(f"/api/live/start?source={tmp_path / 'nope.mp4'}&method=classical")
    assert r.status_code == 400
    assert "could not open" in r.json()["detail"].lower()


def test_a_source_outside_the_media_root_is_refused(client, tmp_path):
    """The service-level check on the SSRF / arbitrary-read surface."""
    outside = tmp_path / "secret.mp4"
    outside.write_bytes(b"x")
    r = client.post(f"/api/live/start?source={outside}&method=classical")
    assert r.status_code == 403
    assert "NETINSPECT_MEDIA_ROOT" in r.json()["detail"]


def test_a_stream_url_is_refused_without_an_allowlist(client):
    r = client.post("/api/live/start?source=rtsp://camera.example.com/1&method=classical")
    assert r.status_code == 403


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

    client = _client_for_media(tmp_path)
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


# --------------------------------------------------------------------------- #
# Net model endpoints
#
# The contract worth testing is not "returns 200" — it is that a response can
# never let a client mistake the declared cage shell for measured geometry.
# --------------------------------------------------------------------------- #
def _has_map(client) -> bool:
    return bool(client.get("/api/maps").json()["maps"])


def test_maps_lists_available_passes(client):
    body = client.get("/api/maps").json()
    assert isinstance(body["maps"], list)


def test_scene_marks_the_cage_as_declared_not_measured(client):
    if not _has_map(client):
        pytest.skip("no inspection map built; run scripts/map_inspection.py")
    clip = client.get("/api/maps").json()["maps"][0]
    s = client.get("/api/scene", params={"clip": clip}).json()
    assert s["pen"]["declared"] is True
    assert s["barge"]["declared"] is True
    assert s["provenance"]["declared"] and s["provenance"]["measured"]
    assert "not measured" in s["pen"]["note"].lower()


def test_scene_reports_coverage_against_the_whole_cage(client):
    if not _has_map(client):
        pytest.skip("no inspection map built")
    clip = client.get("/api/maps").json()["maps"][0]
    s = client.get("/api/scene", params={"clip": clip}).json()
    cov = s["coverage"]
    # A single pass is a sliver of a real cage, and the API must say so rather
    # than reporting only the flattering absolute number.
    assert 0.0 < cov["area_percent"] < 100.0
    assert cov["net_area_m2"] > cov["swept_area_m2"]
    assert cov["passes_to_cover_ring"] >= 1


def test_a_bigger_cage_makes_the_same_pass_a_smaller_fraction(client):
    if not _has_map(client):
        pytest.skip("no inspection map built")
    clip = client.get("/api/maps").json()["maps"][0]
    small = client.get("/api/scene", params={"clip": clip, "circumference_m": 90}).json()
    big = client.get("/api/scene", params={"clip": clip, "circumference_m": 200}).json()
    assert big["coverage"]["area_percent"] < small["coverage"]["area_percent"]


def test_moving_the_barge_changes_where_sites_are_reported_to_be(client):
    if not _has_map(client):
        pytest.skip("no inspection map built")
    clip = client.get("/api/maps").json()["maps"][0]
    a = client.get("/api/scene", params={"clip": clip, "barge_bearing_deg": 0}).json()
    b = client.get("/api/scene", params={"clip": clip, "barge_bearing_deg": 180}).json()
    if not a["sites"]:
        pytest.skip("map has no sites")
    assert (a["sites"][0]["placed"]["arc_from_barge_m"]
            != b["sites"][0]["placed"]["arc_from_barge_m"])


def test_scene_rejects_an_impossible_cage(client):
    if not _has_map(client):
        pytest.skip("no inspection map built")
    clip = client.get("/api/maps").json()["maps"][0]
    r = client.get("/api/scene", params={"clip": clip,
                                        "cylinder_depth_m": 0, "cone_depth_m": 0})
    assert r.status_code == 400


def test_scene_refuses_an_unknown_or_traversing_clip(client):
    assert client.get("/api/scene", params={"clip": "nope"}).status_code == 404
    assert client.get("/api/scene",
                      params={"clip": "../../../etc/passwd"}).status_code == 404


def test_site_crop_is_served_and_path_traversal_is_refused(client):
    if not _has_map(client):
        pytest.skip("no inspection map built")
    clip = client.get("/api/maps").json()["maps"][0]
    s = client.get("/api/scene", params={"clip": clip}).json()
    if s["crops"]:
        site = int(next(iter(s["crops"])))
        ok = client.get("/api/scene/crop", params={"clip": clip, "site": site})
        assert ok.status_code == 200 and ok.headers["content-type"] == "image/jpeg"
    bad = client.get("/api/scene/crop", params={"clip": "../secrets", "site": 1})
    assert bad.status_code == 404


# --------------------------------------------------------------------------- #
# The AGPL-free detector must be reachable from the console, not only the CLI
# --------------------------------------------------------------------------- #
def test_the_permissive_method_is_exposed_when_its_weights_exist():
    """Regression: permissive_baseline shipped and the gate could use it, but
    serve.py had no flag for it — so the licence-clean path existed everywhere
    except the interface people actually look at."""
    weights = Path("models/permissive_v1.pt")
    if not weights.exists():
        pytest.skip("permissive weights not built; run scripts/train_permissive.py")
    from fastapi.testclient import TestClient
    app = serve.build_app(NetInspector(permissive_weights=str(weights)))
    methods = TestClient(app).get("/api/health").json()["methods"]
    assert "permissive" in methods


def test_the_console_offers_every_method_the_server_can_run():
    """The method list in the UI and the one the service advertises must not
    drift apart, or a method exists that nobody can select."""
    import re
    js = Path("web/app.js").read_text(encoding="utf-8")
    order = re.search(r"METHOD_ORDER\s*=\s*\[(.*?)\]", js, re.S).group(1)
    listed = set(re.findall(r'"([a-z]+)"', order))
    from netinspect.inference import METHODS
    assert listed == set(METHODS), f"UI {listed} vs server {set(METHODS)}"
