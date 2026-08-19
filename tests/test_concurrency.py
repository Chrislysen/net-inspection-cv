"""Concurrency safety of the shared inference facade.

The service runs several request threads against ONE NetInspector, which holds
one model object per method. Two hazards follow, and neither shows up in a
single-threaded test:

* the lazy loaders were check-then-set, so two threads arriving together both
  loaded the same weights — double peak memory and a race on the assignment;
* an Ultralytics model is stateful and not documented as thread-safe, yet
  several threads called ``predict`` on the same instance.

These tests use a stand-in model so they run without weights and without a GPU,
and they *fail* against the unlocked implementation rather than merely passing
against the fixed one.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from netinspect.inference import NetInspector


class _SlowModel:
    """A model that is slow enough to overlap, and notices if it is re-entered."""

    def __init__(self):
        self.calls = 0
        self.concurrent = 0
        self.max_concurrent = 0
        self._guard = threading.Lock()

    def __call__(self):
        with self._guard:
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
            self.calls += 1
        time.sleep(0.02)                      # long enough for others to arrive
        with self._guard:
            self.concurrent -= 1


def _run_threads(fn, n=8):
    errors = []

    def wrapped():
        try:
            fn()
        except Exception as exc:              # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=wrapped) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors, f"threads raised: {errors[:3]}"


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def test_a_model_is_loaded_at_most_once_under_concurrent_first_use():
    """Regression: check-then-set let every arriving thread load its own copy."""
    insp = NetInspector()
    loads = []
    gate = threading.Event()

    def build():
        gate.wait(timeout=5)                  # make the window wide on purpose
        loads.append(1)
        return object()

    def use():
        insp._load_once("_yolo_model", build)

    threads = [threading.Thread(target=use) for _ in range(8)]
    for t in threads:
        t.start()
    time.sleep(0.05)
    gate.set()
    for t in threads:
        t.join(timeout=30)

    assert len(loads) == 1, f"model was built {len(loads)} times, expected once"


def test_all_threads_get_the_same_instance():
    insp = NetInspector()
    seen = []
    sentinel = object()
    _run_threads(lambda: seen.append(insp._load_once("_seg_model", lambda: sentinel)))
    assert seen and all(x is sentinel for x in seen)


def test_a_failed_load_does_not_cache_a_broken_model():
    insp = NetInspector()
    with pytest.raises(RuntimeError):
        insp._load_once("_yolo_model", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    # A second attempt must be allowed to succeed rather than returning None.
    ok = object()
    assert insp._load_once("_yolo_model", lambda: ok) is ok


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
def test_inference_on_the_shared_model_is_serialised(monkeypatch):
    """The core fix: only one thread inside the model at a time."""
    from netinspect import inference as I

    model = _SlowModel()
    insp = NetInspector(yolo_weights="fake.pt")
    insp._yolo_model = object()               # pretend it is loaded

    def fake_predict(_model, _img, _cfg):
        model()
        return []

    monkeypatch.setattr("netinspect.model_baseline.predict_image", fake_predict)
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    _run_threads(lambda: insp.predict(img, method="yolo", conf=0.25), n=8)

    assert model.calls == 8, "every request must still be served"
    assert model.max_concurrent == 1, (
        f"{model.max_concurrent} threads were inside the model at once — "
        "the shared Ultralytics object is not thread-safe")


def test_the_ensemble_holds_the_lock_across_both_models(monkeypatch):
    """Both models must judge the SAME frame with nothing interleaved."""
    model = _SlowModel()
    insp = NetInspector(yolo_weights="d.pt", seg_weights="s.pt")
    insp._yolo_model = object()
    insp._seg_model = object()

    monkeypatch.setattr("netinspect.model_baseline.predict_image",
                        lambda *_a, **_k: (model(), [])[1])
    monkeypatch.setattr("netinspect.ensemble.combine", lambda *_a, **_k: [])

    img = np.zeros((32, 32, 3), dtype=np.uint8)
    _run_threads(lambda: insp.predict(img, method="ensemble", conf=0.25), n=6)

    assert model.calls == 12, "two model calls per ensemble request"
    assert model.max_concurrent == 1


def test_the_classical_method_is_not_serialised_unnecessarily():
    """It holds no shared state, so it must not queue behind the model lock."""
    insp = NetInspector()
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    started = time.perf_counter()
    _run_threads(lambda: insp.predict(img, method="classical", conf=0.9), n=4)
    assert time.perf_counter() - started < 20
