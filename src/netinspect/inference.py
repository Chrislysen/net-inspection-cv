"""Unified inference facade over the three methods.

A single ``NetInspector`` lazily loads whichever models are configured and runs
any of them through one ``predict`` call returning a standard result. The CLI
batch/video runner, the FastAPI service, and the Streamlit viewer all go through
this class so behaviour is identical everywhere.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .classical_baseline import ClassicalConfig
from .classical_baseline import detect as classical_detect
from .utils import BBox, get_logger, optional_import

LOGGER = get_logger()

METHODS = ("classical", "anomaly", "patchcore", "yolo", "ensemble", "permissive")


@dataclass
class InferenceResult:
    method: str
    boxes: list[BBox]
    elapsed_ms: float
    heatmap: np.ndarray | None = None      # anomaly only
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "num_detections": len(self.boxes),
            "detections": [
                {"bbox": [round(v, 1) for v in b.to_list()], "score": round(b.score, 4),
                 "class_name": b.class_name, "class_id": b.class_id}
                for b in self.boxes
            ],
            "meta": self.meta,
        }


class NetInspector:
    """Lazily-loaded multi-method inspector.

    Parameters
    ----------
    classical_cfg : ClassicalConfig, optional
    anomaly_model_path : str, optional
        Prefix passed to ``AnomalyModel.load`` (a ``.npz`` file).
    yolo_weights : str, optional
        Path to a trained YOLO ``.pt`` checkpoint.
    """

    def __init__(self, classical_cfg: ClassicalConfig | None = None,
                 anomaly_model_path: str | Path | None = None,
                 patchcore_model_path: str | Path | None = None,
                 yolo_weights: str | Path | None = None,
                 seg_weights: str | Path | None = None,
                 permissive_weights: str | Path | None = None):
        self.classical_cfg = classical_cfg or ClassicalConfig()
        self._anomaly_path = str(anomaly_model_path) if anomaly_model_path else None
        self._patchcore_path = str(patchcore_model_path) if patchcore_model_path else None
        self._yolo_weights = str(yolo_weights) if yolo_weights else None
        self._seg_weights = str(seg_weights) if seg_weights else None
        self._permissive_weights = (str(permissive_weights)
                                    if permissive_weights else None)
        self._anomaly_model = None
        self._patchcore_model = None
        self._yolo_model = None
        self._seg_model = None
        self._permissive_model = None

        # Two separate hazards, two separate locks.
        #
        # LOADING: the lazy getters were check-then-set, so two request threads
        # arriving together both saw None and both loaded the same weights —
        # doubling peak memory for a large model and racing on the assignment.
        #
        # INFERENCE: an Ultralytics model object is stateful and is not
        # documented as thread-safe, yet the service lets several request
        # threads call predict() on the *same* object. Serialising inference is
        # the conservative choice: the concurrency cap in the service already
        # bounds the queue, and a wrong box under load is far worse than a
        # queued one. Swap this for a per-thread model pool if throughput ever
        # matters more than certainty.
        self._load_lock = threading.Lock()
        self._infer_lock = threading.RLock()

    # -- availability -------------------------------------------------------
    def available_methods(self) -> list[str]:
        methods = ["classical"]
        if self._anomaly_path and Path(self._anomaly_path).with_suffix(".npz").exists():
            methods.append("anomaly")
        if self._patchcore_path and Path(self._patchcore_path).with_suffix(".npz").exists() \
                and optional_import("torchvision") is not None:
            methods.append("patchcore")
        if self._yolo_weights and Path(self._yolo_weights).exists() \
                and optional_import("ultralytics") is not None:
            methods.append("yolo")
        if (self._permissive_weights and Path(self._permissive_weights).exists()
                and optional_import("torchvision") is not None):
            methods.append("permissive")
        if self._yolo_weights and self._seg_weights \
                and Path(self._yolo_weights).exists() and Path(self._seg_weights).exists() \
                and optional_import("ultralytics") is not None:
            methods.append("ensemble")
        return methods

    # -- lazy loaders -------------------------------------------------------
    def _load_once(self, attr: str, build):
        """Double-checked lazy load: build the model at most once, ever."""
        got = getattr(self, attr)
        if got is not None:
            return got
        with self._load_lock:
            got = getattr(self, attr)          # another thread may have won
            if got is None:
                got = build()
                setattr(self, attr, got)
            return got

    def _anomaly(self):
        from .anomaly import AnomalyModel
        return self._load_once("_anomaly_model",
                               lambda: AnomalyModel.load(self._anomaly_path))

    def _patchcore(self):
        from .patchcore import PatchCoreModel
        return self._load_once("_patchcore_model",
                               lambda: PatchCoreModel.load(self._patchcore_path))

    def _seg(self):
        from .model_baseline import load_model
        return self._load_once("_seg_model", lambda: load_model(self._seg_weights))

    def _yolo(self):
        from .model_baseline import load_model
        return self._load_once("_yolo_model", lambda: load_model(self._yolo_weights))

    def _permissive(self):
        from .permissive_baseline import load_model as load_permissive
        return self._load_once("_permissive_model",
                               lambda: load_permissive(self._permissive_weights))

    # -- inference ----------------------------------------------------------
    def predict(self, image_rgb: np.ndarray, method: str = "classical",
                conf: float = 0.25) -> InferenceResult:
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")
        t0 = time.perf_counter()

        if method == "classical":
            res = classical_detect(image_rgb, self.classical_cfg)
            boxes = [b for b in res.boxes if b.score >= conf]
            heatmap = None
            meta = res.debug
        elif method == "anomaly":
            from .anomaly import anomaly_heatmap, score_image
            ar = score_image(image_rgb, self._anomaly())
            boxes = [b for b in ar.boxes if b.score >= conf]
            heatmap = anomaly_heatmap(image_rgb, ar, self._anomaly())
            meta = {"max_score": ar.max_score, "threshold": self._anomaly().threshold}
        elif method == "patchcore":
            from .patchcore import heatmap as pc_heatmap
            from .patchcore import score_image as pc_score
            pr = pc_score(image_rgb, self._patchcore())
            boxes = [b for b in pr.boxes if b.score >= conf]
            heatmap = pc_heatmap(image_rgb, pr, self._patchcore())
            meta = {"max_score": pr.max_score, "threshold": self._patchcore().threshold}
        elif method == "ensemble":
            from .ensemble import EnsembleConfig, combine
            from .model_baseline import YoloConfig, predict_image
            # Held across BOTH calls: the ensemble's whole premise is that the
            # two models judged the same frame, and interleaving another
            # request's inference between them is exactly how that stops being
            # true.
            with self._infer_lock:
                det = predict_image(self._yolo(), image_rgb, YoloConfig(conf=0.01, iou=0.5))
                seg = predict_image(self._seg(), image_rgb, YoloConfig(conf=0.01, iou=0.5))
            ecfg = EnsembleConfig(det_conf=conf, seg_conf=conf, mode="agree")
            boxes = combine(det, seg, ecfg)
            heatmap = None
            meta = {"det_weights": self._yolo_weights, "seg_weights": self._seg_weights,
                    "rule": "det proposes, seg confirms (box agreement)"}
        elif method == "permissive":
            # torchvision (BSD-3-Clause). No Ultralytics anywhere in this
            # path, which is the entire reason it exists.
            from .permissive_baseline import PermissiveConfig
            from .permissive_baseline import predict_image as predict_permissive
            with self._infer_lock:
                boxes = predict_permissive(self._permissive(), image_rgb,
                                           PermissiveConfig(conf=conf))
            heatmap = None
            meta = {"weights": self._permissive_weights,
                    "licence": "torchvision BSD-3-Clause; AGPL-free path"}
        else:  # yolo
            from .model_baseline import YoloConfig, predict_image
            with self._infer_lock:
                boxes = predict_image(self._yolo(), image_rgb,
                                      YoloConfig(conf=conf, iou=0.5))
            heatmap = None
            meta = {"weights": self._yolo_weights}

        elapsed = (time.perf_counter() - t0) * 1000
        return InferenceResult(method, boxes, elapsed, heatmap, meta)
