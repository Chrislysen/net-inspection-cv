"""Unified inference facade over the three methods.

A single ``NetInspector`` lazily loads whichever models are configured and runs
any of them through one ``predict`` call returning a standard result. The CLI
batch/video runner, the FastAPI service, and the Streamlit viewer all go through
this class so behaviour is identical everywhere.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .classical_baseline import ClassicalConfig, detect as classical_detect
from .utils import BBox, get_logger, optional_import

LOGGER = get_logger()

METHODS = ("classical", "anomaly", "patchcore", "yolo")


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
                 yolo_weights: str | Path | None = None):
        self.classical_cfg = classical_cfg or ClassicalConfig()
        self._anomaly_path = str(anomaly_model_path) if anomaly_model_path else None
        self._patchcore_path = str(patchcore_model_path) if patchcore_model_path else None
        self._yolo_weights = str(yolo_weights) if yolo_weights else None
        self._anomaly_model = None
        self._patchcore_model = None
        self._yolo_model = None

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
        return methods

    # -- lazy loaders -------------------------------------------------------
    def _anomaly(self):
        if self._anomaly_model is None:
            from .anomaly import AnomalyModel
            self._anomaly_model = AnomalyModel.load(self._anomaly_path)
        return self._anomaly_model

    def _patchcore(self):
        if self._patchcore_model is None:
            from .patchcore import PatchCoreModel
            self._patchcore_model = PatchCoreModel.load(self._patchcore_path)
        return self._patchcore_model

    def _yolo(self):
        if self._yolo_model is None:
            from .model_baseline import load_model
            self._yolo_model = load_model(self._yolo_weights)
        return self._yolo_model

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
            from .anomaly import score_image, anomaly_heatmap
            ar = score_image(image_rgb, self._anomaly())
            boxes = [b for b in ar.boxes if b.score >= conf]
            heatmap = anomaly_heatmap(image_rgb, ar, self._anomaly())
            meta = {"max_score": ar.max_score, "threshold": self._anomaly().threshold}
        elif method == "patchcore":
            from .patchcore import score_image as pc_score, heatmap as pc_heatmap
            pr = pc_score(image_rgb, self._patchcore())
            boxes = [b for b in pr.boxes if b.score >= conf]
            heatmap = pc_heatmap(image_rgb, pr, self._patchcore())
            meta = {"max_score": pr.max_score, "threshold": self._patchcore().threshold}
        else:  # yolo
            from .model_baseline import YoloConfig, predict_image
            boxes = predict_image(self._yolo(), image_rgb,
                                  YoloConfig(conf=conf, iou=0.5))
            heatmap = None
            meta = {"weights": self._yolo_weights}

        elapsed = (time.perf_counter() - t0) * 1000
        return InferenceResult(method, boxes, elapsed, heatmap, meta)
