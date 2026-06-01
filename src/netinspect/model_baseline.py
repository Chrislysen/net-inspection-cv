"""ML baseline: a thin wrapper around Ultralytics YOLOv8 detection / segmentation.

Design goals
------------
* **Graceful degradation.** If ``ultralytics`` is not installed, importing this
  module still works; only the YOLO calls raise a clear, actionable error.
* **One code path for det and seg.** ``task="detect"`` or ``"segment"``.
* **Standard YOLO dataset format** so real labelled data drops straight in.

This is a *baseline*: small model, default hyper-parameters, no domain-specific
tuning. Its job is to establish a trainable reference point, not to be optimal.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .utils import BBox, get_logger, optional_import

LOGGER = get_logger()

ULTRALYTICS_HINT = (
    "Install it with `pip install ultralytics` (pulls in torch). The classical "
    "baseline and evaluation utilities work without it."
)


def ultralytics_available() -> bool:
    return optional_import("ultralytics") is not None


@dataclass
class YoloConfig:
    task: str = "detect"               # "detect" | "segment"
    model: str = "yolov8n.pt"          # base weights (auto-downloaded by ultralytics)
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.50
    epochs: int = 50
    batch: int = 16
    device: str | None = None          # None -> ultralytics auto-selects


def _load_yolo(weights: str, task: str):
    ultra = optional_import("ultralytics")
    if ultra is None:
        raise RuntimeError(f"ultralytics is not installed. {ULTRALYTICS_HINT}")
    return ultra.YOLO(weights, task=task)


def train(data_yaml: str | Path, cfg: YoloConfig, project: str | None = None,
          name: str | None = None) -> Path:
    """Train a YOLO model on a YOLO-format dataset and return the best weights path.

    ``data_yaml`` must point to a standard Ultralytics dataset config (see
    ``configs/yolo_dataset.yaml``). Raises if ultralytics is missing or the
    dataset config does not exist.
    """
    data_yaml = Path(data_yaml)
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {data_yaml}. Point --data at a YOLO "
            "dataset YAML with real labelled images."
        )
    model = _load_yolo(cfg.model, cfg.task)
    LOGGER.info("Training YOLO (%s) for %d epochs on %s", cfg.task, cfg.epochs, data_yaml)
    train_kwargs = dict(
        data=str(data_yaml), epochs=cfg.epochs, imgsz=cfg.imgsz,
        batch=cfg.batch, device=cfg.device,
    )
    if project:
        train_kwargs["project"] = project
    if name:
        train_kwargs["name"] = name
    results = model.train(**train_kwargs)
    save_dir = Path(getattr(results, "save_dir", Path("runs") / "detect" / "train"))
    best = save_dir / "weights" / "best.pt"
    LOGGER.info("Training done. Best weights: %s", best)
    return best


def predict_image(model, image_rgb: np.ndarray, cfg: YoloConfig,
                  class_names: list[str] | None = None) -> list[BBox]:
    """Run inference on a single RGB image and return boxes in pixel coords."""
    results = model.predict(source=image_rgb[..., ::-1],  # ultralytics expects BGR
                            imgsz=cfg.imgsz, conf=cfg.conf, iou=cfg.iou,
                            device=cfg.device, verbose=False)
    boxes: list[BBox] = []
    for res in results:
        names = class_names or getattr(res, "names", None) or {}
        if res.boxes is None:
            continue
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cls = int(b.cls[0])
            score = float(b.conf[0])
            name = names.get(cls, f"class_{cls}") if isinstance(names, dict) else str(cls)
            boxes.append(BBox(x1, y1, x2, y2, cls, name, score))
    return boxes


def load_model(weights: str | Path, task: str = "detect"):
    """Load a trained YOLO checkpoint for inference."""
    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights}. Train a model first (scripts/train_yolo.py) "
            "or supply a checkpoint."
        )
    return _load_yolo(str(weights), task)
