"""netinspect — a prototype toolkit for aquaculture net damage detection.

Modules
-------
* ``data``               dataset discovery, YOLO label parsing, summaries
* ``preprocess``         underwater image enhancement (CLAHE, WB, denoise)
* ``video``              video frame extraction
* ``classical_baseline`` explainable OpenCV anomaly baseline
* ``model_baseline``     Ultralytics YOLOv8 detection/segmentation wrapper
* ``evaluate``           detection / segmentation / image-level metrics
* ``visualize``          overlays, comparisons, galleries
* ``synthetic``          placeholder data generator (pipeline testing only)
* ``utils``              IO, geometry, optional-dependency handling

This is a research prototype. Synthetic-data results do not represent
real-world performance — see the README and report.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .utils import BBox, iou_xyxy, mask_iou  # noqa: F401
