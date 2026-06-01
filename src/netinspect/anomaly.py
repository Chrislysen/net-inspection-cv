"""Anomaly-detection baseline: model "normal net", flag deviations.

Motivation
----------
On real data, damage examples are scarce but *normal* net footage is abundant
(SOLAQUA is exactly this — many frames of undamaged net). That suits a one-class
/ anomaly-detection framing: learn what intact net looks like, then flag regions
that deviate.

Method (a deliberately simple, explainable PaDiM-style model)
------------------------------------------------------------
1. Split each frame into a grid of patches.
2. Describe every patch with a small set of hand-crafted features that capture
   net appearance: Lab colour statistics, local contrast, and edge/texture
   density (intact mesh is regular and textured).
3. Fit a single multivariate Gaussian to all patch features from normal frames
   (standardised, with covariance shrinkage for a stable inverse).
4. Score a new frame by the Mahalanobis distance of each patch to that Gaussian.
   High distance = "unlike normal net" → an anomaly map and candidate regions.

**Honesty.** This flags *deviation from normal appearance*, not validated
damage. On real footage it will also flag fish, biofouling, markers, and strong
lighting changes — those are out-of-distribution too. It is a candidate
screening tool whose threshold must be calibrated on real, reviewed data.

A learned deep backbone or an autoencoder is a natural extension; this
dependency-light statistical model is the honest baseline to beat.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .preprocess import PreprocessConfig, preprocess
from .utils import BBox, ensure_dir, get_logger, optional_import

LOGGER = get_logger()


@dataclass
class AnomalyConfig:
    resize: int = 512          # longest side before patching
    grid: int = 16             # patch grid is grid x grid (aspect-adjusted)
    shrinkage: float = 0.1     # covariance regularisation (0..1)
    threshold_percentile: float = 99.0  # train-distance percentile -> default cutoff


@dataclass
class AnomalyModel:
    mean: np.ndarray            # feature mean (for standardisation)
    std: np.ndarray             # feature std
    cov_inv: np.ndarray         # inverse covariance of standardised features
    feat_mean: np.ndarray       # mean of standardised features (~0)
    threshold: float            # Mahalanobis distance cutoff
    cfg: AnomalyConfig
    train_stats: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        ensure_dir(path.parent)
        np.savez(path, mean=self.mean, std=self.std, cov_inv=self.cov_inv,
                 feat_mean=self.feat_mean, threshold=np.array(self.threshold),
                 cfg=np.array([self.cfg.resize, self.cfg.grid, self.cfg.shrinkage,
                               self.cfg.threshold_percentile], dtype=float))

    @staticmethod
    def load(path: str | Path) -> "AnomalyModel":
        d = np.load(Path(path).with_suffix(".npz"), allow_pickle=True)
        c = d["cfg"]
        cfg = AnomalyConfig(int(c[0]), int(c[1]), float(c[2]), float(c[3]))
        return AnomalyModel(d["mean"], d["std"], d["cov_inv"], d["feat_mean"],
                            float(d["threshold"]), cfg)


def _patch_features(image_rgb: np.ndarray, cfg: AnomalyConfig) -> tuple[np.ndarray, tuple[int, int]]:
    """Return per-patch features [n_patches, F] and the (rows, cols) grid shape."""
    cv2 = optional_import("cv2")
    proc = preprocess(image_rgb, PreprocessConfig(resize=cfg.resize, clahe=False,
                                                  denoise=False, color_normalize=True))
    h, w = proc.shape[:2]
    rows = cfg.grid
    cols = max(1, int(round(cfg.grid * w / h)))
    ph, pw = h // rows, w // cols
    if ph == 0 or pw == 0:
        rows, cols = max(1, h), max(1, w)
        ph, pw = 1, 1

    if cv2 is not None:
        lab = cv2.cvtColor(proc, cv2.COLOR_RGB2LAB).astype(np.float32)
        gray = cv2.cvtColor(proc, cv2.COLOR_RGB2GRAY)
        edges = (cv2.Canny(gray, 40, 120) > 0).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
    else:  # NumPy fallback (no edges/grad detail)
        lab = proc.astype(np.float32)
        gray = proc.mean(axis=2)
        edges = np.zeros_like(gray)
        grad = np.abs(np.gradient(gray)[0]) + np.abs(np.gradient(gray)[1])

    feats = []
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * ph, (r + 1) * ph
            x0, x1 = c * pw, (c + 1) * pw
            cell_lab = lab[y0:y1, x0:x1]
            cell_gray = gray[y0:y1, x0:x1]
            feats.append([
                float(cell_lab[..., 0].mean()),  # L mean (brightness)
                float(cell_lab[..., 1].mean()),  # a mean (colour)
                float(cell_lab[..., 2].mean()),  # b mean (colour)
                float(cell_gray.std()),          # local contrast
                float(edges[y0:y1, x0:x1].mean()),  # edge density (mesh texture)
                float(grad[y0:y1, x0:x1].mean()),   # gradient magnitude
            ])
    return np.asarray(feats, dtype=np.float64), (rows, cols)


def fit(normal_images: list[np.ndarray], cfg: AnomalyConfig | None = None) -> AnomalyModel:
    """Fit the normal-net Gaussian from a list of RGB frames."""
    cfg = cfg or AnomalyConfig()
    if not normal_images:
        raise ValueError("fit() needs at least one normal image.")
    all_feats = [(_patch_features(im, cfg)[0]) for im in normal_images]
    X = np.vstack(all_feats)                       # [N, F]
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    Z = (X - mean) / std                           # standardise
    cov = np.cov(Z, rowvar=False)
    # Shrinkage towards the identity for a stable, invertible covariance.
    cov = (1 - cfg.shrinkage) * cov + cfg.shrinkage * np.eye(cov.shape[0])
    cov_inv = np.linalg.inv(cov)
    feat_mean = Z.mean(axis=0)

    # Calibrate the threshold from the training distance distribution.
    d = _mahalanobis(Z, feat_mean, cov_inv)
    threshold = float(np.percentile(d, cfg.threshold_percentile))
    stats = {"train_patches": int(X.shape[0]), "num_images": len(normal_images),
             "dist_mean": float(d.mean()), "dist_p99": float(np.percentile(d, 99))}
    LOGGER.info("Fitted anomaly model on %d frames / %d patches; threshold=%.3f",
                len(normal_images), X.shape[0], threshold)
    return AnomalyModel(mean, std, cov_inv, feat_mean, threshold, cfg, stats)


def _mahalanobis(Z: np.ndarray, mu: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    delta = Z - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", delta, cov_inv, delta))


@dataclass
class AnomalyResult:
    score_map: np.ndarray       # [rows, cols] Mahalanobis distances
    grid: tuple[int, int]
    boxes: list[BBox]           # candidate anomaly regions (original coords)
    max_score: float


def score_image(image_rgb: np.ndarray, model: AnomalyModel) -> AnomalyResult:
    """Score a frame, returning an anomaly map and thresholded candidate boxes."""
    cv2 = optional_import("cv2")
    feats, (rows, cols) = _patch_features(image_rgb, model.cfg)
    Z = (feats - model.mean) / model.std
    d = _mahalanobis(Z, model.feat_mean, model.cov_inv)
    score_map = d.reshape(rows, cols)

    orig_h, orig_w = image_rgb.shape[:2]
    mask = (score_map >= model.threshold).astype(np.uint8)
    boxes: list[BBox] = []
    if mask.any():
        if cv2 is not None:
            n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for i in range(1, n):
                x, y, bw, bh, area = stats[i]
                # Map grid-cell coords back to original image pixels.
                x1 = x / cols * orig_w
                y1 = y / rows * orig_h
                x2 = (x + bw) / cols * orig_w
                y2 = (y + bh) / rows * orig_h
                region_score = float(score_map[y:y + bh, x:x + bw].max())
                norm = float(np.clip(region_score / (model.threshold * 2), 0, 1))
                boxes.append(BBox(x1, y1, x2, y2, 0, "anomaly", norm))
        else:
            ys, xs = np.where(mask)
            x1, y1 = xs.min() / cols * orig_w, ys.min() / rows * orig_h
            x2, y2 = (xs.max() + 1) / cols * orig_w, (ys.max() + 1) / rows * orig_h
            boxes.append(BBox(x1, y1, x2, y2, 0, "anomaly", 1.0))
    boxes.sort(key=lambda b: b.score, reverse=True)
    return AnomalyResult(score_map, (rows, cols), boxes, float(score_map.max()))


def anomaly_heatmap(image_rgb: np.ndarray, result: AnomalyResult,
                    model: AnomalyModel, alpha: float = 0.5) -> np.ndarray:
    """Overlay the anomaly score map on the image as a heatmap (needs cv2)."""
    cv2 = optional_import("cv2")
    if cv2 is None:
        return image_rgb
    h, w = image_rgb.shape[:2]
    norm = np.clip(result.score_map / (model.threshold * 2), 0, 1)
    heat = (norm * 255).astype(np.uint8)
    heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_CUBIC)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_rgb, 1 - alpha, heat_rgb, alpha, 0)
