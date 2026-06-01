"""Classical (non-ML) baseline for flagging suspicious regions in net imagery.

Rationale
---------
A fish-farm net is, visually, a *regular mesh texture*. Intact net produces a
dense, fairly uniform pattern of edges. Damage breaks that regularity in two
characteristic ways:

1. **Holes / openings** appear as locally *dark, low-texture* blobs — you see
   through the net to open water or background, so the mesh edges disappear.
2. **Tears / deformation** disrupt the local edge density and orientation.

This baseline combines two cheap, explainable cues:

* a **darkness cue** (adaptive threshold on the enhanced luminance), and
* a **low-edge-density cue** (regions where Canny edge density is well below the
  image median — i.e. the mesh pattern is locally missing).

Candidate regions are cleaned with morphology, extracted as contours, filtered
by area and shape, and scored by a transparent heuristic.

This is a *sanity-check baseline*, not a production detector. It has no learned
notion of "damage"; it flags visual anomalies that a human (or an ML model)
should review. Expect false positives on shadows, ropes, fish, and biofouling.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .preprocess import PreprocessConfig, apply_clahe, preprocess
from .utils import BBox, optional_import


@dataclass
class ClassicalConfig:
    """Tunable parameters for the classical baseline (see configs/baseline.yaml)."""
    resize: int = 768
    clahe_clip: float = 3.0
    canny_low: int = 40
    canny_high: int = 120
    edge_density_kernel: int = 31      # window for local edge-density estimate
    # A region is "mesh missing" where local edge density falls below this
    # fraction of the image-wide median density. Relative-to-median (not a fixed
    # percentile) so a uniformly textured net flags little.
    low_density_ratio: float = 0.35
    # A region is "dark" where luminance falls this many std-devs below the mean
    # on the white-balanced (pre-CLAHE) image — isolates genuine see-through dark.
    dark_std_factor: float = 1.8
    # How to combine the darkness and low-density cues:
    #   "or"  — dark OR mesh-missing  (sensitive; the per-region texture gate below
    #           then prunes intact-but-dark net cells)
    #   "and" — dark AND mesh-missing (stricter; can under-detect on high-contrast
    #           real frames where few pixels pass the absolute-darkness test)
    combine_mode: str = "or"
    morph_kernel: int = 5
    min_area_frac: float = 0.0006      # min region area as fraction of image
    max_area_frac: float = 0.25        # ignore huge regions (likely background)
    min_solidity: float = 0.35         # contour area / convex hull area
    # Texture gate (the main false-positive reducer on real net): a real opening
    # lacks mesh, so its internal edge density is well below the image median; an
    # intact-but-dark net patch still has its fibre grid (density ~= median).
    # Reject candidates whose internal edge density exceeds this fraction of the
    # median. Measured: holes ~0.64x median, net ~0.99x median.
    max_region_density_ratio: float = 0.82
    # FFT periodicity gate (disabled by default; kept for experimentation — it did
    # NOT separate holes from net on real data, so 0 = off).
    max_periodicity: float = 0.0
    score_threshold: float = 0.30      # drop low-confidence candidates


def _require_cv2():
    return optional_import("cv2")


def _region_periodicity(region_gray: np.ndarray) -> float:
    """Peak-to-mean ratio of the region's 2-D spectrum, excluding DC.

    Intact net is a regular mesh -> a strong off-centre spectral peak -> high
    ratio. A genuine opening (flat dark water) -> diffuse spectrum -> low ratio.
    Returns 0 for regions too small to analyse.
    """
    if region_gray.size < 64 or min(region_gray.shape) < 8:
        return 0.0
    r = region_gray.astype(np.float32)
    r = r - r.mean()
    win = np.hanning(r.shape[0])[:, None] * np.hanning(r.shape[1])[None, :]
    mag = np.abs(np.fft.fftshift(np.fft.fft2(r * win)))
    cy, cx = mag.shape[0] // 2, mag.shape[1] // 2
    # Zero out a small DC neighbourhood so the (always-large) DC term is ignored.
    mag[max(0, cy - 1):cy + 2, max(0, cx - 1):cx + 2] = 0.0
    mean = float(mag.mean())
    return float(mag.max() / mean) if mean > 1e-6 else 0.0


def _edge_density_map(edges: np.ndarray, kernel: int) -> np.ndarray:
    """Local fraction of edge pixels in a sliding window (0..1)."""
    cv2 = _require_cv2()
    k = kernel if kernel % 2 == 1 else kernel + 1
    e = (edges > 0).astype(np.float32)
    if cv2 is not None:
        return cv2.boxFilter(e, ddepth=-1, ksize=(k, k), normalize=True)
    # NumPy fallback: integral-image box mean.
    integral = e.cumsum(0).cumsum(1)
    integral = np.pad(integral, ((1, 0), (1, 0)))
    h, w = e.shape
    r = k // 2
    out = np.zeros_like(e)
    for y in range(h):
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        for x in range(w):
            x0, x1 = max(0, x - r), min(w, x + r + 1)
            total = (integral[y1, x1] - integral[y0, x1]
                     - integral[y1, x0] + integral[y0, x0])
            out[y, x] = total / ((y1 - y0) * (x1 - x0))
    return out


@dataclass
class ClassicalResult:
    boxes: list[BBox]
    mask: np.ndarray              # uint8 candidate mask at processed resolution
    scale: float                  # processed_size / original_size
    debug: dict = field(default_factory=dict)


def detect(image_rgb: np.ndarray, cfg: ClassicalConfig | None = None) -> ClassicalResult:
    """Run the classical baseline on a single RGB image.

    Returns boxes in **original image coordinates**. Requires OpenCV.
    """
    cfg = cfg or ClassicalConfig()
    cv2 = _require_cv2()
    if cv2 is None:
        raise RuntimeError(
            "The classical baseline needs OpenCV. Install opencv-python-headless."
        )

    orig_h, orig_w = image_rgb.shape[:2]

    # 1. Enhance. Two views of the image are used deliberately:
    #    * white-balanced (no CLAHE) for the DARKNESS cue — CLAHE equalises local
    #      contrast and destroys the absolute darkness that distinguishes a
    #      see-through hole from the mid-toned mesh;
    #    * CLAHE-enhanced for the EDGE/texture cue — CLAHE sharpens the mesh
    #      pattern, which is exactly what we want to measure.
    wb = PreprocessConfig(resize=cfg.resize, clahe=False, denoise=False,
                          color_normalize=True)
    proc_wb = preprocess(image_rgb, wb)
    proc = apply_clahe(proc_wb, cfg.clahe_clip)
    ph, pw = proc.shape[:2]
    scale = pw / float(orig_w)

    gray_dark = cv2.cvtColor(proc_wb, cv2.COLOR_RGB2GRAY)
    gray = cv2.cvtColor(proc, cv2.COLOR_RGB2GRAY)

    # 2. Darkness cue: ABSOLUTE darkness (see-through holes are much darker than
    #    the mid-toned mesh). Threshold against global mean - k*std on the
    #    white-balanced (pre-CLAHE) luminance.
    mean, std = float(gray_dark.mean()), float(gray_dark.std())
    dark_thr = mean - cfg.dark_std_factor * std
    dark = (gray_dark < dark_thr).astype(np.uint8) * 255

    # 3. Low-edge-density cue: where the regular mesh pattern is locally missing.
    edges = cv2.Canny(gray, cfg.canny_low, cfg.canny_high)
    density = _edge_density_map(edges, cfg.edge_density_kernel)
    median_density = float(np.median(density[density > 0])) if np.any(density > 0) else 0.0
    thr = cfg.low_density_ratio * median_density
    low_density = (density <= thr).astype(np.uint8) * 255

    # 4. Combine the darkness and low-density cues.
    #    "and" (default) — a region must be BOTH dark AND missing mesh texture.
    #      On real net this is the key false-positive reducer: intact-but-dark
    #      cells are dark yet still textured, so they are excluded.
    #    "or" — more sensitive (e.g. very thin tears), at the cost of more false
    #      positives; the per-region gates below then prune textured regions.
    if cfg.combine_mode == "or":
        candidate = cv2.bitwise_or(dark, low_density)
    else:
        candidate = cv2.bitwise_and(dark, low_density)
    k_open = np.ones((3, 3), np.uint8)
    k_close = np.ones((cfg.morph_kernel, cfg.morph_kernel), np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, k_open, iterations=1)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, k_close, iterations=3)

    # 5. Extract and filter contours.
    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = float(pw * ph)
    boxes: list[BBox] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        area_frac = area / img_area
        if area_frac < cfg.min_area_frac or area_frac > cfg.max_area_frac:
            continue
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0
        if solidity < cfg.min_solidity:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        # Measure darkness/texture over the ACTUAL contour pixels, not the
        # bounding box. A diagonal tear's bbox is mostly mesh, so a bbox-based
        # density test would wrongly reject it.
        region_mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(region_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        sel = region_mask > 0
        region_density = float(density[sel].mean())
        region_mean = float(gray_dark[sel].mean())
        # Reject regions that still contain a lot of mesh edges (just textured
        # net) — UNLESS the region is clearly dark, in which case it is most
        # likely a thin tear whose own edges raise the local density.
        # Texture gate (always on): reject regions that still contain mesh — a
        # dark *net* patch has near-median internal edge density, whereas a real
        # opening is well below it. This is the main real-net FP reducer.
        if median_density > 0 and region_density > cfg.max_region_density_ratio * median_density:
            continue

        # Optional FFT periodicity gate (off by default; see config note).
        if cfg.max_periodicity > 0:
            if _region_periodicity(gray[y:y + h, x:x + w]) > cfg.max_periodicity:
                continue

        # Heuristic score: darker + larger (saturating) + emptier -> more suspicious.
        darkness = float(np.clip((mean - region_mean) / (std + 1e-6), 0, 2) / 2)
        size_term = min(1.0, area_frac / 0.05)  # saturate around 5% of the image
        emptiness = 1.0 - min(1.0, region_density / (median_density + 1e-6))
        score = float(np.clip(0.5 * darkness + 0.2 * size_term + 0.3 * emptiness, 0.0, 1.0))
        if score < cfg.score_threshold:
            continue

        # Map back to original coordinates.
        boxes.append(BBox(
            x1=x / scale, y1=y / scale,
            x2=(x + w) / scale, y2=(y + h) / scale,
            class_id=0, class_name="damage", score=score,
        ))

    boxes.sort(key=lambda b: b.score, reverse=True)
    return ClassicalResult(
        boxes=boxes, mask=candidate, scale=scale,
        debug={"edge_density_threshold": float(thr), "num_contours": len(contours)},
    )
