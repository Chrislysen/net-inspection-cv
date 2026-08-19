"""Image preprocessing for underwater / net imagery.

These operations are deliberately simple and explainable. They target the
recurring problems in underwater net footage: low contrast, colour casts (the
blue/green attenuation of water), and sensor/turbidity noise. Every step is
optional and configurable so the same code path serves both the classical
baseline and ML data preparation.

OpenCV is used when available for speed and for CLAHE; a NumPy fallback keeps
resize and a global-contrast approximation working without it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import optional_import


@dataclass
class PreprocessConfig:
    """Configuration for the preprocessing pipeline."""
    resize: int | None = None          # longest side in pixels, or None to keep
    clahe: bool = True                 # local contrast enhancement
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    denoise: bool = False              # bilateral / median noise reduction
    color_normalize: bool = False      # gray-world white balance


def resize_keep_aspect(image: np.ndarray, longest_side: int) -> np.ndarray:
    """Resize so the longest side equals ``longest_side``, keeping aspect ratio."""
    h, w = image.shape[:2]
    scale = longest_side / float(max(h, w))
    if scale == 1.0:
        return image
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    cv2 = optional_import("cv2")
    if cv2 is not None:
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        return cv2.resize(image, (new_w, new_h), interpolation=interp)
    # Nearest-neighbour NumPy fallback (adequate for the demo path).
    ys = (np.linspace(0, h - 1, new_h)).astype(int)
    xs = (np.linspace(0, w - 1, new_w)).astype(int)
    return image[ys][:, xs]


def apply_clahe(image: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalisation on the luminance channel."""
    cv2 = optional_import("cv2")
    if cv2 is None:
        # Fallback: global histogram stretch per channel (weaker but dependency-free).
        out = image.astype(np.float32)
        for c in range(out.shape[2]):
            ch = out[..., c]
            lo, hi = np.percentile(ch, 1), np.percentile(ch, 99)
            if hi > lo:
                out[..., c] = np.clip((ch - lo) * 255.0 / (hi - lo), 0, 255)
        return out.astype(np.uint8)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lch, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    lch = clahe.apply(lch)
    return cv2.cvtColor(cv2.merge((lch, a, b)), cv2.COLOR_LAB2RGB)


def denoise(image: np.ndarray) -> np.ndarray:
    """Edge-preserving noise reduction (bilateral filter, or median fallback)."""
    cv2 = optional_import("cv2")
    if cv2 is None:
        # Cheap 3x3 median via NumPy by stacking shifted views would be verbose;
        # fall back to a light box blur which is acceptable for the demo path.
        k = np.ones((3, 3), np.float32) / 9.0
        out = np.empty_like(image)
        for c in range(image.shape[2]):
            padded = np.pad(image[..., c], 1, mode="edge").astype(np.float32)
            acc = np.zeros_like(image[..., c], dtype=np.float32)
            for dy in range(3):
                for dx in range(3):
                    acc += padded[dy:dy + image.shape[0], dx:dx + image.shape[1]] * k[dy, dx]
            out[..., c] = acc.astype(np.uint8)
        return out
    return cv2.bilateralFilter(image, d=5, sigmaColor=50, sigmaSpace=50)


def compensate_red(image: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Restore the attenuated red channel using the better-preserved green one.

        I_rc = I_r + alpha * (mean(I_g) - mean(I_r)) * (1 - I_r) * I_g

    with channels in [0, 1]. From Ancuti, Ancuti, De Vleeschouwer & Bekaert,
    "Color Balance and Fusion for Underwater Image Enhancement", IEEE TIP 27(1),
    2018 — implemented from the published formula, not from any existing code.

    Why this rather than the usual gray-world: water absorbs red first, so by a
    few metres the red channel is not merely *scaled* down, it is nearly gone in
    the dark regions and unrecoverable by a global gain. Gray-world multiplies a
    channel that has little signal left and amplifies its noise. This instead
    borrows structure from green, which survives, and the ``(1 - I_r)`` term
    concentrates the correction where red is most depleted while leaving
    already-bright red pixels alone.

    NOTE for anyone wiring this into training data: :mod:`netinspect.compose`
    paints synthetic damage onto real frames. Apply this consistently on both
    sides or not at all — correcting the background but not the pasted damage
    teaches the detector a colour discontinuity that is an artifact of the
    pipeline rather than a property of damage.
    """
    img = image.astype(np.float32) / 255.0
    r, g = img[..., 0], img[..., 1]
    img[..., 0] = np.clip(r + alpha * (g.mean() - r.mean()) * (1.0 - r) * g, 0.0, 1.0)
    return (img * 255.0).astype(np.uint8)


def gray_world_white_balance(image: np.ndarray) -> np.ndarray:
    """Correct colour cast by equalising per-channel means (gray-world assumption).

    Underwater images are typically blue/green dominated; this is a standard,
    cheap correction that helps the classical baseline and visual inspection.
    """
    img = image.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0)
    gray = means.mean()
    # Guard the divide so a black/uniform frame doesn't warn or produce NaNs.
    scale = np.divide(gray, means, out=np.ones_like(means), where=means > 1e-6)
    return np.clip(img * scale, 0, 255).astype(np.uint8)


def preprocess(image: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Run the configured preprocessing pipeline and return an RGB uint8 image."""
    out = image
    if cfg.color_normalize:
        out = gray_world_white_balance(out)
    if cfg.resize:
        out = resize_keep_aspect(out, cfg.resize)
    if cfg.denoise:
        out = denoise(out)
    if cfg.clahe:
        out = apply_clahe(out, cfg.clahe_clip, cfg.clahe_grid)
    return out
