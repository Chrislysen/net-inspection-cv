"""Synthetic net-image generator for PIPELINE TESTING ONLY.

>>> WARNING <<<
These images are crude, procedurally generated approximations of a net mesh
with artificial "damage". They exist solely to exercise the code path end to
end (ingest -> preprocess -> detect -> evaluate -> visualise) when no real data
is available. **Results on this data say NOTHING about real-world aquaculture
performance.** Do not report any metric from this generator as evidence of
detection quality. Replace it with real inspection footage to get meaningful
numbers.

The generator produces:
* a bluish/greenish "underwater" background with a lighting gradient and noise,
* a regular diamond mesh (the intact net),
* injected damage regions (dark "holes" / elongated "tears") where the mesh is
  removed, plus matching YOLO-format ground-truth boxes.

Determinism: pass an integer ``seed`` for reproducible images (no wall-clock
randomness), so the demo is repeatable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .utils import BBox, ensure_dir, optional_import, write_image


def _underwater_background(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Bluish-green gradient with vignette and noise."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    grad = 1.0 - (yy / h) * 0.5  # brighter near the top
    base = np.zeros((h, w, 3), np.float32)
    base[..., 0] = 20 + 30 * grad   # R (low underwater)
    base[..., 1] = 70 + 60 * grad   # G
    base[..., 2] = 90 + 70 * grad   # B (dominant)
    # Radial vignette.
    cy, cx = h / 2, w / 2
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / np.sqrt(cy ** 2 + cx ** 2)
    base *= (1.0 - 0.35 * dist[..., None])
    base += rng.normal(0, 6, base.shape)
    return np.clip(base, 0, 255).astype(np.uint8)


def _draw_mesh(img: np.ndarray, spacing: int, rng: np.random.Generator) -> None:
    """Draw a diamond net mesh in place (lighter than the background)."""
    h, w = img.shape[:2]
    cv2 = optional_import("cv2")
    color = (180, 200, 190)
    thickness = 1
    offsets = range(-h, w + h, spacing)
    if cv2 is not None:
        for c in offsets:
            cv2.line(img, (c, 0), (c + h, h), color, thickness, cv2.LINE_AA)
            cv2.line(img, (c, 0), (c - h, h), color, thickness, cv2.LINE_AA)
    else:
        # NumPy fallback: mark pixels on the two diagonal families.
        yy, xx = np.mgrid[0:h, 0:w]
        for c in offsets:
            img[np.abs((xx - c) - yy) <= 0] = color
            img[np.abs((xx - c) + yy) <= 0] = color


def _clip(box: BBox, w: int, h: int) -> BBox:
    """Confine a ground-truth box to the frame it describes.

    A tear is a ROTATED rectangle whose length comes from the image width while
    its centre is bounded by the height, so on a wide, short frame its corners
    land outside the image — and nothing clipped them. The box was then written
    to a YOLO label as-is, giving normalised coordinates outside [0, 1] that
    `netinspect.data.parse_yolo_label` refuses to read back.

    Worth being precise about the blast radius: `generate_dataset` only ever
    calls this at its default 540x720, where no box escapes, so no committed
    dataset or reported metric is affected. But `make_synthetic_image` is public
    and takes height and width, and ground-truth geometry is the last place to
    leave a latent off-frame bug.

    Clipping rather than rejecting: the damage really was painted there, and the
    visible part of it is the correct label for the visible part of the frame.
    """
    x1 = min(max(box.x1, 0.0), float(w))
    y1 = min(max(box.y1, 0.0), float(h))
    x2 = min(max(box.x2, 0.0), float(w))
    y2 = min(max(box.y2, 0.0), float(h))
    return BBox(x1, y1, x2, y2, box.class_id, box.class_name, box.score)


def _inject_damage(img: np.ndarray, kind: str, rng: np.random.Generator) -> BBox:
    """Paint a dark damage region and return its ground-truth box.

    ``kind`` is "hole" (compact dark ellipse) or "tear" (elongated dark streak).
    """
    h, w = img.shape[:2]
    cv2 = optional_import("cv2")
    cx = int(rng.integers(int(w * 0.15), int(w * 0.85)))
    cy = int(rng.integers(int(h * 0.15), int(h * 0.85)))
    dark = (10, 25, 30)  # see-through-to-deep-water dark

    def _span(lo: int, hi: int, floor: int = 1) -> int:
        """A positive integer size, whatever the frame dimensions are.

        Every size here is a percentage of a dimension, and on a small frame two
        such percentages collapse onto the same integer — `rng.integers(3, 3)`
        raises "low >= high". Below 134 px wide that happened on roughly half of
        all seeds, so the generator simply crashed rather than producing a small
        image. Clamping keeps the sizes ordered and never returns a zero-radius
        (that is, zero-area) region.
        """
        lo = max(floor, lo)
        return int(rng.integers(lo, max(lo + 1, hi)))

    if kind == "hole":
        rx = _span(int(w * 0.04), int(w * 0.09))
        ry = _span(int(h * 0.04), int(h * 0.09))
        if cv2 is not None:
            cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, dark, -1)
        else:
            yy, xx = np.mgrid[0:h, 0:w]
            mask = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1
            img[mask] = dark
        return _clip(BBox(cx - rx, cy - ry, cx + rx, cy + ry, 1, "hole", 1.0), w, h)

    # tear: an elongated rotated rectangle.
    length = _span(int(w * 0.12), int(w * 0.22), floor=2)
    width = _span(int(w * 0.01), int(w * 0.03), floor=2)
    angle = float(rng.uniform(0, 180))
    if cv2 is not None:
        box = cv2.boxPoints(((cx, cy), (length, width), angle)).astype(np.int32)
        cv2.fillPoly(img, [box], dark)
        xs, ys = box[:, 0], box[:, 1]
        return _clip(BBox(float(xs.min()), float(ys.min()),
                          float(xs.max()), float(ys.max()), 2, "tear", 1.0), w, h)
    # NumPy fallback: axis-aligned streak.
    x1, x2 = cx - length // 2, cx + length // 2
    y1, y2 = cy - width // 2, cy + width // 2
    img[max(0, y1):y2, max(0, x1):x2] = dark
    return _clip(BBox(float(x1), float(y1), float(x2), float(y2), 2, "tear", 1.0), w, h)


def _add_distractors(img: np.ndarray, rng: np.random.Generator) -> None:
    """Add UNLABELLED confounders (soft shadows, biofouling-like patches).

    These are deliberately *not* damage and are *not* added to the ground truth.
    They exist so the baseline produces realistic false positives / negatives,
    which exercises the failure-case tooling and keeps the demo honest about how
    fragile a simple darkness/texture heuristic is.
    """
    h, w = img.shape[:2]
    cv2 = optional_import("cv2")

    # Soft shadow: a large, moderately dark, soft-edged blob (darker than mesh
    # but not as dark as a see-through hole). A naive darkness cue may flag it.
    if rng.random() < 0.7 and cv2 is not None:
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        rx, ry = int(rng.integers(int(w * 0.1), int(w * 0.22))), int(rng.integers(int(h * 0.1), int(h * 0.22)))
        shadow = img.astype(np.float32)
        mask = np.zeros((h, w), np.float32)
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=rx * 0.4)
        shadow *= (1.0 - 0.45 * mask[..., None])
        img[:] = np.clip(shadow, 0, 255).astype(np.uint8)

    # Biofouling-like speckle: clusters of small greenish blobs on the net.
    if rng.random() < 0.6 and cv2 is not None:
        n = int(rng.integers(20, 60))
        bx, by = int(rng.integers(0, w)), int(rng.integers(0, h))
        for _ in range(n):
            px = int(np.clip(bx + rng.normal(0, w * 0.04), 0, w - 1))
            py = int(np.clip(by + rng.normal(0, h * 0.04), 0, h - 1))
            r = int(rng.integers(2, 6))
            cv2.circle(img, (px, py), r, (90, 130, 70), -1)


def make_synthetic_image(
    height: int = 540,
    width: int = 720,
    num_damage: int = 2,
    seed: int = 0,
    mesh_spacing: int = 22,
    distractors: bool = True,
) -> tuple[np.ndarray, list[BBox]]:
    """Create one synthetic net image and its ground-truth damage boxes."""
    rng = np.random.default_rng(seed)
    img = _underwater_background(height, width, rng)
    _draw_mesh(img, mesh_spacing, rng)
    if distractors:
        _add_distractors(img, rng)
    boxes: list[BBox] = []
    for i in range(num_damage):
        kind = "hole" if rng.random() < 0.5 else "tear"
        boxes.append(_inject_damage(img, kind, rng))
    return img, boxes


def _box_to_yolo_line(box: BBox, w: int, h: int) -> str:
    xc = (box.x1 + box.x2) / 2 / w
    yc = (box.y1 + box.y2) / 2 / h
    bw = box.width / w
    bh = box.height / h
    return f"{box.class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def generate_dataset(
    out_dir: str | Path,
    n_images: int = 8,
    n_damaged: int | None = None,
    seed: int = 0,
) -> dict:
    """Generate a small synthetic dataset in YOLO layout (images/ + labels/).

    A fraction of images are left undamaged (no label file) so the evaluation
    code is exercised on both positive and negative images.
    """
    out_dir = Path(out_dir)
    img_dir = ensure_dir(out_dir / "images")
    lbl_dir = ensure_dir(out_dir / "labels")
    n_damaged = n_images * 3 // 4 if n_damaged is None else n_damaged

    rng = np.random.default_rng(seed)
    manifest = []
    for i in range(n_images):
        damaged = i < n_damaged
        num = int(rng.integers(1, 4)) if damaged else 0
        img, boxes = make_synthetic_image(num_damage=num, seed=seed + i + 1)
        h, w = img.shape[:2]
        name = f"synthetic_{i:03d}"
        write_image(img_dir / f"{name}.jpg", img)
        if boxes:
            lines = [_box_to_yolo_line(b, w, h) for b in boxes]
            (lbl_dir / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
        manifest.append({"image": f"{name}.jpg", "num_damage": len(boxes)})

    # A README marker so nobody mistakes this for real data.
    (out_dir / "SYNTHETIC_PLACEHOLDER.txt").write_text(
        "Procedurally generated synthetic net images for pipeline testing ONLY.\n"
        "These do NOT represent real aquaculture conditions and must not be used\n"
        "to claim real-world detection performance.\n",
        encoding="utf-8",
    )
    return {"out_dir": str(out_dir), "n_images": n_images,
            "n_damaged": n_damaged, "manifest": manifest}
