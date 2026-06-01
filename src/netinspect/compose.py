"""Composite realistic synthetic damage onto REAL net frames.

Why
---
We have real net footage (SOLAQUA) but no damage and no labels, and pure
synthetic net images are too easy (trivially separable). The pragmatic middle
ground used by a lot of industrial CV when real defects are scarce is
**cut-and-composite augmentation**: keep the *real* background (true net texture,
biofouling, lighting, turbidity) and paste *plausible* damage onto it with exact
labels. This yields a far more realistic, fully-labelled dataset for training and
for *comparing* methods quantitatively.

Honesty
-------
This is **synthetic damage on real backgrounds**. It is a strong proxy and the
right scaffold, but it is **not** a substitute for real labelled damage:
* the damage *appearance* is modelled, not observed, so a detector can learn our
  compositing artefacts rather than true damage cues;
* metrics on it estimate behaviour on *this kind* of injected damage only.
Treat resulting numbers as "promising on realistic proxy data, pending
validation on real damage." The same pipeline trains on real labels unchanged.

What damage looks like here
---------------------------
* **Hole**: an irregular opening where the mesh is gone — you see through to
  darker open water, with frayed bright fibre ends around the rim.
* **Tear**: an elongated slit, dark, with frayed edges along its length.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .data import Polygon
from .utils import BBox, ensure_dir, optional_import, write_image


@dataclass
class ComposeConfig:
    hole_frac_range: tuple[float, float] = (0.04, 0.10)   # radius as frac of width
    tear_len_range: tuple[float, float] = (0.10, 0.22)
    tear_width_range: tuple[float, float] = (0.012, 0.03)
    darkness: float = 0.35          # multiply local colour toward dark water
    fray: bool = True               # draw torn bright fibre ends
    single_class: bool = True       # label everything as class 0 "damage"
    seamless: bool = True           # Poisson-blend the see-through fill into the net
    radial_depth: bool = True       # darker toward the centre (water depth cue)
    hard_negatives: tuple[int, int] = (0, 3)  # dark-but-textured distractors per image (UNLABELLED)


def _bright_fibre_color(frame: np.ndarray, cx: int, cy: int, r: int) -> tuple[int, int, int]:
    """Sample a bright (net fibre / biofouling) colour near a point."""
    h, w = frame.shape[:2]
    y0, y1 = max(0, cy - r), min(h, cy + r)
    x0, x1 = max(0, cx - r), min(w, cx + r)
    patch = frame[y0:y1, x0:x1].reshape(-1, 3)
    if patch.size == 0:
        return (200, 210, 170)
    lum = patch.mean(axis=1)
    bright = patch[lum >= np.percentile(lum, 85)]
    c = bright.mean(axis=0) if len(bright) else patch.mean(axis=0)
    return tuple(int(v) for v in c)


def _see_through_color(frame: np.ndarray, cx: int, cy: int, r: int,
                       darkness: float) -> np.ndarray:
    """Darker 'open water behind the net' colour, from the local dark tones."""
    h, w = frame.shape[:2]
    y0, y1 = max(0, cy - r), min(h, cy + r)
    x0, x1 = max(0, cx - r), min(w, cx + r)
    patch = frame[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
    if patch.size == 0:
        return np.array([15, 35, 40], np.float32)
    lum = patch.mean(axis=1)
    dark = patch[lum <= np.percentile(lum, 25)]
    base = dark.mean(axis=0) if len(dark) else patch.mean(axis=0)
    return np.clip(base * darkness, 0, 255)


def _irregular_polygon(cx: int, cy: int, rx: int, ry: int,
                       rng: np.random.Generator, n: int = 14) -> np.ndarray:
    angles = np.sort(rng.uniform(0, 2 * np.pi, n))
    radii = 1.0 + rng.normal(0, 0.18, n)
    radii = np.clip(radii, 0.6, 1.4)
    xs = cx + (rx * radii * np.cos(angles))
    ys = cy + (ry * radii * np.sin(angles))
    return np.stack([xs, ys], axis=1).astype(np.int32)


def _draw_fray(frame: np.ndarray, poly: np.ndarray, color, rng: np.random.Generator,
               n: int = 30) -> None:
    cv2 = optional_import("cv2")
    if cv2 is None:
        return
    for _ in range(n):
        i = int(rng.integers(0, len(poly)))
        x, y = poly[i]
        ang = rng.uniform(0, 2 * np.pi)
        length = int(rng.integers(2, 7))
        x2 = int(np.clip(x + length * np.cos(ang), 0, frame.shape[1] - 1))
        y2 = int(np.clip(y + length * np.sin(ang), 0, frame.shape[0] - 1))
        cv2.line(frame, (int(x), int(y)), (x2, y2), color, 1, cv2.LINE_AA)


def inject_hole(frame: np.ndarray, rng: np.random.Generator,
                cfg: ComposeConfig) -> tuple[BBox, Polygon]:
    cv2 = optional_import("cv2")
    h, w = frame.shape[:2]
    rx = int(rng.uniform(*cfg.hole_frac_range) * w)
    ry = int(rx * rng.uniform(0.7, 1.3))
    cx = int(rng.integers(rx + 1, w - rx - 1))
    cy = int(rng.integers(ry + 1, h - ry - 1))
    poly = _irregular_polygon(cx, cy, rx, ry, rng)

    color = _see_through_color(frame, cx, cy, max(rx, ry), cfg.darkness)
    if cv2 is not None:
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        ys, xs = np.where(mask > 0)
        fill = np.empty_like(frame)
        fill[:] = color.astype(np.uint8)
        if cfg.radial_depth:
            # Darker toward the centre (looking deeper into open water).
            dist = np.sqrt(((xs - cx) / max(rx, 1)) ** 2 + ((ys - cy) / max(ry, 1)) ** 2)
            depth = (0.55 + 0.45 * np.clip(dist, 0, 1))[:, None]
            fill[ys, xs] = np.clip(color[None, :] * depth, 0, 255).astype(np.uint8)
        if cfg.seamless:
            # Poisson blend the dark fill into the net so the rim transitions
            # naturally instead of a hard cut (harder for a model to 'cheat' on).
            try:
                center = (int(np.clip(cx, rx + 2, w - rx - 2)), int(np.clip(cy, ry + 2, h - ry - 2)))
                frame[:] = cv2.seamlessClone(fill, frame, mask, center, cv2.NORMAL_CLONE)
            except cv2.error:
                frame[ys, xs] = fill[ys, xs]
        else:
            frame[ys, xs] = fill[ys, xs]
        frame[ys, xs] = cv2.GaussianBlur(frame, (0, 0), sigmaX=1.2)[ys, xs]
        if cfg.fray:
            _draw_fray(frame, poly, _bright_fibre_color(frame, cx, cy, max(rx, ry)), rng)
    xs2, ys2 = poly[:, 0], poly[:, 1]
    box = BBox(float(xs2.min()), float(ys2.min()), float(xs2.max()), float(ys2.max()),
               0 if cfg.single_class else 1, "damage" if cfg.single_class else "hole", 1.0)
    polygon = Polygon([(float(x), float(y)) for x, y in poly], box.class_id, box.class_name)
    return box, polygon


def inject_tear(frame: np.ndarray, rng: np.random.Generator,
                cfg: ComposeConfig) -> tuple[BBox, Polygon]:
    cv2 = optional_import("cv2")
    h, w = frame.shape[:2]
    length = int(rng.uniform(*cfg.tear_len_range) * w)
    width = int(rng.uniform(*cfg.tear_width_range) * w)
    cx = int(rng.integers(length, max(length + 1, w - length)))
    cy = int(rng.integers(length, max(length + 1, h - length)))
    angle = rng.uniform(0, np.pi)

    # Tapered slit polygon (wider middle, pointed ends).
    t = np.linspace(-1, 1, 10)
    half_w = (width / 2) * (1 - t ** 2) + 1
    along = (length / 2) * t
    pts = []
    ca, sa = np.cos(angle), np.sin(angle)
    for a, hw in zip(along, half_w):
        pts.append((a, hw))
    for a, hw in zip(along[::-1], half_w[::-1]):
        pts.append((a, -hw))
    pts = np.array(pts)
    rot = np.array([[ca, -sa], [sa, ca]])
    poly = (pts @ rot.T) + [cx, cy]
    poly = poly.astype(np.int32)
    poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)

    color = _see_through_color(frame, cx, cy, length // 2, cfg.darkness * 0.9)
    if cv2 is not None:
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        ys, xs = np.where(mask > 0)
        frame[ys, xs] = color.astype(np.uint8)
        if cfg.fray:
            _draw_fray(frame, poly, _bright_fibre_color(frame, cx, cy, length // 2), rng, n=40)
    xs2, ys2 = poly[:, 0], poly[:, 1]
    box = BBox(float(xs2.min()), float(ys2.min()), float(xs2.max()), float(ys2.max()),
               0 if cfg.single_class else 2, "damage" if cfg.single_class else "tear", 1.0)
    polygon = Polygon([(float(x), float(y)) for x, y in poly], box.class_id, box.class_name)
    return box, polygon


def add_hard_negatives(frame: np.ndarray, rng: np.random.Generator, n: int) -> None:
    """Add dark-but-TEXTURED distractors (shadow / biofouling) — NOT labelled.

    These resemble damage in being dark, but unlike a real opening they keep the
    net texture (they darken existing pixels rather than replacing them with flat
    see-through water, and have no frayed fibre rim). Training with these forces
    a detector to use *damage* cues — uniform see-through fill + frayed edges —
    instead of simply "this region is dark", which is the cheapest way to cheat.
    """
    cv2 = optional_import("cv2")
    if cv2 is None:
        return
    h, w = frame.shape[:2]
    for _ in range(n):
        rx = int(rng.uniform(0.05, 0.12) * w)
        ry = int(rx * rng.uniform(0.6, 1.4))
        cx = int(rng.integers(rx + 1, w - rx - 1))
        cy = int(rng.integers(ry + 1, h - ry - 1))
        poly = _irregular_polygon(cx, cy, rx, ry, rng)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=rx * 0.3)
        m = (mask.astype(np.float32) / 255.0)[..., None]
        darken = rng.uniform(0.35, 0.6)          # shadow/fouling darkening factor
        frame[:] = np.clip(frame * (1 - m * (1 - darken)), 0, 255).astype(np.uint8)


def composite_damage(frame: np.ndarray, rng: np.random.Generator, num: int,
                     cfg: ComposeConfig | None = None) -> tuple[np.ndarray, list[BBox], list[Polygon]]:
    """Return a damaged copy of ``frame`` plus boxes and polygons."""
    cfg = cfg or ComposeConfig()
    out = frame.copy()
    boxes, polys = [], []
    for _ in range(num):
        if rng.random() < 0.5:
            b, p = inject_hole(out, rng, cfg)
        else:
            b, p = inject_tear(out, rng, cfg)
        boxes.append(b)
        polys.append(p)
    return out, boxes, polys


def _yolo_box_line(b: BBox, w: int, h: int) -> str:
    xc = (b.x1 + b.x2) / 2 / w
    yc = (b.y1 + b.y2) / 2 / h
    return f"{b.class_id} {xc:.6f} {yc:.6f} {b.width / w:.6f} {b.height / h:.6f}"


def _yolo_seg_line(p: Polygon, w: int, h: int) -> str:
    coords = " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in p.points)
    return f"{p.class_id} {coords}"


def build_dataset(
    real_frames: list[Path],
    out_dir: str | Path,
    splits: tuple[float, float, float] = (0.7, 0.15, 0.15),
    damaged_fraction: float = 0.85,
    max_damage_per_image: int = 3,
    seg: bool = False,
    seed: int = 0,
    cfg: ComposeConfig | None = None,
) -> dict:
    """Composite damage onto real frames and write a YOLO dataset (train/val/test).

    Frames are split *before* compositing so train/val/test use disjoint real
    backgrounds (no background leakage). Some frames are left undamaged (no
    label file) so the negative class is represented.
    """
    from .utils import read_image
    cfg = cfg or ComposeConfig()
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)
    frames = sorted(real_frames)
    n = len(frames)
    n_train = int(n * splits[0])
    n_val = int(n * splits[1])
    assignment = (["train"] * n_train + ["val"] * n_val +
                  ["test"] * (n - n_train - n_val))
    rng.shuffle(assignment)

    counts = {s: {"images": 0, "damaged": 0, "instances": 0} for s in ("train", "val", "test")}
    for split in ("train", "val", "test"):
        ensure_dir(out_dir / "images" / split)
        ensure_dir(out_dir / "labels" / split)

    for frame_path, split in zip(frames, assignment):
        img = read_image(frame_path)
        h, w = img.shape[:2]
        # Add UNLABELLED hard negatives first (dark distractors the model must
        # learn to ignore), then composite the (labelled) damage on top.
        lo, hi = cfg.hard_negatives
        if hi > 0:
            add_hard_negatives(img, rng, int(rng.integers(lo, hi + 1)))
        damaged = rng.random() < damaged_fraction
        num = int(rng.integers(1, max_damage_per_image + 1)) if damaged else 0
        out_img, boxes, polys = composite_damage(img, rng, num, cfg)
        name = frame_path.stem
        write_image(out_dir / "images" / split / f"{name}.jpg", out_img)
        if boxes:
            if seg:
                lines = [_yolo_seg_line(p, w, h) for p in polys]
            else:
                lines = [_yolo_box_line(b, w, h) for b in boxes]
            (out_dir / "labels" / split / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
        counts[split]["images"] += 1
        counts[split]["damaged"] += int(bool(boxes))
        counts[split]["instances"] += len(boxes)

    names = ({0: "damage"} if cfg.single_class
             else {0: "damage", 1: "hole", 2: "tear", 3: "abnormal_region"})
    names_yaml = "\n".join(f"  {k}: {v}" for k, v in names.items())
    (out_dir / "dataset.yaml").write_text(
        f"# Synthetic damage composited on REAL SOLAQUA net frames.\n"
        f"# NOT real damage — see src/netinspect/compose.py docstring.\n"
        f"path: {out_dir.resolve().as_posix()}\n"
        f"train: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n{names_yaml}\n", encoding="utf-8")
    (out_dir / "PROVENANCE.txt").write_text(
        "Backgrounds: REAL SOLAQUA frames (SINTEF, CC BY-SA 4.0).\n"
        "Damage: SYNTHETIC, composited by netinspect.compose. NOT real damage.\n"
        "Use for training/comparison scaffolding; validate on real damage before "
        "any reliability claim.\n", encoding="utf-8")
    return {"out_dir": str(out_dir), "num_frames": n, "splits": counts,
            "seg": seg, "single_class": cfg.single_class}
