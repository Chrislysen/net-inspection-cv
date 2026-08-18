"""Animate what temporal confirmation actually does, side by side.

The single most useful thing this system does to a live feed is refuse to alert
on flicker, and that is invisible in a still image — a frame with a box looks
identical whether the box survives the next frame or not. So: the same clip
twice, raw detections on the left, temporally confirmed on the right, running
together.

On SOLAQUA the net is undamaged, so every box on the left is a false alarm and
the right-hand panel going quiet is the system working. That is the honest
demonstration and also the impressive one.

    python scripts/make_temporal_gif.py --source data/processed/solaqua_bag3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _common  # noqa: F401
import numpy as np

from netinspect.classical_baseline import ClassicalConfig
from netinspect.inference import NetInspector
from netinspect.temporal import TemporalConfig, Tracker
from netinspect.utils import ensure_dir, get_logger, list_images, read_image

LOGGER = get_logger()

RAW = (235, 104, 52)          # alarm orange
CONFIRMED = (42, 120, 214)    # measured blue
INK = (28, 30, 32)
PANEL_BG = (250, 250, 249)


def _draw(img: np.ndarray, boxes, colour, title, subtitle, scale):
    from PIL import Image, ImageDraw

    pil = Image.fromarray(img).convert("RGB")
    w = int(pil.width * scale)
    h = int(pil.height * scale)
    pil = pil.resize((w, h), Image.BILINEAR)
    dr = ImageDraw.Draw(pil, "RGBA")
    for b in boxes:
        dr.rectangle([b.x1 * scale, b.y1 * scale, b.x2 * scale, b.y2 * scale],
                     outline=colour + (255,), width=3)
    # Header strip, so each panel says what it is without a caption elsewhere.
    dr.rectangle([0, 0, w, 34], fill=(255, 255, 255, 225))
    dr.text((9, 5), title, fill=colour + (255,))
    dr.text((9, 19), subtitle, fill=(90, 92, 95, 255))
    return pil


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="data/processed/solaqua_bag3",
                    help="a directory of CONSECUTIVE frames")
    ap.add_argument("--weights", default="models/yolo_damage_v1.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--max-age", type=int, default=5)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--start", type=int, default=40)
    ap.add_argument("--scale", type=float, default=0.42)
    ap.add_argument("--ms", type=int, default=110)
    ap.add_argument("--colors", type=int, default=96,
                    help="GIF palette size; underwater footage is photographic, "
                         "so this is the main lever on file size")
    ap.add_argument("--out", default="docs/images/temporal_confirmation.gif")
    args = ap.parse_args()

    paths = list_images(Path(args.source))[args.start:args.start + args.frames]
    if len(paths) < 5:
        raise SystemExit(f"Need at least 5 frames in {args.source}")
    LOGGER.info("Running %s over %d consecutive frames…", args.weights, len(paths))

    insp = NetInspector(classical_cfg=ClassicalConfig(), yolo_weights=args.weights)
    tracker = Tracker(TemporalConfig(min_hits=args.min_hits, max_age=args.max_age))

    from PIL import Image

    frames, raw_total, confirmed_total = [], 0, 0
    for i, p in enumerate(paths, 1):
        img = read_image(p)
        raw = insp.predict(img, method="yolo", conf=args.conf).boxes
        confirmed = tracker.update(raw)
        raw_total += len(raw)
        confirmed_total += len(confirmed)

        left = _draw(img, raw, RAW, f"RAW  ·  {len(raw)} detection(s)",
                     "every frame scored independently", args.scale)
        right = _draw(img, confirmed, CONFIRMED, f"CONFIRMED  ·  {len(confirmed)}",
                      f"must persist {args.min_hits} frames", args.scale)

        gap = 10
        w, h = left.width * 2 + gap, left.height + 26
        sheet = Image.new("RGB", (w, h), PANEL_BG)
        sheet.paste(left, (0, 0))
        sheet.paste(right, (left.width + gap, 0))
        from PIL import ImageDraw
        dr = ImageDraw.Draw(sheet)
        dr.text((9, left.height + 7),
                f"frame {i}/{len(paths)}  ·  undamaged net, so every RAW box is a "
                f"false alarm  ·  running totals: {raw_total} raw -> {confirmed_total} confirmed",
                fill=INK)
        frames.append(sheet.convert("P", palette=Image.ADAPTIVE, colors=args.colors))

    out = Path(args.out)
    ensure_dir(out.parent)
    frames[0].save(out, save_all=True, append_images=frames[1:], loop=0,
                   duration=args.ms, optimize=True, disposal=2)
    kb = out.stat().st_size / 1024
    print(f"wrote {out}  ({len(frames)} frames, {kb:.0f} KB)")
    print(f"raw detections {raw_total} -> confirmed {confirmed_total}"
          f"  ({100 * (1 - confirmed_total / max(1, raw_total)):.0f}% suppressed)")


if __name__ == "__main__":
    main()
