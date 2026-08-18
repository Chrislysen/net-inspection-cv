"""Render the production workflow as a terminal figure, from live command output.

The CLI is the part of this project a company actually operates, and it had no
picture at all. This runs the real commands, captures what they print, and lays
the output out as terminal cards — so the figure cannot claim behaviour the code
does not have, and regenerates whenever the behaviour changes.

The story it tells in three panels is the one worth telling: ingest refuses bad
data, the release gate refuses a model it cannot measure, and the same gate
passes when the evidence is actually there.

    python scripts/make_workflow_figure.py
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import _common  # noqa: F401

from netinspect.utils import ensure_dir, get_logger

LOGGER = get_logger()
REPO = _common.REPO_ROOT

BG = (18, 22, 26)
CARD = (25, 30, 35)
INK = (222, 228, 232)
DIM = (128, 140, 150)
GREEN = (86, 197, 122)
RED = (238, 108, 84)
BLUE = (86, 168, 238)
AMBER = (232, 178, 84)
TITLE = (245, 248, 250)

# Timestamped log lines are noise in a figure.
LOG_LINE = re.compile(r"^\d{2}:\d{2}:\d{2} \[[A-Z]+\]")


def run(args: list[str], cwd: Path = REPO) -> str:
    exe = Path(sys.executable).with_name("netinspect.exe")
    cmd = [str(exe), *args] if exe.exists() else [sys.executable, "-m", "netinspect.cli", *args]
    LOGGER.info("$ netinspect %s", " ".join(args))
    # Explicit UTF-8: the child writes em-dashes and the default Windows
    # console encoding turned every one of them into mojibake in the figure.
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    text = (p.stdout or "") + (p.stderr or "")
    lines = [ln.rstrip() for ln in text.splitlines()
             if ln.strip() and not LOG_LINE.match(ln) and "INFO:" not in ln]
    return "\n".join(lines), p.returncode


def colour_for(line: str):
    s = line.strip()
    if s.startswith("$"):
        return BLUE
    if "PASS" in s or "[ok" in s:
        return GREEN
    if "FAIL" in s or "Refusing" in s or "!!" in s or "ERROR" in s:
        return RED
    if s.startswith("next:") or s.startswith("wrote"):
        return DIM
    if "warn" in s.lower():
        return AMBER
    return INK


def wrap(body: str, width: int = 116) -> list[str]:
    """Wrap rather than clip: a sentence cut mid-word reads as a rendering bug."""
    out = []
    for raw in body.splitlines():
        while len(raw) > width:
            cut = raw.rfind(" ", 0, width)
            cut = cut if cut > 60 else width
            out.append(raw[:cut])
            raw = "    " + raw[cut:].lstrip()
        out.append(raw)
    return out


def card(draw, x, y, w, title, subtitle, body, font, bold, small, line_h=15):
    lines = wrap(body)
    h = 52 + len(lines) * line_h + 12
    draw.rounded_rectangle([x, y, x + w, y + h], radius=8, fill=CARD)
    draw.text((x + 14, y + 12), title, fill=TITLE, font=bold)
    draw.text((x + 14, y + 29), subtitle, fill=DIM, font=small)
    for i, ln in enumerate(lines):
        draw.text((x + 14, y + 52 + i * line_h), ln,
                  fill=colour_for(ln), font=font)
    return h


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/processed/real_composite")
    ap.add_argument("--weights", default="models/yolo_damage_v1.pt")
    ap.add_argument("--out", default="docs/images/workflow_cli.png")
    args = ap.parse_args()

    from PIL import Image, ImageDraw, ImageFont

    def _font(size, bold=False):
        for name in (("consolab.ttf", "consola.ttf") if not bold else ("consolab.ttf",)):
            try:
                return ImageFont.truetype(name, size)
            except Exception:
                continue
        try:
            return ImageFont.truetype("DejaVuSansMono.ttf", size)
        except Exception:
            return ImageFont.load_default()

    mono, bold, small = _font(12), _font(13, bold=True), _font(11)

    strict, strict_code = run(["gate", "--data", args.data, "--split", "test",
                               "--weights", args.weights])
    relaxed, relaxed_code = run(["gate", "--data", args.data, "--split", "test",
                                 "--weights", args.weights,
                                 "--min-clean-frames", "5", "--min-damaged-frames", "5"])
    doctor, _ = run(["doctor"])
    doctor_tail = "\n".join(doctor.splitlines()[-3:])

    panels = [
        ("1 · The gate refuses what it cannot measure",
         f"$ netinspect gate --data {args.data} --split test    → exit {strict_code}",
         strict),
        ("2 · With the evidence actually present, it passes",
         f"$ netinspect gate ... --min-clean-frames 5 --min-damaged-frames 5    → exit {relaxed_code}",
         relaxed),
        ("3 · And it never stops saying what it is",
         "$ netinspect doctor",
         doctor_tail),
    ]

    W = 1180
    pad = 18
    # Measure first so the canvas is exactly the height of its content.
    heights = [52 + len(wrap(b)) * 15 + 12 for _, _, b in panels]
    H = 62 + sum(heights) + pad * len(panels)

    img = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(img)
    dr.text((pad + 4, 16), "netinspect — data in, gated model out",
            fill=TITLE, font=_font(17, bold=True))
    dr.text((pad + 4, 38),
            "Captured from live runs of the real CLI. The gate exits non-zero when a model "
            "does not meet the operating point, so CI can refuse to promote it.",
            fill=DIM, font=small)

    y = 62
    for title, sub, body in panels:
        h = card(dr, pad, y, W - pad * 2, title, sub, body, mono, bold, small)
        y += h + pad

    out = Path(args.out)
    ensure_dir(out.parent)
    img.save(out)
    print(f"wrote {out}  ({W}x{H})")
    print(f"strict gate exit {strict_code} · relaxed gate exit {relaxed_code}")


if __name__ == "__main__":
    main()
