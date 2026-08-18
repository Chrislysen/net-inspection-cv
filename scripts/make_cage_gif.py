"""Build the rotating-cage animation for the README.

The 3-D viewer is the part of this project that is hardest to convey in a still
image: it is a thing you turn around. This renders a full orbit through the
*actual* renderer (``web/net3d.frames.mjs`` drives ``web/net3d.js`` against a
stubbed canvas) and rasterises the frames into a looping GIF, so the animation
in the docs is produced by the code it documents and cannot drift from it.

    # 1. a scene, from the running server or a saved map
    python scripts/make_cage_gif.py --scene reports/results/inspection_maps/<clip>_map.json

Requires node for the render step; everything after that is Pillow.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import _common  # noqa: F401

from netinspect.utils import ensure_dir, get_logger

LOGGER = get_logger()
REPO = _common.REPO_ROOT

SURFACE = (238, 244, 247)


def scene_from_map(map_json: Path, circumference=160.0, cyl=15.0, cone=10.0) -> dict:
    """Build a viewer scene from a saved inspection map, without the server."""
    from netinspect.netmodel import PenGeometry, build_scene

    data = json.loads(map_json.read_text(encoding="utf-8"))
    geom = PenGeometry(circumference_m=circumference, cylinder_depth_m=cyl,
                       cone_depth_m=cone)
    scene = build_scene(sites=data.get("sites", []), track=data.get("track", []),
                        coverage=data.get("coverage", {}), geom=geom)
    scene["clip"] = data.get("clip", map_json.stem)
    return scene


def render_frames(scene: dict, frames: int, workdir: Path) -> dict:
    scene_path = workdir / "scene.json"
    dump_path = workdir / "frames.json"
    scene_path.write_text(json.dumps(scene), encoding="utf-8")
    script = REPO / "web" / "net3d.frames.mjs"
    LOGGER.info("Rendering %d frames through the real viewer…", frames)
    proc = subprocess.run(
        ["node", str(script), str(scene_path), str(dump_path), str(frames)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"node failed:\n{proc.stdout}\n{proc.stderr}")
    LOGGER.info("%s", proc.stdout.strip())
    return json.loads(dump_path.read_text(encoding="utf-8"))


def _rgba(colour: str, alpha: float):
    if not colour or colour == "none":
        return None
    c = colour.lstrip("#")
    if len(c) != 6:
        return None
    return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16),
            int(255 * max(0.0, min(1.0, alpha))))


def rasterise(dump: dict, caption: str):
    from PIL import Image, ImageDraw

    w, h = dump["W"], dump["H"]
    out = []
    for rec in dump["frames"]:
        img = Image.new("RGB", (w, h), SURFACE)
        dr = ImageDraw.Draw(img, "RGBA")
        for q in rec["polys"]:
            pts = [(p["x"], p["y"]) for p in q["pts"]]
            if len(pts) > 2:
                fill = _rgba(q["fill"], q["a"])
                if fill:
                    dr.polygon(pts, fill=fill)
        for s in rec["segs"]:
            col = _rgba(s["s"], s["a"])
            if not col:
                continue
            dr.line([s["x1"], s["y1"], s["x2"], s["y2"]], fill=col,
                    width=max(1, int(s["w"])))
        for a in rec["arcs"]:
            col = _rgba(a["s"], 1.0)
            r = a["r"]
            if col:
                dr.ellipse([a["x"] - r, a["y"] - r, a["x"] + r, a["y"] + r],
                           outline=col, width=max(2, int(a["w"])))
        for t in rec["texts"]:
            col = _rgba(t["f"], 1.0)
            if not col:
                continue
            x = t["x"] - (len(t["t"]) * 3 if t.get("al") == "center" else 0)
            dr.text((x, t["y"] - 7), t["t"], fill=col)
        if caption:
            # Top, not bottom: the viewer draws its provenance legend in the
            # lower-left corner and the two collided.
            dr.rectangle([0, 0, w, 20], fill=(255, 255, 255, 215))
            dr.text((10, 6), caption, fill=(82, 81, 78, 255))
        out.append(img.convert("P", palette=Image.ADAPTIVE, colors=128))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=None,
                    help="a viewer scene JSON, or an inspection map to build one from")
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--ms", type=int, default=70, help="milliseconds per frame")
    ap.add_argument("--out", default="docs/images/cage_orbit.gif")
    ap.add_argument("--circumference", type=float, default=160.0)
    args = ap.parse_args()

    src = Path(args.scene) if args.scene else None
    if src is None:
        maps = sorted((REPO / "reports/results/inspection_maps").glob("*_map.json"))
        if not maps:
            raise SystemExit("No inspection map found — run scripts/map_inspection.py first.")
        src = maps[0]

    raw = json.loads(src.read_text(encoding="utf-8"))
    scene = raw if "pen" in raw else scene_from_map(src, circumference=args.circumference)

    cov = scene.get("coverage", {})
    caption = (f"{scene.get('clip', '')}  ·  {len(scene.get('sites', []))} sites  ·  "
               f"{cov.get('area_percent', '?')}% of a {scene['pen']['circumference_m']:.0f} m cage "
               f"inspected  ·  dashed = declared, solid = measured")

    with tempfile.TemporaryDirectory() as td:
        dump = render_frames(scene, args.frames, Path(td))
    images = rasterise(dump, caption)

    out = Path(args.out)
    ensure_dir(out.parent)
    images[0].save(out, save_all=True, append_images=images[1:], loop=0,
                   duration=args.ms, optimize=True, disposal=2)
    size_kb = out.stat().st_size / 1024
    print(f"wrote {out}  ({len(images)} frames, {size_kb:.0f} KB)")
    if size_kb > 4000:
        print("NOTE: over 4 MB — GitHub will still serve it, but consider "
              "--frames 32 or a smaller canvas.", file=sys.stderr)


if __name__ == "__main__":
    main()
