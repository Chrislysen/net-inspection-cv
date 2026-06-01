"""Extract frames from an inspection video into a directory of images.

Examples
--------
    python scripts/extract_frames.py --video data/raw/video.mp4 --out data/processed/frames
    python scripts/extract_frames.py --video clip.mov --out frames --fps 2 --max-frames 200
"""
from __future__ import annotations

import argparse

import _common  # noqa: F401  (path bootstrap)
from netinspect.video import extract_frames


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True, help="Path to .mp4/.mov/... video")
    ap.add_argument("--out", required=True, help="Output directory for frames")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--every-n", type=int, help="Keep one frame every N frames")
    group.add_argument("--fps", type=float, help="Approx. frames per second to sample")
    ap.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")
    args = ap.parse_args()

    saved = extract_frames(args.video, args.out, every_n=args.every_n,
                           target_fps=args.fps, max_frames=args.max_frames)
    print(f"Saved {len(saved)} frames to {args.out}")


if __name__ == "__main__":
    main()
