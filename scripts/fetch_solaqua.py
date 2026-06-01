"""Download SOLAQUA data and extract real net frames from ROS bags.

SOLAQUA (SINTEF Ocean Large Aquaculture Robotics Dataset, CC BY-SA 4.0) is real
ROV footage of **undamaged** net pens. See src/netinspect/solaqua.py.

Examples
--------
    # List available files (sizes)
    python scripts/fetch_solaqua.py --list

    # Download the smallest video bag and extract ~40 frames
    python scripts/fetch_solaqua.py --smallest-video --frames-out data/processed/solaqua_frames

    # Extract frames from a bag already on disk
    python scripts/fetch_solaqua.py --bag data/raw/solaqua/xxx_video.bag --frames-out data/processed/solaqua_frames
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _common  # noqa: F401

from netinspect import solaqua


def _progress(downloaded: int, total: int) -> None:
    if total:
        pct = 100 * downloaded / total
        sys.stdout.write(f"\r  downloading… {downloaded/1e6:8.1f} / {total/1e6:8.1f} MB ({pct:5.1f}%)")
        sys.stdout.flush()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="List files in the SOLAQUA feature")
    ap.add_argument("--smallest-video", action="store_true",
                    help="Download the smallest *_video.bag (contains camera frames)")
    ap.add_argument("--data-id", default=None, help="Download a specific data id")
    ap.add_argument("--bag", default=None, help="Use a .bag already on disk (skip download)")
    ap.add_argument("--out", default="data/raw/solaqua", help="Download directory")
    ap.add_argument("--frames-out", default=None, help="Extract camera frames to this dir")
    ap.add_argument("--sonar-out", default=None, help="Extract sonar frames to this dir")
    ap.add_argument("--every-n", type=int, default=30, help="Keep 1 frame per N messages")
    ap.add_argument("--max-frames", type=int, default=40)
    ap.add_argument("--topic", default=None, help="Image topic (default: most frames)")
    args = ap.parse_args()

    if args.list:
        files = solaqua.list_files()
        print(f"SOLAQUA feature has {len(files)} files (CC BY-SA 4.0):\n")
        for f in files:
            tag = "video" if f.is_video_bag else "data "
            print(f"  [{tag}] {f.size_mb:9.1f} MB  {f.file_name}  ({f.data_id})")
        return

    bag_path: Path | None = None
    if args.bag:
        bag_path = Path(args.bag)
        if not bag_path.exists():
            sys.exit(f"Bag not found: {bag_path}")
    else:
        target = None
        if args.smallest_video:
            target = solaqua.smallest_video_bag()
            if target is None:
                sys.exit("No video bag found in the feature.")
        elif args.data_id:
            target = next((f for f in solaqua.list_files() if f.data_id == args.data_id), None)
            if target is None:
                sys.exit(f"data id {args.data_id} not found.")
        else:
            ap.error("Specify one of --list, --smallest-video, --data-id, or --bag.")

        out_path = Path(args.out) / target.file_name
        print(f"Downloading {target.file_name} ({target.size_mb:.0f} MB)…")
        solaqua.download_file(target.data_id, out_path, progress=_progress)
        print()
        bag_path = out_path

    if args.frames_out and bag_path is not None:
        print(f"\nTopics in {bag_path.name}:")
        for t in solaqua.list_bag_topics(bag_path):
            print(f"  {t['topic']:45s} {t['msgtype']:35s} msgs={t['count']}")
        frames = solaqua.extract_bag_frames(
            bag_path, args.frames_out, topic=args.topic,
            every_n=args.every_n, max_frames=args.max_frames,
        )
        print(f"\nExtracted {len(frames)} camera frames to {args.frames_out}")
        print("\nNOTE: SOLAQUA nets are UNDAMAGED and UNLABELLED. Use these frames for "
              "preprocessing, false-positive analysis, and anomaly detection — not for "
              "measuring damage-detection accuracy.")

    if args.sonar_out and bag_path is not None:
        sonar = solaqua.extract_sonar_frames(
            bag_path, args.sonar_out, every_n=args.every_n, max_frames=args.max_frames,
        )
        print(f"\nExtracted {len(sonar)} multibeam-sonar frames to {args.sonar_out}")
        print("Sonar is a complementary modality (sees through turbidity); not RGB damage detection.")


if __name__ == "__main__":
    main()
