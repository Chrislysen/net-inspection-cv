"""Extract ROV telemetry from SOLAQUA ``*_data.bag`` files to parquet.

The sensor bags are 1.5–3.3 MB each (versus 0.9–2.3 GB for the paired video
bags), so the whole telemetry record for every clip fits in ~12 MB. See
``src/netinspect/telemetry.py`` for the canonical stream definitions and the
sensor-suite differences between the two recording days.

Examples
--------
    # What is in a bag?
    python scripts/extract_telemetry.py --topics data/raw/solaqua/2024-08-22_14-47-39_data.bag

    # Extract one clip
    python scripts/extract_telemetry.py --bag data/raw/solaqua/2024-08-22_14-47-39_data.bag

    # Extract every downloaded clip and write a cross-clip flight-profile table
    python scripts/extract_telemetry.py --all --profile-out reports/results/telemetry/flight_profiles.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _common  # noqa: F401

from netinspect import telemetry as T
from netinspect.utils import ensure_dir, write_json

DEFAULT_RAW = "data/raw/solaqua"
DEFAULT_OUT = "data/processed/telemetry"


def _flight_profile(clip: str, streams: dict) -> dict:
    """Summarise the flight profile of one clip.

    The commanded values come from the net-following manager (operator intent);
    the achieved values are the estimator's own output. Samples where the
    net-plane estimate was not locked are excluded from the achieved statistics
    because an unlocked estimate is not a measurement of anything.
    """
    import numpy as np

    plane = streams.get("net_plane")
    setp = streams.get("setpoint")
    depth = streams.get("depth_temp")
    if plane is None or plane.empty:
        return {"clip": clip, "error": "no net_plane stream"}

    locked = plane["net_lock"] > 0.5
    p = plane.loc[locked]
    speed = np.hypot(p["net_vel_u"], p["net_vel_v"])

    prof = {
        "clip": clip,
        "day": clip.split("_")[0],
        "samples": int(len(plane)),
        "net_lock_pct": round(100.0 * float(locked.mean()), 2),
        "achieved": {
            "standoff_m_mean": round(float(p["net_distance"].mean()), 3),
            "standoff_m_sd": round(float(p["net_distance"].std()), 3),
            "standoff_m_min": round(float(p["net_distance"].min()), 3),
            "standoff_m_max": round(float(p["net_distance"].max()), 3),
            "net_speed_ms_mean": round(float(speed.mean()), 3),
            "net_pitch_sd": round(float(p["net_pitch"].std()), 4),
            "net_heading_sd": round(float(p["net_heading"].std()), 4),
        },
    }
    if setp is not None and not setp.empty:
        prof["commanded"] = {
            "standoff_m": round(float(setp["d_net_distance"].median()), 3),
            "sweep_speed_ms": round(float(setp["d_net_velocity_horizontal"].median()), 3),
            "vertical_speed_ms": round(float(setp["d_net_velocity_vertical"].median()), 3),
        }
        prof["tracking"] = {
            "standoff_error_m_mean": round(
                float((p["net_distance"] - float(setp["d_net_distance"].median())).mean()), 3),
            "standoff_error_m_abs_mean": round(
                float((p["net_distance"] - float(setp["d_net_distance"].median())).abs().mean()), 3),
        }
    if depth is not None and not depth.empty:
        prof["environment"] = {
            "depth_m_mean": round(float(depth["depth"].mean()), 3),
            "depth_m_max": round(float(depth["depth"].max()), 3),
            "temperature_c_mean": round(float(depth["temperature"].mean()), 3),
        }
    return prof


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", default=None, help="A single *_data.bag to extract")
    ap.add_argument("--all", action="store_true",
                    help="Extract every *_data.bag under --raw-dir")
    ap.add_argument("--raw-dir", default=DEFAULT_RAW)
    ap.add_argument("--out", default=DEFAULT_OUT, help="Parquet output directory")
    ap.add_argument("--topics", default=None, metavar="BAG",
                    help="List every topic in a bag and exit")
    ap.add_argument("--streams", default=None,
                    help="Comma-separated canonical stream names (default: all)")
    ap.add_argument("--describe", action="store_true",
                    help="Print the canonical stream definitions and exit")
    ap.add_argument("--profile-out", default=None,
                    help="Write a cross-clip flight-profile JSON here")
    args = ap.parse_args()

    if args.describe:
        for s in T.describe_streams():
            print(f"{s['name']:12s} {', '.join(s['topics'])}")
            print(f"             {s['description']}")
            if s["units"]:
                print(f"             units: {s['units']}")
        return

    if args.topics:
        for t in T.list_topics(args.topics):
            print(f"  {t['topic']:52s} {t['msgtype']:40s} n={t['count']}")
        return

    if args.bag:
        bags = [Path(args.bag)]
    elif args.all:
        bags = sorted(Path(args.raw_dir).glob("*_data.bag"))
        if not bags:
            raise SystemExit(f"No *_data.bag found in {args.raw_dir}. "
                             f"Download them with scripts/fetch_solaqua.py --data-id ...")
    else:
        ap.error("Specify --bag, --all, --topics, or --describe.")

    streams_wanted = args.streams.split(",") if args.streams else None
    out_dir = ensure_dir(args.out)
    profiles = []

    for bag in bags:
        clip = T.clip_id(bag)
        streams = T.extract_telemetry(bag, streams=streams_wanted)
        written = T.save_telemetry(streams, out_dir, clip)
        print(f"\n{clip}: {len(streams)} streams -> {len(written)} parquet files in {out_dir}")
        for name, info in T.summarise(streams).items():
            print(f"   {name:12s} {info['rows']:6d} rows  {info['rate_hz']:6.2f} Hz  "
                  f"{info['source_topic']}")
        profiles.append(_flight_profile(clip, streams))

    if profiles:
        print("\n" + "=" * 78)
        print("FLIGHT PROFILE PER CLIP  (commanded vs achieved)")
        print("=" * 78)
        hdr = f"{'clip':22s} {'cmd_d':>6s} {'ach_d':>6s} {'cmd_v':>6s} {'ach_v':>6s} {'depth':>6s} {'lock%':>6s}"
        print(hdr)
        for p in profiles:
            if "error" in p:
                continue
            c = p.get("commanded", {})
            a = p["achieved"]
            e = p.get("environment", {})
            print(f"{p['clip']:22s} {c.get('standoff_m', float('nan')):6.2f} "
                  f"{a['standoff_m_mean']:6.2f} {c.get('sweep_speed_ms', float('nan')):6.2f} "
                  f"{a['net_speed_ms_mean']:6.2f} {e.get('depth_m_mean', float('nan')):6.2f} "
                  f"{p['net_lock_pct']:6.1f}")

    if args.profile_out:
        write_json({"profiles": profiles,
                    "streams": T.describe_streams()}, args.profile_out)
        print(f"\nWrote flight profiles -> {args.profile_out}")


if __name__ == "__main__":
    main()
