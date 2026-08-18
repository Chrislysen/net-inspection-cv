"""Join extracted camera frames to ROV telemetry on the bag clock.

Frames in this repo were extracted with :func:`netinspect.solaqua.extract_bag_frames`,
which names each file ``<clip>_video_<index:06d>.jpg`` where ``index`` is the
message index on the chosen image topic. That index is the bridge back to a
timestamp: replaying the video bag's image connection and recording the record
time of every message yields ``index -> t``, and telemetry from the paired
``*_data.bag`` lives on the same clock (see :mod:`netinspect.telemetry`).

Building the index requires one streaming pass over the video bag (0.9–2.3 GB),
so results are cached as JSON next to the frames and reused thereafter.

Why bother
----------
Once frames carry timestamps, every model output can be conditioned on the
*conditions of capture* — standoff distance from the net, sweep speed, vehicle
attitude, whether the net-plane estimate was locked. Because the SOLAQUA nets
are undamaged, every detection on these frames is a known false positive, so
this join measures false-alarm rate as a function of flight profile without
needing a single new annotation.

Examples
--------
>>> from netinspect import frame_sync, telemetry
>>> idx = frame_sync.build_frame_index("data/raw/solaqua/2024-08-22_14-47-39_video.bag")
>>> tele = telemetry.extract_telemetry("data/raw/solaqua/2024-08-22_14-47-39_data.bag")
>>> df = frame_sync.join_frames("data/processed/solaqua_bag2", tele, idx)
>>> df[["frame", "t", "net_plane_net_distance"]].head()
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from .utils import ensure_dir, get_logger, list_images, read_json, require, write_json

LOGGER = get_logger()

# ``2024-08-22_14-47-39_video_000123.jpg`` -> clip, stream tag, message index.
FRAME_RE = re.compile(r"^(?P<clip>.+?)_(?P<tag>video|sonar)_(?P<index>\d+)$")

DEFAULT_CACHE_DIR = Path("data/processed/frame_index")


def parse_frame_name(name: str | Path) -> tuple[str, int] | None:
    """Parse ``<clip>_video_<index>.jpg`` into ``(clip, index)``.

    Returns ``None`` for filenames that do not follow the convention, so
    callers can skip stray files instead of crashing.
    """
    stem = Path(name).stem
    m = FRAME_RE.match(stem)
    if not m:
        return None
    return m.group("clip"), int(m.group("index"))


def _require_rosbags():
    mod = require("rosbags.highlevel", hint="Install the data extra: pip install -e '.[data]'")
    return mod.AnyReader


def _cache_path(clip: str, topic: str | None, cache_dir: str | Path) -> Path:
    tag = (topic or "auto").strip("/").replace("/", "_")
    return Path(cache_dir) / f"{clip}__{tag}__frame_index.json"


def build_frame_index(
    video_bag: str | Path,
    topic: str | None = None,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    force: bool = False,
) -> dict[str, Any]:
    """Map image-topic message index -> bag record time for one video bag.

    Parameters
    ----------
    video_bag : path
        Path to a ``*_video.bag``.
    topic : str, optional
        Image topic. Defaults to the image topic with the most messages, which
        is the same rule :func:`netinspect.solaqua.extract_bag_frames` used, so
        the indices line up with the extracted filenames.
    cache_dir : path
        Where to store/reuse the JSON index.
    force : bool
        Rebuild even if a cache exists.

    Returns
    -------
    dict
        ``{"clip", "topic", "msgtype", "count", "times"}`` where ``times[i]`` is
        the Unix-epoch second of message ``i``.

    Notes
    -----
    This streams the whole bag once. Message payloads are not deserialised —
    only record times are kept — but the bytes are still read from disk, so
    expect tens of seconds per bag on first run.
    """
    from .telemetry import clip_id  # local import: avoids a cycle at module load

    video_bag = Path(video_bag)
    clip = clip_id(video_bag)
    AnyReader = _require_rosbags()

    cache = _cache_path(clip, topic, cache_dir)
    if cache.exists() and not force:
        cached = read_json(cache)
        if topic is None or cached.get("topic") == topic:
            LOGGER.info("Frame index for %s loaded from cache (%d msgs)",
                        clip, cached.get("count", 0))
            return cached

    image_types = ("sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage")
    times: list[float] = []
    with AnyReader([video_bag]) as reader:
        img_conns = [c for c in reader.connections if c.msgtype in image_types]
        if not img_conns:
            raise RuntimeError(f"No image topics in {video_bag.name}.")
        if topic is None:
            chosen = max(img_conns, key=lambda c: c.msgcount)
        else:
            chosen = next((c for c in img_conns if c.topic == topic), None)
            if chosen is None:
                raise ValueError(f"Topic {topic!r} not found or not an image topic.")
        LOGGER.info("Indexing %s on %s (%d msgs) — one streaming pass",
                    video_bag.name, chosen.topic, chosen.msgcount)
        conns = [c for c in reader.connections if c.topic == chosen.topic]
        for _conn, ts, _raw in reader.messages(connections=conns):
            times.append(ts * 1e-9)

    index = {
        "clip": clip,
        "topic": chosen.topic,
        "msgtype": chosen.msgtype,
        "count": len(times),
        "t_start": times[0] if times else None,
        "t_end": times[-1] if times else None,
        "times": times,
    }
    ensure_dir(Path(cache_dir))
    write_json(index, cache)
    LOGGER.info("Indexed %d messages (%.1f s) -> %s",
                len(times), (times[-1] - times[0]) if len(times) > 1 else 0.0, cache.name)
    return index


def load_frame_index(clip: str, topic: str | None = None,
                     cache_dir: str | Path = DEFAULT_CACHE_DIR) -> dict[str, Any] | None:
    """Load a cached index for ``clip``, or ``None`` if it has not been built."""
    cache = _cache_path(clip, topic, cache_dir)
    if cache.exists():
        return read_json(cache)
    matches = sorted(Path(cache_dir).glob(f"{clip}__*__frame_index.json"))
    return read_json(matches[0]) if matches else None


def frame_time(index: dict[str, Any], msg_index: int) -> float:
    """Timestamp of one image-topic message index, or NaN if out of range."""
    times = index.get("times") or []
    if 0 <= msg_index < len(times):
        return float(times[msg_index])
    return float("nan")


def frame_records(
    frames_dir: str | Path,
    indices: dict[str, dict[str, Any]] | dict[str, Any],
) -> list[dict[str, Any]]:
    """Build ``{frame, path, clip, msg_index, t}`` records for a directory of frames.

    Parameters
    ----------
    frames_dir : path
        Directory of extracted ``.jpg`` frames.
    indices : dict
        Either a single frame index (as returned by :func:`build_frame_index`)
        or a mapping ``clip -> index``. A directory may mix clips, so the
        mapping form is preferred.

    Notes
    -----
    Frames whose clip has no index, or whose message index falls outside it,
    get ``t = NaN`` and are kept in the output rather than dropped, so the
    caller can see and report coverage explicitly.
    """
    by_clip: dict[str, dict[str, Any]] = (
        {indices["clip"]: indices} if "times" in indices else indices  # type: ignore[index]
    )
    out: list[dict[str, Any]] = []
    for path in list_images(frames_dir):
        parsed = parse_frame_name(path.name)
        if parsed is None:
            continue
        clip, msg_index = parsed
        idx = by_clip.get(clip)
        out.append({
            "frame": path.name,
            "path": str(path),
            "clip": clip,
            "msg_index": msg_index,
            "t": frame_time(idx, msg_index) if idx else float("nan"),
        })
    return out


def join_frames(
    frames_dir: str | Path,
    streams: dict[str, Any],
    indices: dict[str, dict[str, Any]] | dict[str, Any],
    tolerance_s: float = 0.5,
    include: Iterable[str] | None = None,
):
    """Join a directory of frames to telemetry via nearest-in-time matching.

    Parameters
    ----------
    frames_dir : path
        Directory of extracted frames.
    streams : dict[str, DataFrame]
        Telemetry from :func:`netinspect.telemetry.extract_telemetry`.
    indices : dict
        Frame index/indices from :func:`build_frame_index`.
    tolerance_s : float
        Maximum frame-to-sample gap. Beyond it the telemetry columns are NaN
        rather than stale — an unmatched frame is reported, never imputed.
    include : iterable of str, optional
        Restrict to these telemetry streams.

    Returns
    -------
    pandas.DataFrame
        One row per frame with ``frame``, ``clip``, ``msg_index``, ``t`` and
        prefixed telemetry columns (e.g. ``net_plane_net_distance``).
    """
    from .telemetry import merge_on_times

    pd = require("pandas", hint="pip install pandas")
    records = frame_records(frames_dir, indices)
    if not records:
        return pd.DataFrame()

    base = pd.DataFrame(records).sort_values("t", kind="stable").reset_index(drop=True)
    timed = base[base["t"].notna()].copy()
    if timed.empty:
        LOGGER.warning("No frames in %s could be timestamped.", frames_dir)
        return base

    joined = merge_on_times(streams, timed["t"].tolist(),
                            tolerance_s=tolerance_s, include=include)
    # merge_on_times sorts by t and returns one row per query time, so a
    # positional concat is safe here.
    timed = timed.reset_index(drop=True)
    joined = joined.reset_index(drop=True).drop(columns=["t"])
    out = pd.concat([timed, joined], axis=1)

    untimed = base[base["t"].isna()]
    if not untimed.empty:
        LOGGER.warning("%d/%d frames had no timestamp (missing or short index)",
                       len(untimed), len(base))
        out = pd.concat([out, untimed], ignore_index=True)
    return out.sort_values(["clip", "msg_index"], kind="stable").reset_index(drop=True)


def coverage_report(joined: Any, key: str = "net_plane_net_distance") -> dict[str, Any]:
    """Summarise how completely a join succeeded.

    Reports the fraction of frames that received a timestamp and the fraction
    that matched a telemetry sample within tolerance. Both are worth printing
    before trusting any downstream conditional statistic.
    """
    total = int(len(joined))
    if total == 0:
        return {"frames": 0, "timestamped": 0, "matched": 0}
    timestamped = int(joined["t"].notna().sum())
    matched = int(joined[key].notna().sum()) if key in joined.columns else 0
    return {
        "frames": total,
        "timestamped": timestamped,
        "timestamped_pct": round(100.0 * timestamped / total, 2),
        "matched": matched,
        "matched_pct": round(100.0 * matched / total, 2),
        "match_key": key,
    }


__all__ = [
    "FRAME_RE", "DEFAULT_CACHE_DIR", "parse_frame_name", "build_frame_index",
    "load_frame_index", "frame_time", "frame_records", "join_frames", "coverage_report",
]
