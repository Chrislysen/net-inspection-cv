"""ROV telemetry extraction from SOLAQUA sensor bags.

The SOLAQUA release ships two files per inspection run:

``<stamp>_video.bag``   camera (and multibeam sonar) payload, 0.9–2.3 GB
``<stamp>_data.bag``    everything else, 1.5–3.3 MB

Earlier work in this repo used only the video bags. This module reads the
*data* bags, which carry the vehicle's navigation and sensor state — including
the one signal that matters most for an inspection system:
``/navigation/plane_approximation``, the ROV's estimated **distance and
orientation relative to the net plane**.

Why that matters
----------------
A net-inspection model is not evaluated in a vacuum. Its false-alarm rate
depends on how the imagery was captured: how far the vehicle was from the net,
how fast it was sweeping, and whether the net-plane estimate was even locked.
Joining telemetry to frames (see :mod:`netinspect.frame_sync`) turns a bare
model score into an **operating envelope** — the conditions under which the
detector's measured behaviour actually applies.

Sensor-suite drift is real and is *not* hidden
----------------------------------------------
The two recording days do not share a sensor suite:

============================  ==========================  =========================
Signal                        2024-08-20 (Waterlinked)    2024-08-22 (Nortek)
============================  ==========================  =========================
DVL velocity / altitude       ``/sensor/dvl_velocity``    ``/nucleus1000dvl/bottomtrack``
Depth + water temperature     ``/sensor/depth_temperature``  also ``/nucleus1000dvl/ins``
IMU                           ``/sensor/imu`` (counts)    ``/nucleus1000dvl/imu`` (SI)
============================  ==========================  =========================

Those are different instruments with **different units**. ``/sensor/imu``
reports roughly milli-g and milli-deg/s integer counts (``acc_z ≈ -1012``);
``/nucleus1000dvl/imu`` reports m/s² and rad/s (``accelerometer.y ≈ 9.73``).
Silently concatenating them would manufacture a step change at the day
boundary and corrupt any cross-day comparison, so this module keeps them as
separate canonical streams (``imu_counts`` / ``imu_si``) and stamps every
extracted frame with the topic it came from (``source_topic``). Where a
fallback *is* physically equivalent, it is declared in :data:`STREAM_SPECS`
and recorded per row rather than assumed.

Timebase
--------
Every row carries ``t``, the bag record time in **Unix epoch seconds**. Bag
record time is used (rather than ``header.stamp``) because it is present on
every message — ``/commanded_thrust`` has no header at all — and because the
video bags are stamped on the same clock, which is what makes the frame join
in :mod:`netinspect.frame_sync` exact. Where a header stamp exists it is kept
alongside as ``t_header``; on the clips checked the two agree to ~1 ms.

Examples
--------
>>> from netinspect import telemetry
>>> streams = telemetry.extract_telemetry("data/raw/solaqua/2024-08-22_14-47-39_data.bag")
>>> streams["net_plane"][["t", "net_distance", "net_lock"]].head()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .utils import ensure_dir, get_logger, require

LOGGER = get_logger()

# Sentinel used by the BlueROV thruster telemetry for "channel not present".
THRUST_INVALID = 65535.0


def _require_rosbags():
    """Import ``rosbags.highlevel.AnyReader`` or raise an actionable error."""
    mod = require("rosbags.highlevel", hint="Install the data extra: pip install -e '.[data]'")
    return mod.AnyReader


# --------------------------------------------------------------------------- #
# Field extractors
#
# Each takes a deserialised ROS message and returns a flat dict of scalars.
# They must not raise on missing optional fields — bags differ between days.
# --------------------------------------------------------------------------- #
def _f(value: Any) -> float:
    """Coerce a ROS scalar to float, mapping anything non-numeric to NaN."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _vec3(v: Any, prefix: str) -> dict[str, float]:
    """Flatten a ``geometry_msgs/Vector3`` into ``{prefix}_x/y/z``."""
    return {f"{prefix}_x": _f(getattr(v, "x", "nan")),
            f"{prefix}_y": _f(getattr(v, "y", "nan")),
            f"{prefix}_z": _f(getattr(v, "z", "nan"))}


def _extract_net_plane(m: Any) -> dict[str, float]:
    """``messages/msg/PlaneApproximation`` — ROV pose relative to the net plane.

    ``NetDistance`` is the standoff in metres; ``NetLock`` is the estimator's
    own validity flag (1.0 = locked). ``NormalDVL`` is the net-plane normal in
    the DVL frame.
    """
    normal = getattr(m, "NormalDVL", None)
    out = {
        "net_distance": _f(getattr(m, "NetDistance", "nan")),
        "net_heading": _f(getattr(m, "NetHeading", "nan")),
        "net_pitch": _f(getattr(m, "NetPitch", "nan")),
        "net_lock": _f(getattr(m, "NetLock", "nan")),
        "net_vel_u": _f(getattr(m, "NetVelocity_u", "nan")),
        "net_vel_v": _f(getattr(m, "NetVelocity_v", "nan")),
        "net_vel_w": _f(getattr(m, "NetVelocity_w", "nan")),
        "dvl_altitude": _f(getattr(m, "Altitude", "nan")),
    }
    if normal is not None:
        try:
            out.update({f"net_normal_{a}": _f(normal[i]) for i, a in enumerate("xyz")})
        except (TypeError, IndexError, KeyError):
            pass
    return out


def _extract_net_pose(m: Any) -> dict[str, float]:
    """``messages/msg/PlaneApproximationPosition`` — dead-reckoned pose in the net frame.

    Position is relative to the last estimator reset (``t_last_reset``), not a
    geodetic origin, so it is usable for *relative* swept-path reconstruction
    within a clip and not for absolute localisation across clips.
    """
    return {
        "x": _f(getattr(m, "x", "nan")),
        "y": _f(getattr(m, "y", "nan")),
        "z": _f(getattr(m, "z", "nan")),
        "roll": _f(getattr(m, "roll", "nan")),
        "pitch": _f(getattr(m, "pitch", "nan")),
        "yaw": _f(getattr(m, "yaw", "nan")),
        "t_last_reset": _f(getattr(m, "t_last_reset", "nan")),
    }


def _extract_dvl_a50(m: Any) -> dict[str, float]:
    """``sensors/msg/DVLVelocity`` — Waterlinked A50 (2024-08-20 suite).

    ``fom`` is the vendor's figure of merit (lower is better) and
    ``velocity_valid`` is its own validity flag; both are kept so downstream
    analysis can drop untrustworthy samples rather than averaging them in.
    """
    out = _vec3(getattr(m, "velocity", None), "vel")
    out.update({
        "altitude": _f(getattr(m, "altitude", "nan")),
        "fom": _f(getattr(m, "fom", "nan")),
        "valid": 1.0 if bool(getattr(m, "velocity_valid", False)) else 0.0,
    })
    return out


def _extract_dvl_nucleus(m: Any) -> dict[str, float]:
    """``sensors/msg/Nucleus1000_bottomtrack`` — Nortek Nucleus 1000 (2024-08-22 suite).

    Mapped onto the same column names as the A50 so the two days are
    comparable; ``source_topic`` records which instrument produced each row.
    ``altitude`` is the mean of the valid beam distances, which is the closest
    equivalent the Nucleus message offers to the A50's reported altitude.
    """
    out = _vec3(getattr(m, "dvl_velocity_xyz", None), "vel")
    beam = getattr(m, "beam_distance", None)
    alt = float("nan")
    if beam is not None:
        try:
            vals = [float(b) for b in beam if float(b) > 0.0]
            if vals:
                alt = sum(vals) / len(vals)
        except (TypeError, ValueError):
            pass
    fom = getattr(m, "dvl_fom_xyz", None)
    out.update({
        "altitude": alt,
        "fom": _f(getattr(fom, "x", "nan")) if fom is not None else float("nan"),
        "valid": 1.0 if bool(getattr(m, "data_valid", False)) else 0.0,
    })
    return out


def _extract_depth_temp(m: Any) -> dict[str, float]:
    """``messages/msg/DepthTemperature`` — depth (m), water temperature (°C), pressure (mbar)."""
    return {
        "depth": _f(getattr(m, "depth", "nan")),
        "temperature": _f(getattr(m, "temperature", "nan")),
        "pressure": _f(getattr(m, "pressure", "nan")),
    }


def _extract_ins(m: Any) -> dict[str, float]:
    """``sensors/msg/Nucleus1000_ins`` — INS solution (Nortek suite only).

    Note the pressure here is in bar while ``/sensor/depth_temperature`` is in
    mbar; the columns are deliberately *not* merged into one canonical
    pressure. Temperature is in °C on both and is comparable.
    """
    out = {
        "depth": _f(getattr(m, "depth", "nan")),
        "temperature": _f(getattr(m, "temperature", "nan")),
        "pressure_bar": _f(getattr(m, "pressure", "nan")),
        "speed_over_ground": _f(getattr(m, "speedOverGround", "nan")),
        "course_over_ground": _f(getattr(m, "courseOverGround", "nan")),
        "fom_ins": _f(getattr(m, "fomIns", "nan")),
        "fom_ahrs": _f(getattr(m, "fomAhrs", "nan")),
    }
    out.update(_vec3(getattr(m, "positionFrame", None), "pos"))
    out.update(_vec3(getattr(m, "velocityNed", None), "vel_ned"))
    return out


def _extract_attitude(m: Any) -> dict[str, float]:
    """``messages/msg/Attitude`` — vehicle attitude (rad) and body rates (rad/s)."""
    return {k: _f(getattr(m, k, "nan"))
            for k in ("roll", "pitch", "yaw", "rollspeed", "pitchspeed", "yawspeed")}


def _extract_imu_counts(m: Any) -> dict[str, float]:
    """``messages/msg/IMU`` — raw integer counts (~milli-g, ~milli-deg/s). NOT SI."""
    return {k: _f(getattr(m, k, "nan"))
            for k in ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
                      "mag_x", "mag_y", "mag_z")}


def _extract_imu_si(m: Any) -> dict[str, float]:
    """``sensors/msg/Nucleus1000_imu`` — accelerometer (m/s²) and gyroscope (rad/s)."""
    out = _vec3(getattr(m, "accelerometer", None), "acc")
    out.update(_vec3(getattr(m, "gyroscope", None), "gyro"))
    return out


def _extract_setpoint(m: Any) -> dict[str, float]:
    """``messages/msg/NetFollowingManager`` — the *commanded* inspection envelope.

    This is the operator's intent: desired standoff (``d_net_distance``, 0.6 m
    on these clips) and desired sweep speed (``d_net_velocity_horizontal``).
    Comparing it against the achieved ``net_distance`` is what makes an
    inspection auditable rather than merely recorded.
    """
    return {k: _f(getattr(m, k, "nan"))
            for k in ("d_depth", "d_net_distance", "d_net_velocity_horizontal",
                      "d_net_velocity_vertical", "offset_net_heading")}


def _extract_guidance(m: Any) -> dict[str, float]:
    """``messages/msg/GuidanceManager`` — tracking errors and desired velocities."""
    return {k: _f(getattr(m, k, "nan"))
            for k in ("error_x", "error_y", "error_z", "error_net_distance",
                      "error_surge", "error_sway", "error_heave", "error_depth",
                      "error_yaw", "desired_surge", "desired_sway", "desired_heave",
                      "desired_yaw", "desired_net_distance", "gamma_p")}


def _extract_battery(m: Any) -> dict[str, float]:
    """``messages/msg/BatteryStatus`` — pack voltage in mV plus consumption counters."""
    return {
        "voltage_mv": _f(getattr(m, "voltage", "nan")),
        "current_battery": _f(getattr(m, "current_battery", "nan")),
        "current_consumed": _f(getattr(m, "current_consumed", "nan")),
    }


def _extract_usbl(m: Any) -> dict[str, float]:
    """``sensors/msg/SonardyneUSBL2`` — acoustic position fix (local E/N/D frame)."""
    return {
        "east": _f(getattr(m, "east", "nan")),
        "north": _f(getattr(m, "north", "nan")),
        "depth": _f(getattr(m, "depth", "nan")),
        "heading": _f(getattr(m, "heading", "nan")),
        "roll": _f(getattr(m, "roll", "nan")),
        "pitch": _f(getattr(m, "pitch", "nan")),
        "sigma": _f(getattr(m, "sigma", "nan")),
        "connected": 1.0 if bool(getattr(m, "connected", False)) else 0.0,
    }


def _extract_thrust(m: Any) -> dict[str, float]:
    """``rospy_tutorials/msg/Floats`` — commanded PWM per thruster channel.

    Channels carrying :data:`THRUST_INVALID` (65535) are absent on this
    airframe and are mapped to NaN rather than being averaged in as a huge
    number. ``thrust_effort`` summarises |PWM − 1500| over the valid channels,
    a rough proxy for how hard the vehicle was working to hold station.
    """
    data = getattr(m, "data", None)
    out: dict[str, float] = {}
    efforts: list[float] = []
    if data is not None:
        for i, raw in enumerate(list(data)[:8]):
            v = _f(raw)
            if v == THRUST_INVALID:
                v = float("nan")
            else:
                efforts.append(abs(v - 1500.0))
            out[f"thrust_{i}"] = v
    out["thrust_effort"] = sum(efforts) / len(efforts) if efforts else float("nan")
    return out


# --------------------------------------------------------------------------- #
# Stream specifications
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StreamSpec:
    """One canonical telemetry stream and the topics that can supply it.

    Attributes
    ----------
    name : str
        Canonical stream name (the key in the returned mapping).
    topics : tuple[str, ...]
        Candidate topics in preference order. The first one present in the bag
        wins; the choice is recorded per row in ``source_topic``.
    extractor : callable
        Maps a deserialised message to a flat dict of scalars.
    description : str
        Human-readable summary, surfaced by :func:`describe_streams`.
    units : dict[str, str]
        Column -> unit. Documented explicitly because the two recording days
        use different instruments (see the module docstring).
    """
    name: str
    topics: tuple[str, ...]
    extractor: Callable[[Any], dict[str, float]]
    description: str
    units: dict[str, str] = field(default_factory=dict)


STREAM_SPECS: tuple[StreamSpec, ...] = (
    StreamSpec(
        "net_plane", ("/navigation/plane_approximation",), _extract_net_plane,
        "ROV pose relative to the estimated net plane (standoff, heading, pitch, lock).",
        {"net_distance": "m", "net_heading": "rad", "net_pitch": "rad",
         "net_lock": "bool", "net_vel_u": "m/s", "net_vel_v": "m/s",
         "net_vel_w": "m/s", "dvl_altitude": "m"},
    ),
    StreamSpec(
        "net_pose", ("/navigation/plane_approximation_position",), _extract_net_pose,
        "Dead-reckoned position in the net frame since the last estimator reset.",
        {"x": "m", "y": "m", "z": "m", "roll": "deg", "pitch": "deg", "yaw": "deg"},
    ),
    StreamSpec(
        # Preference order matters: the A50 topic is the one common to both days.
        "dvl", ("/sensor/dvl_velocity", "/nucleus1000dvl/bottomtrack"),
        _extract_dvl_a50,
        "Doppler velocity log: body-frame velocity, altitude, quality flags.",
        {"vel_x": "m/s", "vel_y": "m/s", "vel_z": "m/s", "altitude": "m",
         "fom": "m/s (lower better)", "valid": "bool"},
    ),
    StreamSpec(
        "depth_temp", ("/sensor/depth_temperature",), _extract_depth_temp,
        "Pressure-derived depth and in-situ water temperature.",
        {"depth": "m", "temperature": "degC", "pressure": "mbar"},
    ),
    StreamSpec(
        "ins", ("/nucleus1000dvl/ins",), _extract_ins,
        "Nortek INS solution (2024-08-22 suite only): depth, temperature, SOG/COG, FOM.",
        {"depth": "m", "temperature": "degC", "pressure_bar": "bar",
         "speed_over_ground": "m/s", "course_over_ground": "deg"},
    ),
    StreamSpec(
        "attitude", ("/sensor/attitude",), _extract_attitude,
        "Vehicle attitude and body rates.",
        {"roll": "rad", "pitch": "rad", "yaw": "rad",
         "rollspeed": "rad/s", "pitchspeed": "rad/s", "yawspeed": "rad/s"},
    ),
    StreamSpec(
        "imu_counts", ("/sensor/imu",), _extract_imu_counts,
        "Raw IMU counts (~milli-g, ~milli-deg/s). NOT SI — do not merge with imu_si.",
        {"acc_x": "counts (~mg)", "gyro_x": "counts (~mdeg/s)"},
    ),
    StreamSpec(
        "imu_si", ("/nucleus1000dvl/imu",), _extract_imu_si,
        "Nortek IMU in SI units. NOT comparable to imu_counts without calibration.",
        {"acc_x": "m/s2", "gyro_x": "rad/s"},
    ),
    StreamSpec(
        "setpoint", ("/gui/netFollowing_manager",), _extract_setpoint,
        "Commanded net-following envelope: desired standoff and sweep velocity.",
        {"d_net_distance": "m", "d_net_velocity_horizontal": "m/s",
         "d_net_velocity_vertical": "m/s", "d_depth": "m", "offset_net_heading": "rad"},
    ),
    StreamSpec(
        "guidance", ("/guidance",), _extract_guidance,
        "Guidance tracking errors and desired body velocities.",
        {"error_net_distance": "m", "desired_net_distance": "m",
         "error_surge": "m/s", "error_sway": "m/s", "error_yaw": "deg"},
    ),
    StreamSpec(
        "battery", ("/bluerov2/battery",), _extract_battery,
        "Battery pack voltage and consumption counters.",
        {"voltage_mv": "mV", "current_consumed": "mAh"},
    ),
    StreamSpec(
        "usbl", ("/sensor/usbl",), _extract_usbl,
        "Sonardyne USBL acoustic position fix.",
        {"east": "m", "north": "m", "depth": "m", "heading": "deg", "sigma": "m"},
    ),
    StreamSpec(
        "thrust", ("/commanded_thrust",), _extract_thrust,
        "Commanded thruster PWM per channel plus a mean-effort summary.",
        {"thrust_0": "PWM us", "thrust_effort": "PWM us from neutral"},
    ),
)

SPECS_BY_NAME: dict[str, StreamSpec] = {s.name: s for s in STREAM_SPECS}


def describe_streams() -> list[dict[str, Any]]:
    """Return a serialisable description of every canonical stream."""
    return [
        {"name": s.name, "topics": list(s.topics),
         "description": s.description, "units": dict(s.units)}
        for s in STREAM_SPECS
    ]


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def _header_time(msg: Any) -> float:
    """Seconds from a ``std_msgs/Header`` stamp, or NaN when absent."""
    hdr = getattr(msg, "header", None)
    stamp = getattr(hdr, "stamp", None) if hdr is not None else None
    if stamp is None:
        return float("nan")
    return _f(getattr(stamp, "sec", 0)) + _f(getattr(stamp, "nanosec", 0)) * 1e-9


def list_topics(bag_path: str | Path) -> list[dict[str, Any]]:
    """Return every topic in a bag with its message type and count."""
    AnyReader = _require_rosbags()
    with AnyReader([Path(bag_path)]) as reader:
        return sorted(
            ({"topic": c.topic, "msgtype": c.msgtype, "count": c.msgcount}
             for c in reader.connections),
            key=lambda d: -d["count"],
        )


def extract_stream(bag_path: str | Path, spec: StreamSpec | str):
    """Extract one canonical stream from a bag as a ``pandas.DataFrame``.

    Returns an empty DataFrame when none of the spec's candidate topics are
    present — a missing sensor suite is normal across recording days and is
    not treated as an error.
    """
    pd = require("pandas", hint="pip install pandas")
    if isinstance(spec, str):
        spec = SPECS_BY_NAME[spec]
    AnyReader = _require_rosbags()

    rows: list[dict[str, Any]] = []
    chosen_topic: str | None = None
    with AnyReader([Path(bag_path)]) as reader:
        by_topic: dict[str, list] = {}
        for c in reader.connections:
            by_topic.setdefault(c.topic, []).append(c)
        for candidate in spec.topics:
            if candidate in by_topic:
                chosen_topic = candidate
                break
        if chosen_topic is None:
            return pd.DataFrame()

        # The Nucleus bottomtrack message needs its own extractor even though
        # it feeds the same canonical 'dvl' stream as the A50.
        extractor = spec.extractor
        if chosen_topic == "/nucleus1000dvl/bottomtrack":
            extractor = _extract_dvl_nucleus

        conns = by_topic[chosen_topic]
        for conn, ts, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            row: dict[str, Any] = {"t": ts * 1e-9, "t_header": _header_time(msg)}
            row.update(extractor(msg))
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("t", kind="stable").reset_index(drop=True)
        df.attrs["source_topic"] = chosen_topic
        df["source_topic"] = chosen_topic
    LOGGER.info("  %-12s %-42s n=%d", spec.name, chosen_topic or "(absent)", len(df))
    return df


def extract_telemetry(
    bag_path: str | Path,
    streams: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Extract all canonical streams from a SOLAQUA ``*_data.bag``.

    Parameters
    ----------
    bag_path : path
        Path to a ``*_data.bag``.
    streams : iterable of str, optional
        Restrict to these canonical stream names. Defaults to all of
        :data:`STREAM_SPECS`.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Stream name -> DataFrame. Streams absent from the bag are omitted, so
        callers should use ``.get(name)`` rather than assuming presence.
    """
    bag_path = Path(bag_path)
    wanted = list(streams) if streams is not None else [s.name for s in STREAM_SPECS]
    LOGGER.info("Extracting telemetry from %s", bag_path.name)
    out: dict[str, Any] = {}
    for name in wanted:
        df = extract_stream(bag_path, SPECS_BY_NAME[name])
        if not df.empty:
            out[name] = df
    return out


def clip_id(bag_path: str | Path) -> str:
    """Canonical clip identifier shared by a run's video and data bags.

    ``2024-08-22_14-47-39_data.bag`` and ``2024-08-22_14-47-39_video.bag`` both
    map to ``2024-08-22_14-47-39``, which is also the prefix used by the
    extracted frame filenames.
    """
    stem = Path(bag_path).stem
    for suffix in ("_data", "_video"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


# --------------------------------------------------------------------------- #
# Persistence + summary
# --------------------------------------------------------------------------- #
def save_telemetry(streams: dict[str, Any], out_dir: str | Path, clip: str) -> list[Path]:
    """Write each stream to ``<out_dir>/<clip>__<stream>.parquet``."""
    out_dir = ensure_dir(out_dir)
    written: list[Path] = []
    for name, df in streams.items():
        path = out_dir / f"{clip}__{name}.parquet"
        df.to_parquet(path, index=False)
        written.append(path)
    return written


def load_telemetry(in_dir: str | Path, clip: str) -> dict[str, Any]:
    """Load every stream previously written for ``clip`` by :func:`save_telemetry`."""
    pd = require("pandas", hint="pip install pandas")
    in_dir = Path(in_dir)
    out: dict[str, Any] = {}
    for path in sorted(in_dir.glob(f"{clip}__*.parquet")):
        name = path.stem.split("__", 1)[1]
        out[name] = pd.read_parquet(path)
    return out


def summarise(streams: dict[str, Any]) -> dict[str, Any]:
    """Compact per-stream summary: row count, source topic, duration, rate."""
    summary: dict[str, Any] = {}
    for name, df in streams.items():
        if df.empty:
            continue
        t0, t1 = float(df["t"].iloc[0]), float(df["t"].iloc[-1])
        span = t1 - t0
        summary[name] = {
            "rows": int(len(df)),
            "source_topic": df.attrs.get("source_topic")
            or (df["source_topic"].iloc[0] if "source_topic" in df else None),
            "t_start": t0,
            "t_end": t1,
            "duration_s": round(span, 3),
            "rate_hz": round(len(df) / span, 2) if span > 0 else None,
        }
    return summary


def merge_on_times(
    streams: dict[str, Any],
    times: Iterable[float],
    tolerance_s: float = 0.5,
    include: Iterable[str] | None = None,
):
    """As-of join every stream onto an arbitrary set of query timestamps.

    Uses a nearest-neighbour join bounded by ``tolerance_s``: a query time with
    no sample inside the tolerance gets NaN for that stream's columns rather
    than a silently stale value. Column names are prefixed with the stream name
    (``net_plane_net_distance``) so provenance survives the merge.

    Parameters
    ----------
    streams : dict[str, DataFrame]
        Output of :func:`extract_telemetry`.
    times : iterable of float
        Query timestamps in Unix epoch seconds (e.g. frame capture times).
    tolerance_s : float
        Maximum allowed gap between a query time and the matched sample.
    include : iterable of str, optional
        Restrict to these stream names.

    Returns
    -------
    pandas.DataFrame
        One row per query time, with a ``t`` column and prefixed stream columns.
    """
    pd = require("pandas", hint="pip install pandas")
    base = pd.DataFrame({"t": [float(x) for x in times]}).sort_values("t").reset_index(drop=True)
    names = list(include) if include is not None else list(streams)
    for name in names:
        df = streams.get(name)
        if df is None or df.empty:
            continue
        right = df.drop(columns=[c for c in ("t_header", "source_topic") if c in df.columns])
        right = right.sort_values("t").reset_index(drop=True)
        right = right.rename(columns={c: f"{name}_{c}" for c in right.columns if c != "t"})
        base = pd.merge_asof(
            base, right, on="t", direction="nearest",
            tolerance=float(tolerance_s), suffixes=("", f"_{name}"),
        )
    return base


__all__ = [
    "STREAM_SPECS", "SPECS_BY_NAME", "StreamSpec", "THRUST_INVALID",
    "describe_streams", "list_topics", "extract_stream", "extract_telemetry",
    "clip_id", "save_telemetry", "load_telemetry", "summarise", "merge_on_times",
]
