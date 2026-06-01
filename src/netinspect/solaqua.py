"""SOLAQUA dataset client + ROS-bag frame extraction.

SOLAQUA — *SINTEF Ocean Large Aquaculture Robotics Dataset* — is real ROV
footage from operational Norwegian fish farms (CC BY-SA 4.0). Data:
https://data.sintef.no/product/dp-7141fcd5-0fb8-4be3-b9ce-e5f7f5bb4a58
Paper: https://arxiv.org/abs/2504.01790

**Important characteristics for this prototype**
- The nets in the vision data are **undamaged** (with fish and marine growth);
  there are **no hole/tear/damage annotations**. We therefore use SOLAQUA for
  (a) real-image preprocessing, (b) false-positive / robustness analysis of the
  classical baseline, and (c) as "normal" data for anomaly detection — *not* for
  measuring damage-detection accuracy.
- Camera data ships inside ROS1 ``.bag`` files (``sensor_msgs/Image`` or
  ``CompressedImage``). Frame extraction uses the pure-Python ``rosbags`` lib;
  no ROS installation is required.

This module talks to SINTEF's public data API (reverse-engineered from the
portal): list features under the product, list files under a feature, and
download a file by id (the API 302-redirects to a presigned S3 URL).
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from .utils import ensure_dir, get_logger, optional_import

LOGGER = get_logger()

API_BASE = "https://data.sintef.no/api/public"
# The "Aquaculture robotics public datasets" product and its main SOLAQUA feature.
PRODUCT_ID = "dp-7141fcd5-0fb8-4be3-b9ce-e5f7f5bb4a58"
SOLAQUA_FEATURE_ID = "fe-a8f86232-5107-495e-a3dd-a86460eebef6"


@dataclass
class DataFile:
    """One downloadable file (a "data distribution") in a SOLAQUA feature."""
    data_id: str
    file_name: str
    file_size: int
    feature_id: str

    @property
    def size_mb(self) -> float:
        return self.file_size / (1024 * 1024)

    @property
    def is_video_bag(self) -> bool:
        return self.file_name.endswith("_video.bag")


# --------------------------------------------------------------------------- #
# Public API client
# --------------------------------------------------------------------------- #
def _get_json(url: str, timeout: int = 60) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def list_features(product_id: str = PRODUCT_ID) -> list[dict]:
    """List features (sub-datasets) under a data product."""
    data = _get_json(f"{API_BASE}/dataproducts/{product_id}/features")
    return data.get("items", [])


def list_files(feature_id: str = SOLAQUA_FEATURE_ID) -> list[DataFile]:
    """List downloadable files in a feature, sorted by size (ascending)."""
    data = _get_json(f"{API_BASE}/features/{feature_id}/data")
    files = []
    for item in data.get("items", []):
        attrs = item.get("fileAttributes") or {}
        if not attrs.get("fileName"):
            continue
        files.append(DataFile(
            data_id=item["dataID"],
            file_name=attrs["fileName"],
            file_size=int(attrs.get("fileSize", 0)),
            feature_id=item.get("featureID", feature_id),
        ))
    return sorted(files, key=lambda f: f.file_size)


def smallest_video_bag(feature_id: str = SOLAQUA_FEATURE_ID) -> DataFile | None:
    """The smallest ``*_video.bag`` (contains camera frames)."""
    vids = [f for f in list_files(feature_id) if f.is_video_bag]
    return vids[0] if vids else None


def download_file(
    data_id: str,
    out_path: str | Path,
    resume: bool = True,
    chunk_size: int = 1 << 20,
    progress: Callable[[int, int], None] | None = None,
    timeout: int = 120,
) -> Path:
    """Download a SOLAQUA file by id, following the API's 302 to presigned S3.

    Supports HTTP-range resume (the S3 backend advertises ``Accept-Ranges``).
    ``progress(downloaded, total)`` is called periodically if provided.
    """
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    url = f"{API_BASE}/data/{data_id}/file"

    existing = out_path.stat().st_size if (resume and out_path.exists()) else 0
    headers: dict[str, str] = {}
    if existing:
        headers["Range"] = f"bytes={existing}-"
        LOGGER.info("Resuming download at %d bytes", existing)

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        # total size (Content-Range when resuming, else Content-Length)
        total = existing
        cr = resp.headers.get("Content-Range")
        if cr and "/" in cr:
            total = int(cr.split("/")[-1])
        elif resp.headers.get("Content-Length"):
            total = existing + int(resp.headers["Content-Length"])

        mode = "ab" if existing else "wb"
        downloaded = existing
        with open(out_path, mode) as fh:
            while True:
                buf = resp.read(chunk_size)
                if not buf:
                    break
                fh.write(buf)
                downloaded += len(buf)
                if progress:
                    progress(downloaded, total)
    LOGGER.info("Downloaded %s (%d bytes)", out_path.name, out_path.stat().st_size)
    return out_path


# --------------------------------------------------------------------------- #
# ROS-bag frame extraction
# --------------------------------------------------------------------------- #
_IMAGE_TYPES = ("sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage")


def list_bag_topics(bag_path: str | Path) -> list[dict]:
    """Return the image-bearing topics in a ROS bag (topic, type, count)."""
    AnyReader = _require_rosbags()
    bag_path = Path(bag_path)
    with AnyReader([bag_path]) as reader:
        return [
            {"topic": c.topic, "msgtype": c.msgtype, "count": c.msgcount}
            for c in reader.connections
        ]


def _require_rosbags():
    rb = optional_import("rosbags.highlevel")
    if rb is None:
        raise RuntimeError(
            "The 'rosbags' package is required to read .bag files. "
            "Install it with `pip install rosbags`."
        )
    return rb.AnyReader


def _decode_message(msg, msgtype: str):
    """Decode a ROS image message to an RGB uint8 numpy array (or None)."""
    import numpy as np
    cv2 = optional_import("cv2")

    if "CompressedImage" in msgtype:
        if cv2 is None:
            return None
        buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Raw sensor_msgs/Image
    enc = (msg.encoding or "").lower()
    h, w = msg.height, msg.width
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if enc in ("rgb8", "bgr8"):
        img = data.reshape(h, w, 3)
        if enc == "bgr8" and cv2 is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif enc == "bgr8":
            img = img[..., ::-1]
        return np.ascontiguousarray(img)
    if enc in ("rgba8", "bgra8"):
        img = data.reshape(h, w, 4)[..., :3]
        if enc == "bgra8":
            img = img[..., ::-1]
        return np.ascontiguousarray(img)
    if enc == "mono8":
        gray = data.reshape(h, w)
        return np.repeat(gray[..., None], 3, axis=2)
    if enc.startswith("bayer") and cv2 is not None:
        bayer = data.reshape(h, w)
        # Best-effort: treat as BG bayer; exact pattern varies by sensor.
        return cv2.cvtColor(bayer, cv2.COLOR_BayerBG2RGB)
    return None


SONAR_MSGTYPE = "sensors/msg/SonoptixECHO"


def extract_sonar_frames(
    bag_path: str | Path,
    out_dir: str | Path,
    topic: str | None = None,
    every_n: int = 10,
    max_frames: int | None = 40,
    colormap: bool = True,
) -> list[Path]:
    """Extract multibeam-sonar (SonoptixECHO) frames as images.

    The SonoptixECHO message carries a ``Float32MultiArray`` of acoustic
    intensities (a square fan image, e.g. 512x512). We reshape it using its
    layout (or a square fallback), normalise to 8-bit, and optionally apply a
    perceptual colormap. This demonstrates **multi-modal ingestion**: sonar sees
    through turbidity where optical cameras fail, and is complementary for net
    inspection (gross structure/standoff) even though it is not RGB.
    """
    import numpy as np
    from .utils import write_image
    AnyReader = _require_rosbags()
    cv2 = optional_import("cv2")
    bag_path = Path(bag_path)
    out_dir = ensure_dir(out_dir)
    prefix = (topic or "sonar").strip("/").replace("/", "_")

    saved: list[Path] = []
    with AnyReader([bag_path]) as reader:
        conns = [c for c in reader.connections if c.msgtype == SONAR_MSGTYPE
                 and (topic is None or c.topic == topic)]
        if not conns:
            raise RuntimeError(f"No sonar ({SONAR_MSGTYPE}) topic in {bag_path.name}.")
        conn = max(conns, key=lambda c: c.msgcount)
        LOGGER.info("Extracting sonar from %s (%d msgs)", conn.topic, conn.msgcount)

        idx = 0
        for c, _ts, raw in reader.messages(connections=[conn]):
            if idx % every_n == 0:
                msg = reader.deserialize(raw, c.msgtype)
                data = np.asarray(msg.array_data.data, dtype=np.float32)
                dims = [d.size for d in getattr(msg.array_data.layout, "dim", []) if d.size > 0]
                if len(dims) >= 2:
                    h, w = dims[0], dims[1]
                else:
                    side = int(round(np.sqrt(data.size)))
                    h, w = side, side
                if h * w != data.size:
                    side = int(round(np.sqrt(data.size)))
                    h, w = side, side
                frame = data[:h * w].reshape(h, w)
                # Multibeam returns are sparse and span a wide dynamic range, so
                # a linear min-max stretch renders mostly black. Use a log + 99th
                # -percentile stretch to bring up faint structure.
                logf = np.log1p(np.clip(frame, 0, None))
                hi = float(np.percentile(logf, 99)) or float(logf.max() or 1.0)
                norm = np.clip(logf / (hi + 1e-6) * 255, 0, 255).astype(np.uint8)
                if colormap and cv2 is not None:
                    img = cv2.applyColorMap(norm, cv2.COLORMAP_OCEAN)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    img = np.repeat(norm[..., None], 3, axis=2)
                out_path = out_dir / f"{prefix}_{idx:06d}.png"
                write_image(out_path, img)
                saved.append(out_path)
                if max_frames and len(saved) >= max_frames:
                    break
            idx += 1
    LOGGER.info("Wrote %d sonar frames to %s", len(saved), out_dir)
    return saved


def extract_bag_frames(
    bag_path: str | Path,
    out_dir: str | Path,
    topic: str | None = None,
    every_n: int = 30,
    max_frames: int | None = 40,
    prefix: str | None = None,
) -> list[Path]:
    """Extract sampled RGB frames from a ROS bag's image topic.

    Parameters
    ----------
    topic : str, optional
        Image topic to read. Defaults to the image topic with the most messages.
    every_n : int
        Keep one frame every ``every_n`` messages on that topic.
    max_frames : int, optional
        Stop after writing this many frames (keeps the demo light).
    """
    from .utils import write_image
    AnyReader = _require_rosbags()
    bag_path = Path(bag_path)
    out_dir = ensure_dir(out_dir)
    prefix = prefix or bag_path.stem

    saved: list[Path] = []
    with AnyReader([bag_path]) as reader:
        img_conns = [c for c in reader.connections if c.msgtype in _IMAGE_TYPES]
        if not img_conns:
            raise RuntimeError(f"No image topics found in {bag_path.name}. "
                               f"Topics: {[c.topic for c in reader.connections]}")
        if topic is None:
            chosen = max(img_conns, key=lambda c: c.msgcount)
        else:
            chosen = next((c for c in img_conns if c.topic == topic), None)
            if chosen is None:
                raise ValueError(f"Topic {topic!r} not found or not an image topic.")
        LOGGER.info("Extracting from topic %s (%s, %d msgs)",
                    chosen.topic, chosen.msgtype, chosen.msgcount)

        idx = 0
        for conn, _timestamp, raw in reader.messages(connections=[chosen]):
            if idx % every_n == 0:
                msg = reader.deserialize(raw, conn.msgtype)
                img = _decode_message(msg, conn.msgtype)
                if img is not None:
                    out_path = out_dir / f"{prefix}_{idx:06d}.jpg"
                    write_image(out_path, img)
                    saved.append(out_path)
                    if max_frames and len(saved) >= max_frames:
                        break
            idx += 1

    LOGGER.info("Wrote %d frames to %s", len(saved), out_dir)
    return saved
