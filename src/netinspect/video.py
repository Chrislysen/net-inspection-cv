"""Video frame extraction.

Pulls frames from ``.mp4`` / ``.mov`` (and similar) inspection clips into a
directory of images so the rest of the image pipeline can consume them. Frame
sampling can be controlled by a fixed stride or a target frames-per-second.

Requires OpenCV (``cv2``); raises a clear error if it is unavailable.
"""
from __future__ import annotations

from pathlib import Path

from .utils import ensure_dir, get_logger, require, write_image

LOGGER = get_logger()


def extract_frames(
    video_path: str | Path,
    out_dir: str | Path,
    every_n: int | None = None,
    target_fps: float | None = None,
    max_frames: int | None = None,
    prefix: str | None = None,
) -> list[Path]:
    """Extract frames from a video to ``out_dir`` as JPEG images.

    Parameters
    ----------
    every_n : int, optional
        Keep one frame every ``every_n`` frames. Mutually exclusive with
        ``target_fps``.
    target_fps : float, optional
        Sample at approximately this many frames per second (uses the video's
        reported FPS to compute a stride).
    max_frames : int, optional
        Stop after writing this many frames.
    prefix : str, optional
        Filename prefix; defaults to the video stem.
    """
    cv2 = require("cv2", hint="Install opencv-python-headless to extract video frames.")
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    out_dir = ensure_dir(out_dir)
    prefix = prefix or video_path.stem

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if target_fps and src_fps > 0:
        stride = max(1, int(round(src_fps / target_fps)))
    else:
        stride = max(1, every_n or 1)

    LOGGER.info("Extracting frames from %s (src_fps=%.2f, stride=%d)",
                video_path.name, src_fps, stride)

    saved: list[Path] = []
    idx = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                out_path = out_dir / f"{prefix}_{idx:06d}.jpg"
                write_image(out_path, rgb)
                saved.append(out_path)
                if max_frames and len(saved) >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()

    LOGGER.info("Wrote %d frames to %s", len(saved), out_dir)
    return saved
