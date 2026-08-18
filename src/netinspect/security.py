"""Security posture for the inspection service: authentication and input limits.

The console started life as a demo on a laptop and grew into something a farm
might actually run. Two things in it are dangerous the moment it leaves
localhost, and both are fixed here rather than documented as caveats:

**Every route was unauthenticated.** Anyone who could reach the port could run
inference, open a camera, and read frames.

**``POST /api/live/start`` takes a free-form source string** and hands it to
OpenCV. That is a file read of anything the process can open, and an outbound
request to anywhere the host can reach — including cloud instance metadata at
``169.254.169.254``, which is the standard route from "can hit an internal
endpoint" to "has your credentials". An allowlist is the only correct control
here; blocklists of bad hosts are a game you lose.

Design rule: **secure by default, without making local development miserable.**

* No API key set and bound to loopback → runs, with a warning. This is the demo.
* No API key set and bound to anything else → **refuses to start**. It is not
  possible to accidentally publish an unauthenticated service to a network,
  which is the failure that actually happens.
* API key set → every mutating and data-returning route requires it.

Configuration is environment-only. A key in a config file gets committed.
"""
from __future__ import annotations

import fnmatch
import hmac
import ipaddress
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .utils import get_logger

LOGGER = get_logger()

ENV_API_KEY = "NETINSPECT_API_KEY"
ENV_LIVE_ALLOW = "NETINSPECT_LIVE_ALLOW"
ENV_MEDIA_ROOT = "NETINSPECT_MEDIA_ROOT"
ENV_CORS = "NETINSPECT_CORS_ORIGINS"
ENV_MAX_UPLOAD = "NETINSPECT_MAX_UPLOAD_MB"
ENV_MAX_PIXELS = "NETINSPECT_MAX_PIXELS"
ENV_MAX_CONCURRENCY = "NETINSPECT_MAX_CONCURRENCY"

# A 16k x 16k PNG decompresses to ~1 GB of RGB. Pillow will happily do it, and
# one request becomes an out-of-memory kill. This is the decompression-bomb
# ceiling, applied before any pixel is touched.
DEFAULT_MAX_PIXELS = 50_000_000            # ~50 MP, far above any inspection frame
DEFAULT_MAX_UPLOAD_BYTES = 16 * 1024 * 1024

LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def _is_loopback(host: str) -> bool:
    if host in LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass
class SecurityConfig:
    """Everything that decides what this service will accept."""
    api_key: str | None = None
    # Glob patterns a live source must match. Empty means "cameras and files
    # under media_root only" — deliberately restrictive, because the default for
    # a network-reachable stream opener has to be no.
    live_allow: tuple[str, ...] = ()
    media_root: Path | None = None
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES
    max_pixels: int = DEFAULT_MAX_PIXELS
    cors_origins: tuple[str, ...] = ()
    max_concurrency: int = 4
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_key)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "SecurityConfig":
        env = os.environ if env is None else env

        def _int(name, default):
            raw = env.get(name)
            try:
                return int(raw) if raw else default
            except ValueError:
                LOGGER.warning("%s=%r is not an integer; using %d", name, raw, default)
                return default

        allow = tuple(p.strip() for p in env.get(ENV_LIVE_ALLOW, "").split(",") if p.strip())
        origins = tuple(p.strip() for p in env.get(ENV_CORS, "").split(",") if p.strip())
        root = env.get(ENV_MEDIA_ROOT)
        return cls(
            api_key=env.get(ENV_API_KEY) or None,
            live_allow=allow,
            media_root=Path(root).resolve() if root else None,
            max_upload_bytes=_int(ENV_MAX_UPLOAD, DEFAULT_MAX_UPLOAD_BYTES // (1024 * 1024)) * 1024 * 1024
            if env.get(ENV_MAX_UPLOAD) else DEFAULT_MAX_UPLOAD_BYTES,
            max_pixels=_int(ENV_MAX_PIXELS, DEFAULT_MAX_PIXELS),
            cors_origins=origins,
            max_concurrency=max(1, _int(ENV_MAX_CONCURRENCY, 4)),
        )

    def describe(self) -> str:
        return (f"auth={'on' if self.auth_enabled else 'OFF'} · "
                f"live_allow={list(self.live_allow) or 'cameras/media-root only'} · "
                f"media_root={self.media_root or 'unset'} · "
                f"max_upload={self.max_upload_bytes // (1024*1024)}MB · "
                f"max_pixels={self.max_pixels:,} · "
                f"cors={list(self.cors_origins) or 'same-origin only'} · "
                f"max_concurrency={self.max_concurrency}")


class InsecureBinding(RuntimeError):
    """Raised when the service would be published without authentication."""


def check_binding(host: str, cfg: SecurityConfig) -> None:
    """Refuse to expose an unauthenticated service beyond loopback.

    A warning would not be enough. "I'll add auth before we deploy" is the
    sentence that precedes an open inference endpoint on a public IP, so this
    fails closed instead.
    """
    if cfg.auth_enabled or _is_loopback(host):
        if not cfg.auth_enabled:
            LOGGER.warning(
                "No %s set — running WITHOUT authentication on %s. Fine for local "
                "use; set the variable before binding anywhere else.", ENV_API_KEY, host)
        return
    raise InsecureBinding(
        f"Refusing to bind {host} without authentication.\n"
        f"Set {ENV_API_KEY} to a secret, or bind 127.0.0.1 for local use.\n"
        "Every route would otherwise be open to anyone who can reach the port, "
        "including the one that opens a camera or stream.")


def check_api_key(supplied: str | None, cfg: SecurityConfig) -> bool:
    """Constant-time key comparison; always true when auth is disabled."""
    if not cfg.auth_enabled:
        return True
    if not supplied:
        return False
    return hmac.compare_digest(str(supplied), str(cfg.api_key))


# --------------------------------------------------------------------------- #
# Live source validation — the SSRF / arbitrary-read surface
# --------------------------------------------------------------------------- #
class SourceRejected(ValueError):
    """A live source that policy does not permit."""


_CAMERA_INDEX = re.compile(r"^\d{1,2}$")

# Networks that must never be reachable from a client-supplied URL. Cloud
# instance metadata lives on link-local and is the usual pivot from SSRF to
# credential theft; the rest are internal ranges a farm's LAN would use.
BLOCKED_NETWORKS = tuple(ipaddress.ip_network(n) for n in (
    "169.254.0.0/16",       # link-local, incl. 169.254.169.254 metadata
    "127.0.0.0/8",
    "10.0.0.0/8",
    "172.16.0.0/12",
    "192.168.0.0/16",
    "::1/128",
    "fc00::/7",
    "fe80::/10",
))


def _url_matches(parsed, pattern: str) -> bool:
    """Component-wise glob match for a stream URL.

    Scheme must match exactly; host and path are matched separately so a
    wildcard cannot slide across the "/" that separates them.
    """
    pp = urlparse(pattern)
    if not pp.scheme or not pp.netloc:
        return False                                   # not a URL pattern
    if pp.scheme != parsed.scheme:
        return False
    host, phost = (parsed.hostname or ""), (pp.hostname or "")
    if not fnmatch.fnmatch(host, phost):
        return False
    if pp.port is not None and parsed.port != pp.port:
        return False
    return fnmatch.fnmatch(parsed.path or "/", pp.path or "*")


def _host_is_blocked(host: str) -> bool:
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False                # a name; the allowlist is what permits it
    return any(ip in net for net in BLOCKED_NETWORKS)


def validate_live_source(source: str | int, cfg: SecurityConfig) -> str | int:
    """Decide whether a client may open this source. Default deny.

    Accepts, in order of how a user actually means them:

    * ``"0"``–``"99"`` — a locally attached camera index.
    * a path under ``media_root`` — a recorded clip standing in for a camera.
    * a URL or path matching one of the ``live_allow`` glob patterns.

    Everything else is refused, and the refusal names the setting that would
    permit it, because a security control nobody can configure gets disabled.
    """
    raw = str(source).strip()
    if not raw:
        raise SourceRejected("Empty source.")

    if _CAMERA_INDEX.match(raw):
        return int(raw)

    parsed = urlparse(raw)
    if parsed.scheme in {"http", "https", "rtsp", "rtmp", "udp", "tcp"}:
        # Match scheme, host and path SEPARATELY. fnmatch's "*" happily crosses
        # "/", so a single whole-string match lets
        #   rtsp://cam-*.farm.local/*
        # be satisfied by
        #   rtsp://cam-evil.example.com/x.farm.local/y
        # — the attacker's host, with the expected suffix pushed into the path.
        # Splitting first confines the wildcard to one component, because a
        # hostname cannot contain a slash.
        matched = any(_url_matches(parsed, pat) for pat in cfg.live_allow)
        if not matched:
            raise SourceRejected(
                f"Stream URL {raw!r} is not allowed. Add a pattern to "
                f"{ENV_LIVE_ALLOW} (e.g. 'rtsp://cam-*.farm.local/*') to permit it.")
        host = parsed.hostname or ""
        if _host_is_blocked(host):
            raise SourceRejected(
                f"Refusing to open {raw!r}: {host} is a loopback, link-local or "
                "private address. Client-supplied URLs must not be able to reach "
                "internal services or cloud instance metadata.")
        return raw

    if parsed.scheme and len(parsed.scheme) > 1:
        # file://, gopher://, and friends. len>1 avoids treating a Windows drive
        # letter ("C:/...") as a scheme.
        raise SourceRejected(f"Unsupported scheme in {raw!r}.")

    # A filesystem path. Whole-string globbing is fine here: paths are what the
    # pattern is describing, and containment under media_root is checked below.
    if any(fnmatch.fnmatch(raw, pat) for pat in cfg.live_allow):
        return raw
    if cfg.media_root is None:
        raise SourceRejected(
            f"File sources are not allowed: {ENV_MEDIA_ROOT} is not set. "
            "Set it to the directory holding your clips.")
    try:
        resolved = Path(raw).resolve()
    except OSError as exc:
        raise SourceRejected(f"Unusable path {raw!r}: {exc}")
    if not resolved.is_relative_to(cfg.media_root):
        raise SourceRejected(
            f"Refusing to open {resolved} — outside {ENV_MEDIA_ROOT} "
            f"({cfg.media_root}).")
    return str(resolved)


# --------------------------------------------------------------------------- #
# Decoding limits
# --------------------------------------------------------------------------- #
def apply_decoder_limits(cfg: SecurityConfig) -> None:
    """Cap what the image decoder will expand, before anything is decoded."""
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = int(cfg.max_pixels)
    except Exception as exc:                       # pragma: no cover - defensive
        LOGGER.debug("Could not set Pillow pixel limit: %s", exc)


def check_image_size(width: int, height: int, cfg: SecurityConfig) -> None:
    """Reject an image whose pixel count would blow up memory."""
    if width <= 0 or height <= 0:
        raise ValueError("Image has no pixels.")
    if width * height > cfg.max_pixels:
        raise ValueError(
            f"Image is {width}x{height} = {width * height:,} pixels, over the "
            f"{cfg.max_pixels:,} limit. Raise {ENV_MAX_PIXELS} if this is genuinely "
            "your frame size.")


__all__ = ["SecurityConfig", "InsecureBinding", "SourceRejected",
           "check_binding", "check_api_key", "validate_live_source",
           "apply_decoder_limits", "check_image_size",
           "ENV_API_KEY", "ENV_LIVE_ALLOW", "ENV_MEDIA_ROOT", "ENV_CORS",
           "BLOCKED_NETWORKS", "DEFAULT_MAX_PIXELS", "DEFAULT_MAX_UPLOAD_BYTES"]
