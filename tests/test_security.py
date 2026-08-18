"""Tests for the service's security posture.

A security control is only worth having if it fails closed, so most of these
assert a refusal. The two that matter most: an unauthenticated service must not
be bindable to a network interface, and a client-supplied stream source must not
be able to reach the filesystem or an internal address.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from netinspect import security as S


def _cfg(**kw):
    return S.SecurityConfig(**kw)


# --------------------------------------------------------------------------- #
# binding
# --------------------------------------------------------------------------- #
def test_loopback_without_a_key_is_allowed_for_local_use():
    S.check_binding("127.0.0.1", _cfg())          # must not raise
    S.check_binding("localhost", _cfg())
    S.check_binding("::1", _cfg())


def test_binding_a_network_interface_without_a_key_is_refused():
    """The failure that actually happens: shipping the demo to 0.0.0.0."""
    for host in ("0.0.0.0", "192.168.1.10", "10.0.0.5", "::"):
        with pytest.raises(S.InsecureBinding):
            S.check_binding(host, _cfg())


def test_binding_a_network_interface_with_a_key_is_allowed():
    S.check_binding("0.0.0.0", _cfg(api_key="s3cret"))


def test_the_refusal_names_the_variable_that_would_fix_it():
    with pytest.raises(S.InsecureBinding) as e:
        S.check_binding("0.0.0.0", _cfg())
    assert S.ENV_API_KEY in str(e.value)


# --------------------------------------------------------------------------- #
# api key
# --------------------------------------------------------------------------- #
def test_any_key_passes_when_auth_is_disabled():
    assert S.check_api_key(None, _cfg()) is True


def test_the_right_key_passes_and_a_wrong_one_does_not():
    cfg = _cfg(api_key="correct-horse")
    assert S.check_api_key("correct-horse", cfg) is True
    assert S.check_api_key("wrong", cfg) is False
    assert S.check_api_key(None, cfg) is False
    assert S.check_api_key("", cfg) is False


def test_a_prefix_of_the_key_does_not_pass():
    cfg = _cfg(api_key="correct-horse")
    assert S.check_api_key("correct", cfg) is False
    assert S.check_api_key("correct-horse-battery", cfg) is False


# --------------------------------------------------------------------------- #
# live sources — SSRF and arbitrary file access
# --------------------------------------------------------------------------- #
def test_a_camera_index_is_allowed_and_returned_as_an_int():
    assert S.validate_live_source("0", _cfg()) == 0
    assert S.validate_live_source("2", _cfg()) == 2


def test_a_stream_url_is_refused_by_default():
    with pytest.raises(S.SourceRejected):
        S.validate_live_source("rtsp://camera.example.com/stream", _cfg())


def test_an_allowlisted_stream_url_is_permitted():
    cfg = _cfg(live_allow=("rtsp://cam-*.farm.local/*",))
    assert S.validate_live_source("rtsp://cam-3.farm.local/main", cfg)


def test_cloud_metadata_is_refused_even_if_the_pattern_would_match():
    """The classic SSRF pivot: 169.254.169.254 hands out cloud credentials."""
    cfg = _cfg(live_allow=("http://*",))
    with pytest.raises(S.SourceRejected) as e:
        S.validate_live_source("http://169.254.169.254/latest/meta-data/", cfg)
    assert "metadata" in str(e.value).lower() or "link-local" in str(e.value).lower()


@pytest.mark.parametrize("host", [
    "127.0.0.1", "10.1.2.3", "192.168.0.9", "172.16.5.5", "[::1]",
])
def test_internal_addresses_are_refused_even_when_allowlisted(host):
    cfg = _cfg(live_allow=("http://*", "rtsp://*"))
    with pytest.raises(S.SourceRejected):
        S.validate_live_source(f"http://{host}/x", cfg)


def test_a_public_host_passes_when_allowlisted():
    cfg = _cfg(live_allow=("rtsp://*",))
    assert S.validate_live_source("rtsp://203.0.113.7/stream", cfg)


def test_a_file_path_is_refused_when_no_media_root_is_configured():
    with pytest.raises(S.SourceRejected) as e:
        S.validate_live_source("C:/Windows/win.ini", _cfg())
    assert S.ENV_MEDIA_ROOT in str(e.value)


def test_a_file_under_the_media_root_is_allowed(tmp_path):
    clip = tmp_path / "clips" / "pass1.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"x")
    cfg = _cfg(media_root=tmp_path.resolve())
    assert Path(S.validate_live_source(str(clip), cfg)) == clip.resolve()


def test_a_file_outside_the_media_root_is_refused(tmp_path):
    root = tmp_path / "clips"
    root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("x")
    with pytest.raises(S.SourceRejected):
        S.validate_live_source(str(outside), _cfg(media_root=root.resolve()))


def test_traversal_out_of_the_media_root_is_refused(tmp_path):
    root = tmp_path / "clips"
    root.mkdir()
    (tmp_path / "secret.txt").write_text("x")
    with pytest.raises(S.SourceRejected):
        S.validate_live_source(str(root / ".." / "secret.txt"),
                               _cfg(media_root=root.resolve()))


def test_exotic_schemes_are_refused():
    for src in ("file:///etc/passwd", "gopher://x/", "ftp://x/y"):
        with pytest.raises(S.SourceRejected):
            S.validate_live_source(src, _cfg(live_allow=("*",)))


def test_an_empty_source_is_refused():
    with pytest.raises(S.SourceRejected):
        S.validate_live_source("   ", _cfg())


# --------------------------------------------------------------------------- #
# decoding limits
# --------------------------------------------------------------------------- #
def test_a_normal_frame_passes_the_pixel_check():
    S.check_image_size(1920, 1080, _cfg())


def test_a_decompression_bomb_is_refused():
    with pytest.raises(ValueError) as e:
        S.check_image_size(30000, 30000, _cfg())
    assert "limit" in str(e.value)


def test_a_zero_sized_image_is_refused():
    with pytest.raises(ValueError):
        S.check_image_size(0, 100, _cfg())


def test_applying_decoder_limits_sets_the_pillow_ceiling():
    pytest.importorskip("PIL")
    from PIL import Image
    original = Image.MAX_IMAGE_PIXELS
    try:
        S.apply_decoder_limits(_cfg(max_pixels=1234))
        assert Image.MAX_IMAGE_PIXELS == 1234
    finally:
        Image.MAX_IMAGE_PIXELS = original


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
def test_config_reads_the_environment():
    cfg = S.SecurityConfig.from_env({
        S.ENV_API_KEY: "k",
        S.ENV_LIVE_ALLOW: "rtsp://a/*, rtsp://b/*",
        S.ENV_CORS: "https://ops.example.com",
        S.ENV_MAX_CONCURRENCY: "8",
    })
    assert cfg.api_key == "k"
    assert cfg.live_allow == ("rtsp://a/*", "rtsp://b/*")
    assert cfg.cors_origins == ("https://ops.example.com",)
    assert cfg.max_concurrency == 8


def test_config_defaults_are_safe_when_the_environment_is_empty():
    cfg = S.SecurityConfig.from_env({})
    assert cfg.api_key is None
    assert cfg.live_allow == ()          # default deny for streams
    assert cfg.media_root is None        # default deny for files
    assert cfg.cors_origins == ()        # same-origin only


def test_a_nonsense_numeric_setting_falls_back_instead_of_crashing():
    cfg = S.SecurityConfig.from_env({S.ENV_MAX_CONCURRENCY: "banana"})
    assert cfg.max_concurrency >= 1


def test_describe_states_whether_auth_is_on():
    assert "auth=OFF" in S.SecurityConfig().describe()
    assert "auth=on" in S.SecurityConfig(api_key="x").describe()


# --------------------------------------------------------------------------- #
# release identity
# --------------------------------------------------------------------------- #
def test_one_version_string_everywhere():
    """A deployment must be identifiable. These four had already drifted: the
    package said 0.1.0 while the HTTP service reported 0.4.0."""
    import importlib.metadata as md

    import netinspect
    from netinspect import cli

    assert netinspect.__version__ == cli.VERSION
    assert md.version("net-inspection-cv") == netinspect.__version__
