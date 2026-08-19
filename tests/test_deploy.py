"""The deployment files must agree with the code they deploy.

There is no container runtime in this environment, so the image is never built
and CI cannot catch a Dockerfile that starts and immediately dies. That is not
hypothetical: the documented quickstart was

    docker run --rm -p 8000:8000 netinspect

for a CMD that binds ``0.0.0.0``, while :func:`netinspect.security.check_binding`
refuses any non-loopback bind without authentication. The image would have exited
at startup, every time, and the HEALTHCHECK would never have gone green.

These tests read the deployment files as text and check them against the real
security logic. They are cheap, they need no daemon, and they fail on exactly the
class of mistake that "we'll find out when we build it" leaves in the repo.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from netinspect.security import InsecureBinding, SecurityConfig, check_binding

REPO = Path(__file__).resolve().parents[1]
DOCKERFILE = REPO / "Dockerfile"
COMPOSE = REPO / "deploy" / "docker-compose.yml"
README = REPO / "README.md"

ENV_KEY = "NETINSPECT_API_KEY"


def _text(p: Path) -> str:
    if not p.exists():
        pytest.skip(f"{p.name} is not part of this checkout")
    return p.read_text(encoding="utf-8")


def _instruction(text: str, keyword: str) -> str | None:
    """One Dockerfile instruction, following backslash continuations.

    Regex across the whole file is not good enough: an instruction ends at the
    first line that does not continue, and the comment block that follows is not
    part of it. Reading the comments as if they were the command is how a passing
    test can be looking at the wrong text entirely.
    """
    lines, out, collecting = text.splitlines(), [], False
    for line in lines:
        if not collecting:
            if not line.startswith(keyword):
                continue
            collecting = True
        elif line.lstrip().startswith("#"):
            continue                        # a comment inside a continuation
        out.append(line)
        if not line.rstrip().endswith("\\"):
            break
    return "\n".join(out) if out else None


# --------------------------------------------------------------------------- #
# the contract itself
# --------------------------------------------------------------------------- #
def test_the_container_cmd_binds_a_host_that_requires_a_key():
    """Whatever CMD binds, the docs must be consistent with check_binding.

    Not an assertion that the bind is loopback — inside a container it must not
    be, because the container's own loopback is unreachable from the host. The
    point is that binding 0.0.0.0 has a *consequence*, and the file has to own it.
    """
    text = _text(DOCKERFILE)
    cmd = re.search(r'^CMD \[(.+)\]', text, re.MULTILINE)
    assert cmd, "Dockerfile has no exec-form CMD"
    argv = re.findall(r'"([^"]*)"', cmd.group(1))
    assert "serve" in argv, f"CMD no longer starts the service: {argv}"

    host = argv[argv.index("--host") + 1] if "--host" in argv else "127.0.0.1"
    unauthenticated = SecurityConfig(api_key=None)

    try:
        check_binding(host, unauthenticated)
    except InsecureBinding:
        # The bind needs a key. The file must say so, or the first person to run
        # it gets a container that dies with no clue why.
        assert ENV_KEY in text, (
            f"CMD binds {host}, which check_binding refuses without "
            f"authentication, but the Dockerfile never mentions {ENV_KEY}. "
            "The image would exit at startup and the header would not explain it.")


def test_every_documented_docker_run_that_publishes_a_port_sets_the_key():
    """A copy-pasteable command that cannot work is worse than no command."""
    offenders = []
    for path in (DOCKERFILE, README, COMPOSE):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            # Compound lines are the easy miss: the offending README command was
            # "docker build -t x . ; docker run -p 8000:8000 x", which does not
            # *start* with `docker run` at all. Split first, then match.
            for part in re.split(r'[;&|]+', line.lstrip("# ").strip()):
                cmd = part.strip()
                if not cmd.startswith("docker run"):
                    continue
                if not re.search(r'-p\s*\d+:\d+', cmd):
                    continue                 # not publishing, so not binding for real
                if ENV_KEY not in cmd:
                    offenders.append(f"{path.name}: {cmd}")
    assert not offenders, (
        "these documented commands publish a port but set no API key, so the "
        "service refuses to bind and the container exits:\n  " + "\n  ".join(offenders))


def test_compose_refuses_to_start_without_a_key_rather_than_defaulting_one():
    """``${VAR:?msg}`` fails the run; ``${VAR:-default}`` would ship a known secret."""
    text = _text(COMPOSE)
    assert ENV_KEY in text, "compose does not configure authentication at all"
    assert re.search(rf'\$\{{{ENV_KEY}:\?', text), (
        "compose must use ${VAR:?...} so a missing key aborts the run. A "
        "${VAR:-default} would hand every deployment the same published secret.")


# --------------------------------------------------------------------------- #
# the healthcheck has to be reachable by the thing that runs it
# --------------------------------------------------------------------------- #
def test_the_healthcheck_probes_a_route_that_stays_open_under_auth():
    """The probe carries no credentials, so it must hit an unauthenticated route."""
    text = _text(DOCKERFILE)
    body = _instruction(text, "HEALTHCHECK")
    assert body, "no HEALTHCHECK — an orchestrator cannot tell a wedged container from a healthy one"
    assert ENV_KEY not in body and "X-API-Key" not in body, \
        "the healthcheck cannot authenticate; it must use an open route instead"

    url = re.search(r'https?://[^\s\'"]+', body)
    assert url, f"healthcheck probes no URL: {body!r}"
    path = "/" + url.group(0).split("/", 3)[-1] if url.group(0).count("/") > 2 else "/"
    # Mirrors the exemption in scripts/serve.py.
    assert path.split("?")[0] in ("/api/health", "/api/ready"), (
        f"healthcheck probes {path}, which requires a key once one is set — so "
        "the check fails on exactly the deployments that are configured correctly")


def test_the_healthcheck_allows_slower_startup_than_a_model_load_takes():
    """Weights load lazily, but a cold first request must not trip the probe."""
    body = _instruction(_text(DOCKERFILE), "HEALTHCHECK") or ""
    start = re.search(r'--start-period=(\d+)s', body)
    assert start, "no --start-period: the container is killed while still importing torch"
    assert int(start.group(1)) >= 30, (
        f"start-period is {start.group(1)}s; importing torch and opencv alone "
        "routinely exceeds that on a cold container")


# --------------------------------------------------------------------------- #
# things the image must not do
# --------------------------------------------------------------------------- #
def test_the_image_does_not_run_as_root():
    text = _text(DOCKERFILE)
    users = re.findall(r'^USER\s+(\S+)', text, re.MULTILINE)
    assert users, "no USER directive — the service would run as root"
    assert users[-1] not in ("root", "0"), f"final USER is {users[-1]!r}"


def test_no_secret_is_baked_into_the_image():
    """ENV/ARG lines must not carry a value for anything secret-shaped."""
    text = _text(DOCKERFILE)
    bad = []
    for m in re.finditer(r'^(?:ENV|ARG)\s+(\w*(?:KEY|TOKEN|SECRET|PASSWORD)\w*)\s*=\s*(\S+)',
                         text, re.MULTILINE | re.IGNORECASE):
        bad.append(m.group(0))
    assert not bad, f"secrets baked into image layers (they survive deletion): {bad}"


def test_the_build_context_excludes_the_local_environment():
    """`COPY . .` without these bakes gigabytes of Windows binaries into a Linux image."""
    ignore = REPO / ".dockerignore"
    text = _text(ignore)
    entries = {ln.strip().rstrip("/") for ln in text.splitlines()
               if ln.strip() and not ln.startswith("#")}
    for required in (".venv", ".git", "data", "outputs"):
        assert required in entries, f".dockerignore does not exclude {required}/"
