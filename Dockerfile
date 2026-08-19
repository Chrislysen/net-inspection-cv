# Net-inspection toolkit — CPU image.
#
#   docker build -t netinspect .
#   docker run --rm -p 8000:8000 -e NETINSPECT_API_KEY="$(openssl rand -hex 24)" netinspect
#   docker run --rm -v "$PWD/mydata:/data" netinspect \
#       netinspect onboard /data/raw --out /data/prepared      # your own footage
#
# The API key is NOT optional here. CMD binds 0.0.0.0 — the only useful bind
# inside a container — and the service refuses to serve a non-loopback address
# without authentication (netinspect.security.check_binding). Omit the key and
# the container exits at startup with that reason on stderr; it does not come up
# unauthenticated. Reach it with `-H "X-API-Key: ..."`, or `?key=` for the
# console. `deploy/docker-compose.yml` wires this up and adds a TLS proxy.
#
# CPU by design: it runs anywhere, and inference is the common case. For
# training, start from a CUDA base (e.g. pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime)
# and install the same extras — nothing else changes.
#
# STATUS: written but NOT BUILT — no container runtime was available in the
# environment where this was authored (Docker, Podman and WSL were all checked),
# so treat it as a starting point rather than a verified artifact. The CI
# pipeline does not build it either. Everything it installs is exercised by the
# test suite, and the startup contract above was verified by running
# check_binding directly — but the layer ordering, the apt package list and the
# CPU-wheel resolution are the parts nobody has confirmed.
FROM python:3.12-slim AS base

# opencv-python-headless still needs libGL's transitive deps for some codecs,
# and git is needed only if you install from a checkout with submodules.
RUN apt-get update && apt-get install --no-install-recommends -y \
        libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Ultralytics otherwise tries to write config into a read-only home.
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# Which extras to install. The default includes `ml`, which pulls Ultralytics
# (AGPL-3.0, viral over a network — and this image serves over a network). For a
# build with no strong copyleft in it at all:
#
#   docker build --build-arg EXTRAS=cv,permissive,serve -t netinspect-permissive .
#   docker run --rm netinspect-permissive netinspect sbom --fail-on copyleft
#
# The second command is the verification, not a formality: it exits non-zero if
# any strong-copyleft package made it into the image.
ARG EXTRAS=cv,ml,serve,export

# Dependency layer first, so source edits do not re-resolve the whole stack.
COPY pyproject.toml README.md ./
COPY src/netinspect/__init__.py src/netinspect/__init__.py
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install ".[${EXTRAS}]"

COPY . .
RUN pip install --no-deps -e .

# Run as a non-root user. Nothing here needs root, and a container that does is
# one escape away from being the host.
RUN useradd --create-home --uid 10001 netinspect \
    && mkdir -p /tmp/ultralytics /tmp/matplotlib \
    && chown -R netinspect:netinspect /app /tmp/ultralytics /tmp/matplotlib
USER netinspect

# Smoke-test the ACTUAL entry point used by CMD, not just the two commands that
# happen not to be wrapped: `netinspect serve` delegates to a script, and that
# delegation was broken while `version` and `doctor` stayed green.
RUN netinspect version \
    && netinspect doctor > /dev/null \
    && netinspect serve --help > /dev/null

EXPOSE 8000

# Liveness for the orchestrator. /api/health stays open even when an API key is
# configured, precisely so this works.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status==200 else 1)"

# Bind 0.0.0.0 so the port is reachable from outside the container — the
# container's own loopback is not. That is precisely the case check_binding
# fails closed on, so NETINSPECT_API_KEY must be set or this exits immediately
# (see the header). Authentication is not a substitute for TLS: terminate it at
# a proxy before this is reachable from anywhere but a trusted network.
CMD ["netinspect", "serve", "--host", "0.0.0.0", "--port", "8000"]
