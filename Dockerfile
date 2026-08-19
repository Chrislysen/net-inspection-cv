# Net-inspection toolkit — CPU image.
#
#   docker build -t netinspect .
#   docker run --rm -p 8000:8000 -e NETINSPECT_API_KEY="$(openssl rand -hex 24)" netinspect
#   docker run --rm -v "$PWD/mydata:/data" netinspect \
#       netinspect onboard /data/raw --out /data/prepared      # your own footage
#
# Two things the image does NOT do, stated rather than discovered:
#
# * `patchcore` and `permissive` fetch their torchvision backbone from
#   download.pytorch.org on FIRST inference. Nothing pre-warms that cache, so an
#   air-gapped or egress-filtered host can serve `classical`, `yolo` and
#   `anomaly` out of the box but not those two. Bake them in with a
#   `RUN python -c "import torchvision; torchvision.models.resnet18(weights='DEFAULT')"`
#   if that matters to you; it is left out because it adds ~100 MB most
#   deployments never use.
# * The AGPL-free build (EXTRAS below) ships the YOLO weights but not
#   ultralytics, so with the default CMD `/api/ready` correctly returns 503 with
#   "yolo configured but unavailable". Point it at a permissive checkpoint
#   (`--permissive-weights`) or drop the YOLO weights from that image.
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
# STATUS: BUILT AND RUN — Docker 29.7.2, image 4.47 GB. Verified end to end:
# starting it with no key exits 1 with the InsecureBinding reason; with a key it
# comes up, the HEALTHCHECK reaches healthy, /api/health and /api/ready answer
# unauthenticated, /api/version is 401 → 200 → 401 for absent/valid/wrong keys,
# the console serves, the process runs as uid 10001, and all six methods resolve
# (classical, anomaly, patchcore, yolo, permissive, ensemble). CI still does not
# build it, so a change here is unverified until someone runs the build again.
#
# The first build failed in a way no amount of reading would have found: the apt
# list was libglib2.0-0 + libgomp1, which is not enough. `pyproject.toml` asks
# for opencv-python-headless, but ultralytics depends on the full opencv-python,
# so both wheels install and the non-headless one wins `import cv2` — which needs
# libGL and libxcb. cv2 failed to import, taking the classical detector, video
# decoding and YOLO with it, while the container still started and reported
# healthy. See the apt line below.
FROM python:3.12-slim AS base

# These four are load-bearing, and the list was wrong until the image was
# actually built. `pyproject.toml` asks for opencv-python-headless, but
# ultralytics depends on the FULL opencv-python, so both wheels end up installed
# and the non-headless one wins the `import cv2` — which needs libGL and libxcb.
# Without them cv2 fails to import, taking the classical detector, all video
# decoding and YOLO with it, while the service still starts and answers /health.
# libgomp1 is OpenMP for torch; libglib2.0-0 is opencv's own dependency.
#
# The AGPL-free build (EXTRAS=cv,permissive,serve) has no ultralytics and so gets
# genuinely headless opencv, which needs neither libGL nor libxcb — they are
# harmless there, and kept unconditional so one apt line serves both variants.
RUN apt-get update && apt-get install --no-install-recommends -y \
        libglib2.0-0 libgomp1 libgl1 libxcb1 \
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
