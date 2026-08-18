# Net-inspection toolkit — CPU image.
#
#   docker build -t netinspect .
#   docker run --rm -p 8000:8000 netinspect                    # the console
#   docker run --rm -v "$PWD/mydata:/data" netinspect \
#       netinspect onboard /data/raw --out /data/prepared      # your own footage
#
# CPU by design: it runs anywhere, and inference is the common case. For
# training, start from a CUDA base (e.g. pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime)
# and install the same extras — nothing else changes.
#
# STATUS: written but NOT BUILT — no Docker daemon was available in the
# environment where this was authored, so treat it as a starting point that has
# never been executed rather than a verified artifact. The CI pipeline does not
# build it either. Everything it installs is exercised by the test suite; the
# layer ordering and the apt package list are the parts nobody has confirmed.
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

# Dependency layer first, so source edits do not re-resolve the whole stack.
COPY pyproject.toml README.md ./
COPY src/netinspect/__init__.py src/netinspect/__init__.py
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install ".[cv,ml,serve,export]"

COPY . .
RUN pip install --no-deps -e .

# Fails the build if the package cannot even introspect itself.
RUN netinspect version && netinspect doctor > /dev/null

EXPOSE 8000

# Bind 0.0.0.0 so the port is reachable from outside the container. The service
# is UNAUTHENTICATED — put it behind an authenticating proxy before exposing it
# anywhere but a laptop, and note that /api/live/start takes an arbitrary source
# string, so anyone who can reach the port can make the server open a stream.
CMD ["netinspect", "serve", "--host", "0.0.0.0", "--port", "8000"]
