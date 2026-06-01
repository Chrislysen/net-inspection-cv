# Containerised FastAPI inference service for the net-inspection prototype.
#
# Build:  docker build -t net-inspection-cv .
# Run:    docker run -p 8000:8000 net-inspection-cv
#         # then: curl -F file=@frame.jpg "http://localhost:8000/predict?method=yolo"
#
# Ships with the committed prototype models (classical needs none; anomaly and
# YOLO use models/). This serves a PROTOTYPE — predictions are not validated on
# real damage and require human review.
FROM python:3.12-slim

# OpenCV runtime needs libGL / glib even with the headless wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for better layer caching.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[cv,ml,serve,export]"

# App code + web console + committed prototype models.
COPY scripts ./scripts
COPY configs ./configs
COPY web ./web
COPY models ./models

EXPOSE 8000
# Liveness/readiness: the service exposes /api/ready (503 until ready).
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/ready').status==200 else 1)"
CMD ["python", "scripts/serve.py", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--anomaly-model", "models/anomaly_normal_net", \
     "--patchcore-model", "models/patchcore_normal_net", \
     "--yolo-weights", "models/yolo_damage_v1.pt"]
