# Production-readiness scorecard (self-critical)

An honest 0–10 rating of each dimension of "deployable industry product",
written to be **critical of my own work**. The headline: the *engineering* is
genuinely production-shaped; the *detector* is not validated and cannot be
without real labelled damage. Don't average these into one number — the low one
gates real deployment.

| Dimension | Score | Honest assessment |
|---|---:|---|
| **Validated detection accuracy** | **1/10** | The hard gate. Every metric is on *synthetic* damage from one generator. There is **zero** validated real-world damage-detection performance. No amount of engineering changes this; it needs real labelled damage. This single 1 is why the product is not deployable. |
| Reproducibility | 8/10 | Pinned lockfile, deterministic seeds, ruff + pre-commit, CI (lint + 3.11/3.12 tests), one-command dataset/plot regeneration, **a built and verified container**. Missing: full data/DVC versioning, a container-pinned build digest, CI that builds the image. |
| Serving / API robustness | 8/10 | FastAPI with path-traversal-safe access, upload size/type validation, **API-key auth that fails closed on any non-loopback bind**, bounded inference concurrency and stream viewers, blocking work kept off the event loop, structured logging + request IDs, `/health` `/ready` `/metrics`, a no-leak exception handler. Missing: rate limiting and TLS (both delegated to the reverse proxy), async batching, multi-worker scaling. |
| Deployable inference | 7/10 | Torch-free **ONNX** path (`onnx_infer.py`) with **verified parity** to the torch path — runs with only onnxruntime+numpy+cv2. Missing: the on-device **TensorRT FP16/INT8** build + measured on-device latency (needs the hardware). |
| Streaming / operational shape | 6/10 | `stream_inspect.py` consumes video/bag/dir, applies temporal confirmation, emits one alert per *new* confirmed track + heartbeats with throughput. Missing: real message-bus/event-sink integration, backpressure, reconnect. |
| Observability | 5/10 | Prometheus-style `/metrics` (latency, counts, errors, uptime) + structured logs. Missing: a real metrics backend, dashboards, model-output/drift monitoring, alerting wired to on-call. |
| Testing | 8/10 | **647 tests across 36 files**: unit (geometry, metrics, data, compose, temporal, water physics), service integration (auth boundary, readiness, metrics cardinality), deployment/container contract, licence + SBOM classification, ONNX parity, plus 20 headless-renderer checks. Missing: coverage gating, property-based tests, load tests, a full end-to-end integration job in CI. |
| Evaluation rigor | 9/10 | Class-agnostic P/R/F1, COCO mAP@[.5:.95], per-class metrics, FROC, image-level, an **adversarial "is it cheating?"** suite that evaluates FP on real undamaged net and *caught a regression in my own model*. This is the strongest dimension. |
| Documentation | 8/10 | README, technical report, model card, this scorecard, deployment runbook, data-collection protocol, a research decision log, and a code-grounded defense Q&A. Missing: API reference autodoc. |
| MLOps lifecycle | 4/10 | Run manifests, committed models + provenance NOTICE, a documented active-learning/retraining path. Missing: a real model registry, automated retraining, a feature/label store, CD. |
| Security / compliance | 7/10 | Input validation, no path traversal, no stack-trace leakage, **API-key auth**, **default-deny stream-source allowlist** (private/loopback/link-local refused), decompression-bomb limits, non-root container, **a CycloneDX SBOM with a copyleft gate** (`netinspect sbom --fail-on copyleft`), documented AGPL/CC-BY-SA provenance and an AGPL-free build. Missing: authz/roles, secrets management beyond env vars, automated dependency scanning, a written threat model. |

## The one-line verdict
**As an honest *prototype/research framework*: ~9/10.** **As a *deployable
product an operator could put in the water*: ~3/10**, gated entirely by the absence
of validated real-damage performance. The work here maximises every dimension
that does **not** require faking that — and stops, loudly, at the one that does.

## What would move the gate (and only this)
1. A few hundred **labelled real-damage frames** across sites/seasons/cameras.
2. A labelling guideline + an agreed false-positive/false-negative operating point.
3. Re-run the *existing* `train_yolo.py` + `compare_methods.py` + `adversarial_eval.py`
   on that data, evaluated by **held-out site**. Then — and only then — the 1/10
   becomes a real number, and the product score can rise.

See [`DATA_COLLECTION.md`](DATA_COLLECTION.md) for exactly what to collect.
