"""Streamlit viewer for net-inspection predictions.

Browse frames and compare methods (classical / anomaly / YOLO) interactively,
with live threshold controls and per-frame stats.

Run
---
    streamlit run streamlit_app.py

Notes
-----
Prototype tool. Models are synthetic/proxy-trained and NOT validated on real
damage — predictions are for exploration and must be human-reviewed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from netinspect.classical_baseline import ClassicalConfig  # noqa: E402
from netinspect.inference import NetInspector  # noqa: E402
from netinspect.utils import list_images, read_image  # noqa: E402
from netinspect.visualize import overlay_boxes  # noqa: E402

st.set_page_config(page_title="net-inspection-cv", layout="wide")

DEFAULT_DIRS = [
    "data/processed/solaqua_frames",
    "data/processed/solaqua_frames_dense",
    "data/processed/real_composite/images/test",
    "data/sample/images",
]


@st.cache_data(show_spinner=False)
def _load_image(path: str) -> np.ndarray:
    return read_image(path)


def main() -> None:
    st.title("🪝 net-inspection-cv — prototype viewer")
    st.caption("Aquaculture net damage inspection. Prototype: synthetic/proxy-trained "
               "models, **not validated on real damage** — predictions need human review.")

    sb = st.sidebar
    sb.header("Data")
    existing = [d for d in DEFAULT_DIRS if Path(d).exists() and list_images(d)]
    frame_dir = sb.selectbox("Frame directory", existing or DEFAULT_DIRS,
                             help="Folders found under the repo.")
    custom = sb.text_input("…or a custom directory", "")
    frame_dir = custom or frame_dir

    images = list_images(frame_dir)
    if not images:
        st.warning(f"No images in `{frame_dir}`. Generate or download frames first "
                   "(see README — make_demo_data.py / fetch_solaqua.py).")
        return

    sb.header("Models")
    anomaly_model = sb.text_input("Anomaly model prefix", "outputs/anomaly/model")
    yolo_weights = sb.text_input("YOLO weights", "runs/detect/train/weights/best.pt")
    gate = sb.slider("Classical texture gate (lower = fewer false alarms)",
                     0.5, 1.2, 0.82, 0.02)
    conf = sb.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)

    inspector = NetInspector(
        classical_cfg=ClassicalConfig(max_region_density_ratio=gate),
        anomaly_model_path=anomaly_model or None,
        yolo_weights=yolo_weights or None,
    )
    methods = inspector.available_methods()
    method = sb.radio("Method", methods, horizontal=False)

    idx = st.slider("Frame", 0, len(images) - 1, 0)
    path = images[idx]
    img = _load_image(str(path))

    result = inspector.predict(img, method=method, conf=conf)
    vis = result.heatmap if result.heatmap is not None else overlay_boxes(img, preds=result.boxes)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Original")
        st.image(img, use_container_width=True)
    with c2:
        st.subheader(f"{method} — {len(result.boxes)} region(s)")
        st.image(vis, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Detections", len(result.boxes))
    m2.metric("Latency (ms)", f"{result.elapsed_ms:.0f}")
    m3.metric("Frame", f"{idx + 1}/{len(images)}")
    st.caption(f"`{path.name}` · method=`{method}` · conf≥{conf}")

    if result.boxes:
        st.dataframe(
            [{"class": b.class_name, "score": round(b.score, 3),
              "x1": int(b.x1), "y1": int(b.y1), "x2": int(b.x2), "y2": int(b.y2)}
             for b in result.boxes],
            use_container_width=True,
        )
    if method == "anomaly":
        st.info("Anomaly view: heatmap shows deviation from *normal net*, not "
                "confirmed damage (biofouling/lighting also light up).")


if __name__ == "__main__":
    main()
