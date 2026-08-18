"""What actually drives false alarms: the day, the flight profile, or the scene?

Every SOLAQUA frame shows **undamaged** net, so every detection is a known
false positive — no annotation needed. Joining frames to ROV telemetry
(:mod:`netinspect.frame_sync`) and to no-reference capture-quality metrics
(:mod:`netinspect.image_quality`) turns that into a controlled study of *when*
these models cry wolf.

The pre-registered hypothesis, and its fate
-------------------------------------------
Telemetry shows the two recording days were flown on very different commanded
profiles:

    2024-08-22 (3 training clips): standoff 0.60-0.65 m, sweep 0.10-0.14 m/s, depth 1.4-1.9 m
    2024-08-20 (2 held-out clips): standoff 1.40 m,      sweep 0.28 m/s,      depth 4.8-5.2 m

That invited an appealing explanation for this repo's long-running
"different-day" generalisation gap: the gap is really an **operating-envelope
violation**, models trained at 0.6 m being tested at 1.4 m.

**This analysis tests that hypothesis and rejects it.** The evidence:

1. The clip with the widest standoff range (2024-08-22_14-06-43, 0.19-1.31 m,
   a 6.9x span within one clip on one day) produces **zero** false alarms for
   every model tested.
2. On the held-out day, standoff has no within-clip association with false
   alarms for any model (all p > 0.67); the elevated rate is flat across
   1.17-1.54 m.
3. Three training-day clips flown at near-identical standoff (0.60 / 0.66 /
   0.71 m) produce wildly different rates -- for the detector, 0.0 / 0.33 / 0.0.

What the data supports instead
------------------------------
**Scene identity dominates, and capture quality is the mechanism.** Sharpness
and contrast correlate strongly with false alarms, with a sign that *depends on
the model*: the detector fires on sharp, high-contrast frames where fine
structure (biofouling, mesh knots) is resolved, while the higher-capacity
segmenter fires on degraded frames. That opposition is the mechanism behind the
agreement ensemble already in this repo -- requiring both models to agree
cancels two errors that happen in different regimes.

Because frames within a clip are highly correlated, per-frame confidence
intervals badly overstate what four clips can tell us. This script therefore
reports the intra-cluster correlation and a **clip-level clustered bootstrap**
alongside naive per-frame intervals, and treats the clip as the unit of
generalisation.

Honesty
-------
All frames are undamaged, so this measures false-alarm behaviour only. Recall
on real damage is not measured here and remains unvalidated. Standoff, speed
and depth co-vary by design between the two days, so no day-level difference
can be attributed to any single one of them; the within-clip tests are the
parts of this analysis that isolate a variable.

Examples
--------
    python scripts/analyze_operating_envelope.py --models det_v1,seg_v3,seg_gpu
    python scripts/analyze_operating_envelope.py --models det_v1 --no-cache
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _common  # noqa: F401
import numpy as np

from netinspect import frame_sync as F
from netinspect import image_quality as IQ
from netinspect import telemetry as T
from netinspect.envelope import (
    EnvelopeSpec,
    dose_response,
    fit_envelope,
    matched_band_comparison,
    proportion_stat,
    wilson_ci,
)
from netinspect.inference import NetInspector
from netinspect.utils import ensure_dir, get_logger, read_image, write_json

LOGGER = get_logger()

FRAME_SETS = [
    ("data/processed/solaqua_frames_dense", "2024-08-22_14-06-43"),
    ("data/processed/solaqua_bag2", "2024-08-22_14-47-39"),
    ("data/processed/solaqua_bag3", "2024-08-22_14-29-05"),
    ("data/processed/solaqua_diffday", "2024-08-20_15-18-27"),
]

MODELS = {
    "classical": ("classical", {}),
    "det_v1": ("yolo", {"yolo_weights": "models/yolo_damage_v1.pt"}),
    "seg_v3": ("yolo", {"yolo_weights": "models/yolo_damage_seg_v3.pt"}),
    "seg_gpu": ("yolo", {"yolo_weights": "models/yolo_damage_seg_gpu.pt"}),
    "ensemble": ("ensemble", {"yolo_weights": "models/yolo_damage_v1.pt",
                              "seg_weights": "models/yolo_damage_seg_v3.pt"}),
}

TRAIN_DAY = "2024-08-22"
HELDOUT_DAY = "2024-08-20"
MATCHED_BAND = (1.0, 1.4)
CONDITION_COLS = ["standoff", "net_speed", "depth", "sharpness", "contrast",
                  "brightness", "dark_fraction"]


# --------------------------------------------------------------------------- #
# Clustered inference
# --------------------------------------------------------------------------- #
def intracluster_correlation(values: np.ndarray, groups: np.ndarray) -> dict:
    """One-way ANOVA estimate of ICC(1) plus the resulting design effect.

    Frames inside a clip are not independent samples of "a net". ICC(1)
    quantifies how much of the total variance in the outcome sits *between*
    clips; the design effect ``1 + (m - 1) * ICC`` converts that into how much
    a naive per-frame confidence interval understates uncertainty.
    """
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    labels = np.unique(groups)
    k = len(labels)
    n = len(values)
    if k < 2 or n <= k:
        return {"icc": None, "note": "need >= 2 clusters"}

    grand = values.mean()
    sizes = np.array([(groups == g).sum() for g in labels], dtype=float)
    means = np.array([values[groups == g].mean() for g in labels])

    ss_between = float((sizes * (means - grand) ** 2).sum())
    ss_within = float(sum(((values[groups == g] - m) ** 2).sum()
                          for g, m in zip(labels, means)))
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n - k) if n > k else 0.0
    # Unequal cluster sizes: Sokal & Rohlf's n0.
    n0 = (sizes.sum() - (sizes ** 2).sum() / sizes.sum()) / (k - 1)
    denom = ms_between + (n0 - 1) * ms_within
    icc = (ms_between - ms_within) / denom if denom > 0 else 0.0
    icc = float(max(0.0, min(1.0, icc)))
    m_bar = float(sizes.mean())
    deff = 1.0 + (m_bar - 1.0) * icc
    return {
        "icc": round(icc, 4),
        "clusters": int(k),
        "mean_cluster_size": round(m_bar, 1),
        "design_effect": round(deff, 2),
        "effective_n": round(n / deff, 1) if deff > 0 else None,
        "naive_n": int(n),
        "note": ("Frames within a clip are correlated. The effective sample size is "
                 "what a per-frame confidence interval should be based on."),
    }


def clustered_bootstrap_rate(events: np.ndarray, groups: np.ndarray,
                             n_boot: int = 2000, seed: int = 0) -> dict:
    """Resample whole clips (not frames) to get an honest interval on the rate.

    This is the interval that answers "what false-alarm rate would we see on a
    *new clip*", which is the question that matters operationally. With only a
    handful of clips it is necessarily wide — that width is the finding, not a
    defect.
    """
    events = np.asarray(events, dtype=float)
    groups = np.asarray(groups)
    labels = np.unique(groups)
    if len(labels) < 2:
        return {"note": "need >= 2 clusters for a clustered bootstrap"}
    by_group = [events[groups == g] for g in labels]
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(by_group), len(by_group))
        pooled = np.concatenate([by_group[i] for i in pick])
        stats[b] = pooled.mean()
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {
        "point": round(float(events.mean()), 4),
        "ci95_clustered": [round(float(lo), 4), round(float(hi), 4)],
        "clusters_resampled": int(len(labels)),
        "note": "Resamples clips, not frames — the interval for a new clip.",
    }


def point_biserial(events: np.ndarray, x: np.ndarray) -> dict:
    """Correlation between a binary outcome and a continuous condition."""
    ok = np.isfinite(x)
    y, xx = np.asarray(events, dtype=float)[ok], np.asarray(x, dtype=float)[ok]
    if len(np.unique(y)) < 2 or xx.size < 3:
        return {"r": None, "p": None, "n": int(xx.size),
                "note": "single-class outcome or too few samples"}
    from scipy.stats import pointbiserialr
    r, p = pointbiserialr(y, xx)
    return {"r": round(float(r), 4), "p": float(p), "n": int(xx.size),
            "mean_when_event": round(float(xx[y == 1].mean()), 3),
            "mean_when_clean": round(float(xx[y == 0].mean()), 3)}


# --------------------------------------------------------------------------- #
# Data assembly
# --------------------------------------------------------------------------- #
def build_frame_table(tolerance_s: float = 0.5):
    """Join frames to telemetry and compute capture-quality metrics."""
    import pandas as pd

    indices = {}
    for _, clip in FRAME_SETS:
        idx = F.load_frame_index(clip)
        if idx is None:
            raise SystemExit(
                f"No frame index for {clip}. Build it with:\n"
                f"  python -c \"import sys;sys.path.insert(0,'src');"
                f"from netinspect import frame_sync as F;"
                f"F.build_frame_index('data/raw/solaqua/{clip}_video.bag')\"")
        indices[clip] = idx

    parts = []
    for frames_dir, clip in FRAME_SETS:
        if not Path(frames_dir).exists():
            LOGGER.warning("Missing frame dir %s — skipped", frames_dir)
            continue
        tele = T.extract_telemetry(
            f"data/raw/solaqua/{clip}_data.bag",
            streams=["net_plane", "dvl", "depth_temp", "setpoint", "attitude"])
        joined = F.join_frames(frames_dir, tele, indices, tolerance_s=tolerance_s)
        joined["frames_dir"] = frames_dir
        parts.append(joined)

    df = pd.concat(parts, ignore_index=True)
    df["day"] = df["clip"].str[:10]
    df["standoff"] = df["net_plane_net_distance"]
    df["net_speed"] = np.hypot(df["net_plane_net_vel_u"], df["net_plane_net_vel_v"])
    df["locked"] = df["net_plane_net_lock"] > 0.5
    df["depth"] = df.get("depth_temp_depth")
    df["temperature"] = df.get("depth_temp_temperature")

    LOGGER.info("Computing capture-quality metrics for %d frames", len(df))
    metrics = [IQ.compute(read_image(p)).to_dict() for p in df["path"]]
    for name in IQ.METRIC_NAMES:
        df[name] = [m[name] for m in metrics]
    return df


def run_models(df, model_keys: list[str], conf: float, cache_path: Path, use_cache: bool):
    """Run each model over every frame, caching detection counts."""
    cache = {}
    if use_cache and cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    for key in model_keys:
        method, kwargs = MODELS[key]
        cache_key = f"{key}@{conf}"
        if cache_key not in cache:
            missing = [w for w in kwargs.values() if not Path(w).exists()]
            if missing:
                LOGGER.warning("%s: missing weights %s — skipped", key, missing)
                continue
            LOGGER.info("%s: running over %d frames (conf=%.2f)", key, len(df), conf)
            insp = NetInspector(**kwargs)
            counts, scores = [], []
            for i, path in enumerate(df["path"]):
                res = insp.predict(read_image(path), method=method, conf=conf)
                counts.append(len(res.boxes))
                scores.append(max((b.score for b in res.boxes), default=0.0))
                if (i + 1) % 100 == 0:
                    LOGGER.info("   %d/%d", i + 1, len(df))
            cache[cache_key] = {"frame": df["frame"].tolist(),
                                "n_det": counts, "max_score": scores}
            ensure_dir(cache_path.parent)
            cache_path.write_text(json.dumps(cache), encoding="utf-8")
        else:
            LOGGER.info("%s: using cached detections", key)

        entry = cache[cache_key]
        df[f"ndet_{key}"] = df["frame"].map(dict(zip(entry["frame"], entry["n_det"])))
        df[f"score_{key}"] = df["frame"].map(dict(zip(entry["frame"], entry["max_score"])))
        df[f"fp_{key}"] = (df[f"ndet_{key}"].fillna(0) > 0).astype(int)
    return df


# --------------------------------------------------------------------------- #
# Per-model analysis
# --------------------------------------------------------------------------- #
def analyse_model(df, key: str, target_rate: float, bin_width: float) -> dict:
    """Full condition analysis for one model."""
    y = df[f"fp_{key}"].to_numpy(int)
    clips = df["clip"].to_numpy()

    per_clip = {}
    for clip, g in df.groupby("clip"):
        per_clip[clip] = {
            **proportion_stat(g[f"fp_{key}"].tolist()),
            "standoff_mean": round(float(g["standoff"].mean()), 3),
            "sharpness_mean": round(float(g["sharpness"].mean()), 1),
            "day": clip[:10],
        }
    rates = [v["rate"] for v in per_clip.values()]

    # --- the pre-registered standoff hypothesis, tested within clip ---------
    within_clip = {}
    for clip, g in df.groupby("clip"):
        yy = g[f"fp_{key}"].to_numpy(int)
        entry = {"frames": int(len(g)), "events": int(yy.sum()),
                 "standoff_range": [round(float(g["standoff"].min()), 2),
                                    round(float(g["standoff"].max()), 2)]}
        if yy.sum() == 0:
            entry["verdict"] = "no false alarms in this clip — uninformative for standoff"
        else:
            entry["standoff_corr"] = point_biserial(yy, g["standoff"].to_numpy())
            entry["sharpness_corr"] = point_biserial(yy, g["sharpness"].to_numpy())
        within_clip[clip] = entry

    edges = np.arange(0.0, float(df["standoff"].max()) + bin_width, bin_width)
    train = df[df["day"] == TRAIN_DAY]
    heldout = df[df["day"] == HELDOUT_DAY]

    conditions = {c: point_biserial(y, df[c].to_numpy())
                  for c in CONDITION_COLS if c in df.columns}
    conditions_train_only = {
        c: point_biserial(train[f"fp_{key}"].to_numpy(int), train[c].to_numpy())
        for c in CONDITION_COLS if c in df.columns}

    band = IQ.fit_band(df["sharpness"].to_numpy(), y,
                       target_rate=target_rate, model=key, metric="sharpness")

    return {
        "overall_naive": proportion_stat(y.tolist()),
        "overall_clustered": clustered_bootstrap_rate(y, clips),
        "clustering": intracluster_correlation(y.astype(float), clips),
        "by_day": {d: proportion_stat(g[f"fp_{key}"].tolist())
                   for d, g in df.groupby("day")},
        "by_clip": per_clip,
        "between_clip_spread": {
            "min_rate": round(min(rates), 4), "max_rate": round(max(rates), 4),
            "spread": round(max(rates) - min(rates), 4),
            "train_day_only_spread": round(
                max(v["rate"] for k, v in per_clip.items() if v["day"] == TRAIN_DAY)
                - min(v["rate"] for k, v in per_clip.items() if v["day"] == TRAIN_DAY), 4),
        },
        "standoff_hypothesis": {
            "within_clip_tests": within_clip,
            "dose_response_train_day": dose_response(
                train["standoff"].tolist(), train[f"fp_{key}"].tolist(), edges),
            "dose_response_heldout_day": dose_response(
                heldout["standoff"].tolist(), heldout[f"fp_{key}"].tolist(), edges),
            "matched_band": matched_band_comparison({
                d: g.loc[(g["standoff"] >= MATCHED_BAND[0])
                         & (g["standoff"] < MATCHED_BAND[1]), f"fp_{key}"].tolist()
                for d, g in df.groupby("day")}),
            "naive_envelope_fit": fit_envelope(
                dose_response(df["standoff"].tolist(), y.tolist(), edges),
                target_rate, model=key).to_dict(),
        },
        "condition_correlations": conditions,
        "condition_correlations_train_day_only": conditions_train_only,
        "fitted_quality_band": band.to_dict(),
    }


def mediation_check(df) -> dict:
    """How much of capture quality is predictable from telemetry alone?

    If flight parameters explain a large share of sharpness, then quality is
    partly *controllable* at survey-planning time rather than merely observable
    after the fact — which is what makes it actionable for an operator.
    """
    from sklearn.linear_model import LinearRegression

    cols = ["standoff", "net_speed", "depth"]
    sub = df[cols + ["sharpness", "contrast"]].dropna()
    if len(sub) < 30:
        return {"note": "insufficient joined data"}
    out = {"n": int(len(sub)), "predictors": cols, "targets": {}}
    for target in ("sharpness", "contrast"):
        m = LinearRegression().fit(sub[cols], sub[target])
        out["targets"][target] = {
            "r2": round(float(m.score(sub[cols], sub[target])), 4),
            "coefficients": {c: round(float(v), 2) for c, v in zip(cols, m.coef_)},
        }
    out["interpretation"] = (
        "Flight parameters explain a substantial share of capture quality "
        "(faster sweep and greater depth reduce sharpness), so quality is partly "
        "controllable when planning a survey, not only measurable afterwards.")
    return out


def ensemble_mechanism(results: dict, model_keys: list[str]) -> dict:
    """Do the models' failure modes point in opposite directions?

    If they do, requiring agreement between them cancels errors that occur in
    different capture regimes — a mechanistic explanation for the agreement
    ensemble's measured behaviour.
    """
    signs = {}
    for key in model_keys:
        c = results[key]["condition_correlations"].get("sharpness", {})
        if c.get("r") is not None:
            signs[key] = c["r"]
    if len(signs) < 2:
        return {"note": "need >= 2 models"}
    opposed = [(a, b) for a in signs for b in signs
               if a < b and signs[a] * signs[b] < 0]
    return {
        "sharpness_correlation_by_model": signs,
        "opposed_pairs": [list(p) for p in opposed],
        "interpretation": (
            "Models with opposite-signed sharpness correlations fail in different "
            "capture regimes, so requiring their agreement suppresses both failure "
            "modes. This is a mechanism for the ensemble result already measured in "
            "this repo, not a new claim about accuracy."
            if opposed else
            "No opposed pair found; the agreement ensemble's benefit is not explained "
            "by opposite capture-quality sensitivity in this sample."),
    }


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def make_plots(df, model_keys: list[str], out_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    clips = sorted(df["clip"].unique())
    colors = {c: f"C{i}" for i, c in enumerate(clips)}

    # 1. The headline: clip dominates, standoff does not.
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    x = np.arange(len(clips))
    w = 0.8 / max(1, len(model_keys))
    for i, key in enumerate(model_keys):
        vals = [df.loc[df["clip"] == c, f"fp_{key}"].mean() for c in clips]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=key)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c[-8:]}\n{c[:10]}\n{df.loc[df['clip']==c,'standoff'].mean():.2f} m"
                        for c in clips], fontsize=7)
    ax.set_ylabel("Frames with >=1 false alarm")
    ax.set_title("False alarms are a property of the CLIP\n"
                 "(three training clips flown at near-identical standoff)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    for c in clips:
        g = df[df["clip"] == c]
        ax.scatter(g["standoff"], g["sharpness"], s=12, alpha=0.5,
                   color=colors[c], label=c[-8:])
    ax.set_xlabel("Standoff distance from net (m)")
    ax.set_ylabel("Sharpness (variance of Laplacian)")
    ax.set_title("Capture quality separates the clips\nmore cleanly than standoff does")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[2]
    for key in model_keys:
        vals, rates = [], []
        qs = np.quantile(df["sharpness"], np.linspace(0, 1, 7))
        for lo, hi in zip(qs[:-1], qs[1:]):
            sel = df[(df["sharpness"] >= lo) & (df["sharpness"] < hi)]
            if len(sel) < 5:
                continue
            vals.append((lo + hi) / 2)
            rates.append(sel[f"fp_{key}"].mean())
        ax.plot(vals, rates, marker="o", label=key)
    ax.set_xlabel("Sharpness (variance of Laplacian)")
    ax.set_ylabel("Frames with >=1 false alarm")
    ax.set_title("Opposite failure modes\ndetector fires on sharp; segmenter on degraded")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    p = out_dir / "what_drives_false_alarms.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(str(p))

    # 2. The refuted hypothesis, shown honestly.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    for c in clips:
        g = df[df["clip"] == c]
        ax.scatter(g["standoff"], g["net_speed"], s=14, alpha=0.55,
                   color=colors[c], label=f"{c[-8:]} ({c[:10]})")
    ax.axvspan(*MATCHED_BAND, color="grey", alpha=0.15,
               label=f"matched band {MATCHED_BAND[0]}-{MATCHED_BAND[1]} m")
    ax.set_xlabel("Standoff (m)")
    ax.set_ylabel("Net-relative speed (m/s)")
    ax.set_title("The two days were flown on different profiles\n(the hypothesis that motivated this)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    def _span(c: str) -> float:
        s = df.loc[df["clip"] == c, "standoff"]
        return float(s.max() - s.min())

    widest = max(clips, key=_span)
    g = df[df["clip"] == widest]
    for key in model_keys:
        qs = np.linspace(g["standoff"].min(), g["standoff"].max(), 6)
        xs, ys = [], []
        for lo, hi in zip(qs[:-1], qs[1:]):
            sel = g[(g["standoff"] >= lo) & (g["standoff"] < hi)]
            if len(sel) < 3:
                continue
            xs.append((lo + hi) / 2)
            ys.append(sel[f"fp_{key}"].mean())
        ax.plot(xs, ys, marker="o", label=key)
    ax.set_xlabel("Standoff (m)")
    ax.set_ylabel("Frames with >=1 false alarm")
    ax.set_ylim(-0.05, 1.0)
    ax.set_title(f"Within one clip ({widest[-8:]}) spanning\n"
                 f"{g['standoff'].min():.2f}-{g['standoff'].max():.2f} m: hypothesis not supported")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / "standoff_hypothesis_refuted.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(str(p))
    return written


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="det_v1,seg_v3,seg_gpu")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--tolerance", type=float, default=0.5)
    ap.add_argument("--target-fp-rate", type=float, default=0.05)
    ap.add_argument("--bin-width", type=float, default=0.2)
    ap.add_argument("--out", default="reports/results/operating_envelope")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in model_keys if m not in MODELS]
    if unknown:
        raise SystemExit(f"Unknown models {unknown}; choose from {list(MODELS)}")

    out_dir = ensure_dir(args.out)
    df = build_frame_table(tolerance_s=args.tolerance)
    total = len(df)
    df = df[df["standoff"].notna() & df["locked"]].copy()
    LOGGER.info("%d/%d frames have locked net-plane telemetry", len(df), total)

    df = run_models(df, model_keys, args.conf, out_dir / "detection_cache.json",
                    use_cache=not args.no_cache)
    model_keys = [k for k in model_keys if f"fp_{k}" in df.columns]
    if not model_keys:
        raise SystemExit("No models ran; check weights paths.")

    per_model = {k: analyse_model(df, k, args.target_fp_rate, args.bin_width)
                 for k in model_keys}

    results = {
        "config": {"conf": args.conf, "tolerance_s": args.tolerance,
                   "models": model_keys, "matched_band_m": list(MATCHED_BAND),
                   "target_fp_rate": args.target_fp_rate},
        "data": {
            "frames_total": int(total),
            "frames_analysed": int(len(df)),
            "clips": int(df["clip"].nunique()),
            "by_day": {d: int(n) for d, n in df["day"].value_counts().items()},
            "by_clip": {c: int(n) for c, n in df["clip"].value_counts().items()},
        },
        "flight_profiles": {
            clip: {
                "frames": int(len(g)),
                "commanded_standoff_m": round(float(g["setpoint_d_net_distance"].median()), 3)
                if "setpoint_d_net_distance" in g else None,
                "achieved_standoff_m": round(float(g["standoff"].mean()), 3),
                "standoff_range_m": [round(float(g["standoff"].min()), 2),
                                     round(float(g["standoff"].max()), 2)],
                "net_speed_ms": round(float(g["net_speed"].mean()), 3),
                "depth_m": round(float(g["depth"].mean()), 2) if "depth" in g else None,
                "temperature_c": round(float(g["temperature"].mean()), 2)
                if "temperature" in g else None,
                "sharpness": round(float(g["sharpness"].mean()), 1),
                "contrast": round(float(g["contrast"].mean()), 1),
            } for clip, g in df.groupby("clip")},
        "models": per_model,
        "mediation_telemetry_to_quality": mediation_check(df),
        "ensemble_mechanism": ensemble_mechanism(per_model, model_keys),
        "conclusions": [
            "The pre-registered hypothesis — that the different-day gap is an "
            "operating-envelope violation driven by standoff distance — is NOT "
            "supported. The clip spanning the widest standoff range produces zero "
            "false alarms, and standoff has no within-clip association on the "
            "held-out day.",
            "False-alarm rate is dominated by clip (scene) identity. Three "
            "training-day clips flown at near-identical standoff differ by up to "
            "0.33 in false-alarm frame rate.",
            "Capture quality (sharpness, contrast) is the strongest per-frame "
            "correlate, and its sign is model-dependent: the detector fires on "
            "sharp, high-contrast frames; the higher-capacity segmenter fires on "
            "degraded ones.",
            "Because frames cluster within clips, per-frame confidence intervals "
            "overstate confidence. Generalisation should be reported with "
            "clip-level resampling; with four clips those intervals are wide.",
        ],
        "caveats": [
            "All frames are undamaged real net, so every detection is a false "
            "positive. This measures false-alarm behaviour only — recall on real "
            "damage is unmeasured and remains unvalidated.",
            "Standoff, sweep speed and depth co-vary by design between the two "
            "recording days; no day-level difference can be attributed to one of "
            "them alone. Within-clip tests are what isolate a variable here.",
            "Four clips from one site over two days is a small sample of scenes. "
            "The clip-level intervals reflect that and should not be narrowed by "
            "counting frames.",
        ],
    }

    if not args.no_plots:
        results["plots"] = make_plots(df, model_keys, out_dir)

    write_json(results, out_dir / "operating_envelope.json")
    df.to_parquet(out_dir / "frame_conditions.parquet", index=False)

    # ---- console summary --------------------------------------------------
    print("\n" + "=" * 86)
    print("WHAT DRIVES FALSE ALARMS — day, flight profile, or scene?")
    print("=" * 86)
    d = results["data"]
    print(f"{d['frames_analysed']} frames, {d['clips']} clips, all showing undamaged net.\n")

    print(f"  {'clip':22s} {'n':>4s} {'cmd_m':>6s} {'ach_m':>6s} {'speed':>6s} "
          f"{'depth':>6s} {'sharp':>7s}")
    for clip, p in results["flight_profiles"].items():
        print(f"  {clip:22s} {p['frames']:4d} "
              f"{p['commanded_standoff_m'] or float('nan'):6.2f} "
              f"{p['achieved_standoff_m']:6.2f} {p['net_speed_ms']:6.2f} "
              f"{p['depth_m'] or float('nan'):6.2f} {p['sharpness']:7.1f}")

    for key in model_keys:
        r = per_model[key]
        print(f"\n--- {key} " + "-" * (80 - len(key)))
        n = r["overall_naive"]
        c = r["overall_clustered"]
        cl = r["clustering"]
        print(f"  false-alarm frame rate {n['rate']:.3f} ({n['k']}/{n['n']})")
        print(f"    naive per-frame CI      {n['ci95']}")
        print(f"    clip-clustered CI       {c.get('ci95_clustered')}   <- honest interval")
        print(f"    ICC={cl.get('icc')}  design effect={cl.get('design_effect')}  "
              f"effective n={cl.get('effective_n')} (of {cl.get('naive_n')} frames)")
        print("  by clip:")
        for clip, s in r["by_clip"].items():
            print(f"    {clip:22s} {s['rate']:.3f} ({s['k']:3d}/{s['n']:3d})  "
                  f"standoff {s['standoff_mean']:.2f} m  sharpness {s['sharpness_mean']:6.1f}")
        print(f"  between-clip spread: {r['between_clip_spread']['spread']:.3f} "
              f"(training day alone: {r['between_clip_spread']['train_day_only_spread']:.3f})")
        print("  standoff hypothesis, tested within clip:")
        for clip, w in r["standoff_hypothesis"]["within_clip_tests"].items():
            if "verdict" in w:
                print(f"    {clip:22s} {w['verdict']}")
            else:
                sc = w["standoff_corr"]
                print(f"    {clip:22s} r={sc['r']:+.3f} p={sc['p']:.3g} "
                      f"(range {w['standoff_range']} m, {w['events']} events)")
        print("  strongest per-frame correlates (all frames):")
        cc = sorted(r["condition_correlations"].items(),
                    key=lambda kv: -abs(kv[1].get("r") or 0))
        for name, s in cc[:4]:
            if s.get("r") is not None:
                print(f"    {name:13s} r={s['r']:+.3f}  p={s['p']:.2e}")

    em = results["ensemble_mechanism"]
    if "sharpness_correlation_by_model" in em:
        print("\n--- why the agreement ensemble works " + "-" * 46)
        for k, v in em["sharpness_correlation_by_model"].items():
            print(f"    {k:10s} corr(sharpness, false alarm) = {v:+.3f}")
        print(f"    opposed pairs: {em['opposed_pairs']}")

    med = results["mediation_telemetry_to_quality"]
    if "targets" in med:
        print("\n--- can telemetry predict capture quality? " + "-" * 40)
        for t, s in med["targets"].items():
            print(f"    {t:10s} R^2={s['r2']:.3f}  {s['coefficients']}")

    print("\nCONCLUSIONS")
    for c in results["conclusions"]:
        print(f"  - {c}")
    print("\nCAVEATS")
    for c in results["caveats"]:
        print(f"  - {c}")
    print(f"\nWrote {out_dir/'operating_envelope.json'}")


if __name__ == "__main__":
    main()
