"""A local reporting warehouse over this project's artifacts.

Each analysis in this repo writes its own JSON or parquet. That is right for
reproducibility — every number traces to the run that produced it — but it makes
cross-run questions awkward: *which passes had poor coverage, and were those the
ones with elevated false alarms?* needs three files opened at once.

This module registers those artifacts as DuckDB views so such questions are one
query. It is deliberately a **view layer, not a copy**: DuckDB reads the parquet
and JSON in place, so the warehouse can never drift from the artifacts, and
deleting it loses nothing. Rebuilding is idempotent.

Views
-----
``frames``
    Per-frame capture conditions joined to telemetry and model outputs.
``inspections``
    One row per ROV pass: coverage, detections, capture summary.
``sites``
    Licensed aquaculture localities (when fetched).
``telemetry_<stream>``
    Raw ROV sensor streams, one view per canonical stream.
``evidence``
    The assistant's evidence ledger, queryable alongside the numbers it bounds.

Examples
--------
>>> from netinspect import warehouse
>>> wh = warehouse.build()
>>> wh.sql("SELECT clip, frames, coverage FROM inspections ORDER BY coverage").df()
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import get_logger, require

LOGGER = get_logger()

FRAME_CONDITIONS = Path("reports/results/operating_envelope/frame_conditions.parquet")
INSPECTION_REPORTS = Path("reports/results/inspection_reports")
TELEMETRY_DIR = Path("data/processed/telemetry")
SITES_CACHE = Path("data/raw/sites/cod_localities.json")


@dataclass
class Warehouse:
    """A DuckDB connection with this project's artifacts registered as views."""
    conn: Any
    views: dict[str, str]

    def sql(self, query: str):
        """Run a query and return the DuckDB relation."""
        return self.conn.sql(query)

    def df(self, query: str):
        """Run a query and return a pandas DataFrame."""
        return self.conn.sql(query).df()

    def tables(self) -> list[str]:
        return sorted(self.views)

    def describe(self, view: str) -> list[tuple[str, str]]:
        """Column names and types for one view."""
        rows = self.conn.sql(f"DESCRIBE {view}").fetchall()
        return [(r[0], r[1]) for r in rows]

    def close(self) -> None:
        self.conn.close()


def _register_inspections(conn, reports_dir: Path) -> bool:
    """Flatten per-pass inspection reports into one row each.

    Written as an explicit projection rather than ``read_json`` over the whole
    document because the reports are nested and only a handful of fields are
    worth a column; the full JSON stays on disk for anything deeper.
    """
    files = sorted(reports_dir.glob("*_inspection_report.json"))
    if not files:
        return False

    rows = []
    for path in files:
        try:
            r = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            LOGGER.warning("Skipping unreadable report %s", path.name)
            continue
        if "error" in r:
            continue
        geom = r.get("validity_geometry") or {}
        qual = r.get("validity_quality") or {}
        det = r.get("detections") or {}
        cap = r.get("capture") or {}
        rows.append({
            "clip": r.get("clip"),
            "day": r.get("day"),
            "frames": r.get("frames"),
            "coverage": geom.get("compliance"),
            "verdict": geom.get("verdict"),
            "quality_share_in_band": qual.get("share_in_band"),
            "sharpness_mean": cap.get("sharpness_mean"),
            "standoff_m_mean": cap.get("standoff_m_mean"),
            "net_speed_ms_mean": cap.get("net_speed_ms_mean"),
            "depth_m_mean": cap.get("depth_m_mean"),
            "water_temperature_c": cap.get("water_temperature_c"),
            "model": det.get("model"),
            "frames_with_detection": det.get("frames_with_detection"),
            "confirmed_events": det.get("confirmed_events_min3_frames"),
            "source_report": str(path),
        })
    if not rows:
        return False

    pd = require("pandas", hint="pip install pandas")
    conn.register("_inspections_df", pd.DataFrame(rows))
    conn.sql("CREATE OR REPLACE VIEW inspections AS SELECT * FROM _inspections_df")
    return True


def _register_evidence(conn) -> bool:
    from .assistant.evidence import ledger_dicts

    pd = require("pandas", hint="pip install pandas")
    rows = ledger_dicts()
    if not rows:
        return False
    conn.register("_evidence_df", pd.DataFrame(rows))
    conn.sql("CREATE OR REPLACE VIEW evidence AS SELECT * FROM _evidence_df")
    return True


def build(frame_conditions: Path = FRAME_CONDITIONS,
          reports_dir: Path = INSPECTION_REPORTS,
          telemetry_dir: Path = TELEMETRY_DIR,
          sites_cache: Path = SITES_CACHE,
          database: str = ":memory:") -> Warehouse:
    """Register every available artifact as a DuckDB view.

    Missing artifacts are skipped with a log line rather than raising — a fresh
    clone has none of them, and the warehouse should still open so the user can
    see which views exist and what would produce the rest.
    """
    duckdb = require("duckdb", hint="pip install -e '.[assistant]'")
    conn = duckdb.connect(database)
    views: dict[str, str] = {}

    if frame_conditions.exists():
        conn.sql("CREATE OR REPLACE VIEW frames AS "
                 f"SELECT * FROM read_parquet('{frame_conditions.as_posix()}')")
        views["frames"] = str(frame_conditions)
    else:
        LOGGER.info("No frame conditions — run scripts/analyze_operating_envelope.py")

    if _register_inspections(conn, reports_dir):
        views["inspections"] = str(reports_dir)
    else:
        LOGGER.info("No inspection reports — run scripts/inspection_report.py --all")

    for path in sorted(telemetry_dir.glob("*__*.parquet")) if telemetry_dir.exists() else []:
        clip, stream = path.stem.split("__", 1)
        view = f"telemetry_{stream}"
        if view not in views:
            pattern = (telemetry_dir / f"*__{stream}.parquet").as_posix()
            conn.sql(f"CREATE OR REPLACE VIEW {view} AS "
                     f"SELECT * FROM read_parquet('{pattern}', filename=true)")
            views[view] = pattern
    if not any(v.startswith("telemetry_") for v in views):
        LOGGER.info("No telemetry — run scripts/extract_telemetry.py --all")

    if sites_cache.exists():
        conn.sql("CREATE OR REPLACE VIEW sites AS "
                 f"SELECT * FROM read_json_auto('{sites_cache.as_posix()}')")
        views["sites"] = str(sites_cache)

    if _register_evidence(conn):
        views["evidence"] = "netinspect.assistant.evidence.LEDGER"

    LOGGER.info("Warehouse ready with %d views: %s", len(views), ", ".join(sorted(views)))
    return Warehouse(conn=conn, views=views)


# --------------------------------------------------------------------------- #
# Canned reporting queries
# --------------------------------------------------------------------------- #
REPORTS: dict[str, tuple[str, str]] = {
    "pass_quality": (
        "Inspection passes ranked by verified coverage",
        """
        SELECT clip, frames,
               ROUND(coverage * 100, 1)              AS coverage_pct,
               ROUND(quality_share_in_band * 100, 1) AS quality_pct,
               frames_with_detection, confirmed_events, verdict
        FROM inspections
        ORDER BY coverage ASC, quality_pct ASC
        """),
    "detections_by_standoff": (
        "False-alarm rate by standoff band (all frames show undamaged net)",
        """
        SELECT ROUND(standoff * 5) / 5            AS standoff_band_m,
               COUNT(*)                            AS frames,
               SUM(fp_det_v1)                      AS false_alarms,
               ROUND(AVG(fp_det_v1) * 100, 1)      AS false_alarm_pct
        FROM frames
        WHERE standoff IS NOT NULL
        GROUP BY 1 ORDER BY 1
        """),
    "clip_vs_day": (
        "Between-clip spread compared with the day effect",
        """
        SELECT day, clip, COUNT(*) AS frames,
               ROUND(AVG(fp_det_v1) * 100, 1) AS det_v1_fa_pct,
               ROUND(AVG(sharpness), 1)       AS sharpness
        FROM frames GROUP BY day, clip ORDER BY day, det_v1_fa_pct DESC
        """),
    "unvalidated": (
        "Claims this project explicitly cannot make",
        """
        SELECT topic, statement, caveat FROM evidence
        WHERE level = 'not_validated'
        """),
    "water": (
        "Water temperature and depth per pass, from ROV telemetry",
        """
        SELECT clip, frames,
               water_temperature_c, depth_m_mean, standoff_m_mean, net_speed_ms_mean
        FROM inspections ORDER BY clip
        """),
}


def run_report(wh: Warehouse, name: str):
    """Run one canned report by name."""
    if name not in REPORTS:
        raise KeyError(f"Unknown report {name!r}; choose from {sorted(REPORTS)}")
    _, query = REPORTS[name]
    return wh.df(query)


__all__ = ["Warehouse", "build", "REPORTS", "run_report",
           "FRAME_CONDITIONS", "INSPECTION_REPORTS", "TELEMETRY_DIR", "SITES_CACHE"]
