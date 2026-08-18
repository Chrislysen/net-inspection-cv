"""Tests for the reporting warehouse.

The important property is that a fresh clone — with none of the analysis
artifacts generated yet — still opens the warehouse and reports which views are
missing, rather than raising. A reporting layer that crashes when there is
nothing to report is worse than useless during a demo.

Known environment conflict
--------------------------
On Windows, ``duckdb.connect()`` raises a native access violation when the same
process has already loaded the heavy ML stack (torch + ultralytics +
onnxruntime + OpenCV). Each library works fine alongside duckdb on its own — the
fault only appears once several are resident, which points at native allocator
or handle pressure rather than any single incompatibility.

This is guarded rather than worked around. CI never hits it (CI installs only
the ``cv,dev`` extras — no duckdb, no torch), and the warehouse itself is
unaffected in normal use, where it runs in its own process. Run these tests on
their own to exercise them locally:

    pytest tests/test_warehouse.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb", reason="warehouse needs the assistant extra")

_HEAVY = ("torch", "ultralytics", "onnxruntime")


@pytest.fixture(autouse=True)
def _skip_when_ml_stack_resident():
    """Skip at call time, not collection time.

    Collection happens before any test imports torch, so a module-level
    ``skipif`` would always see a clean ``sys.modules`` and never fire. The
    check has to run when the test does.
    """
    loaded = [m for m in _HEAVY if m in sys.modules]
    if loaded:
        pytest.skip(f"duckdb segfaults on Windows once {', '.join(loaded)} are "
                    "resident in the same process; run "
                    "'pytest tests/test_warehouse.py' separately")


from netinspect import warehouse  # noqa: E402


def _empty(tmp_path: Path):
    return warehouse.build(
        frame_conditions=tmp_path / "nope.parquet",
        reports_dir=tmp_path / "no_reports",
        telemetry_dir=tmp_path / "no_telemetry",
        sites_cache=tmp_path / "no_sites.json",
    )


def test_build_survives_missing_artifacts(tmp_path):
    wh = _empty(tmp_path)
    assert isinstance(wh.views, dict)
    wh.close()


def test_evidence_view_is_always_available(tmp_path):
    """The ledger is code, not an artifact, so it registers with no data present."""
    wh = _empty(tmp_path)
    assert "evidence" in wh.views
    assert wh.df("SELECT count(*) AS n FROM evidence").iloc[0, 0] > 0
    wh.close()


def test_unvalidated_report_runs_against_the_ledger_alone(tmp_path):
    wh = _empty(tmp_path)
    df = warehouse.run_report(wh, "unvalidated")
    assert len(df) >= 1
    assert "recall_on_real_damage" in set(df["topic"])
    wh.close()


def test_unknown_report_name_raises(tmp_path):
    wh = _empty(tmp_path)
    with pytest.raises(KeyError):
        warehouse.run_report(wh, "not_a_report")
    wh.close()


def test_every_canned_report_declares_a_description():
    for name, (desc, query) in warehouse.REPORTS.items():
        assert desc and isinstance(desc, str), name
        assert "SELECT" in query.upper(), name


def test_registered_views_are_queryable(tmp_path):
    wh = _empty(tmp_path)
    for view in wh.tables():
        assert wh.df(f"SELECT * FROM {view} LIMIT 1") is not None
    wh.close()


def test_describe_returns_columns(tmp_path):
    wh = _empty(tmp_path)
    cols = wh.describe("evidence")
    assert any(c == "topic" for c, _ in cols)
    wh.close()


def test_frames_view_registers_when_the_parquet_exists(tmp_path):
    pd = pytest.importorskip("pandas")
    p = tmp_path / "frames.parquet"
    pd.DataFrame({"clip": ["a"], "standoff": [0.6], "fp_det_v1": [0]}).to_parquet(p)
    wh = warehouse.build(
        frame_conditions=p,
        reports_dir=tmp_path / "none",
        telemetry_dir=tmp_path / "none",
        sites_cache=tmp_path / "none.json",
    )
    assert "frames" in wh.views
    assert wh.df("SELECT count(*) AS n FROM frames").iloc[0, 0] == 1
    wh.close()


def test_malformed_inspection_report_is_skipped_not_fatal(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "bad_inspection_report.json").write_text("{not json", encoding="utf-8")
    wh = warehouse.build(
        frame_conditions=tmp_path / "none.parquet",
        reports_dir=reports,
        telemetry_dir=tmp_path / "none",
        sites_cache=tmp_path / "none.json",
    )
    assert "inspections" not in wh.views
    wh.close()
