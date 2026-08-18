"""Query this project's artifacts as one reporting layer.

Registers every result artifact — frame conditions, per-pass inspection reports,
ROV telemetry, site registry, evidence ledger — as DuckDB views and runs either a
canned report or arbitrary SQL. Views read the files in place, so the warehouse
cannot drift from the artifacts.

Examples
--------
    python scripts/report.py --list
    python scripts/report.py pass_quality
    python scripts/report.py --sql "SELECT clip, AVG(sharpness) FROM frames GROUP BY 1"
    python scripts/report.py --schema frames
"""
from __future__ import annotations

import argparse

import _common  # noqa: F401

from netinspect import warehouse
from netinspect.utils import get_logger

LOGGER = get_logger()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", nargs="?", default=None,
                    help=f"Canned report: {', '.join(sorted(warehouse.REPORTS))}")
    ap.add_argument("--sql", default=None, help="Run arbitrary SQL instead")
    ap.add_argument("--list", action="store_true", help="List views and reports")
    ap.add_argument("--schema", default=None, help="Show one view's columns")
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()

    wh = warehouse.build()

    if args.list or not (args.report or args.sql or args.schema):
        print("\nVIEWS")
        if wh.views:
            for name in wh.tables():
                print(f"  {name:22s} <- {wh.views[name]}")
        else:
            print("  none — run scripts/analyze_operating_envelope.py first")
        print("\nREPORTS")
        for name, (desc, _) in sorted(warehouse.REPORTS.items()):
            print(f"  {name:22s} {desc}")
        print()
        return

    if args.schema:
        if args.schema not in wh.views:
            raise SystemExit(f"No view {args.schema!r}. Available: {wh.tables()}")
        print(f"\n{args.schema}")
        for col, typ in wh.describe(args.schema):
            print(f"  {col:34s} {typ}")
        print()
        return

    import pandas as pd
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)

    if args.sql:
        print(wh.df(args.sql).head(args.limit).to_string(index=False))
        return

    if args.report not in warehouse.REPORTS:
        raise SystemExit(f"Unknown report {args.report!r}. "
                         f"Choose from: {', '.join(sorted(warehouse.REPORTS))}")
    desc, _ = warehouse.REPORTS[args.report]
    print(f"\n{desc}\n" + "-" * len(desc))
    try:
        df = warehouse.run_report(wh, args.report)
    except Exception as exc:
        raise SystemExit(f"Report failed — a required view is probably missing "
                         f"({exc}). Run scripts/report.py --list to see what exists.")
    print(df.head(args.limit).to_string(index=False))
    print()


if __name__ == "__main__":
    main()
