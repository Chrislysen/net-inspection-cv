"""Plan net inspections across an operator's real sites, from public data.

Joins two open Norwegian sources to answer an operational question the models
alone cannot:

* **Fiskeridirektoratet** (NLOD) — every licensed cod locality: coordinates,
  capacity, permits, production form.
* **MET Norway** (NLOD + CC BY 4.0) — hourly sea temperature, current speed and
  wave height at each of those coordinates.

Output is a per-site inspection plan: when conditions resemble those this
repo's detection models were characterised in, and what the water temperature
means for the fish at that site given their size.

This is the decision-support half of the project. The computer-vision half
answers "is there damage in this frame"; this half answers "is it worth flying
today, and was the footage captured under conditions where that answer means
anything".

Honesty
-------
Window thresholds are reference points, not certified operating limits — see
``src/netinspect/ocean.py``. Growth projections use published cod thermal
relationships but are **not fitted to production data**, because no per-cohort
cod production records were publicly available; the Fiskeridirektoratet
production-overview files cover salmon and rainbow trout only. Fitting needs an
operator's own records.

Examples
--------
    python scripts/site_planner.py --operator "ODE"
    python scripts/site_planner.py --operator "ODE" --forecast --fish-weight-g 2000
    python scripts/site_planner.py --list-operators
"""
from __future__ import annotations

import argparse

import _common  # noqa: F401

from netinspect import ocean, sites
from netinspect.utils import ensure_dir, get_logger, write_json

LOGGER = get_logger()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--operator", default=None,
                    help='Operator name substring, e.g. "ODE"')
    ap.add_argument("--list-operators", action="store_true",
                    help="List every operator holding a cod licence, then exit")
    ap.add_argument("--forecast", action="store_true",
                    help="Fetch the MET ocean forecast for each sea site")
    ap.add_argument("--fish-weight-g", type=float, default=2000.0,
                    help="Mean fish weight for the thermal-stress readout")
    ap.add_argument("--max-sites", type=int, default=8,
                    help="Cap forecast calls (be polite to MET)")
    ap.add_argument("--refresh", action="store_true",
                    help="Bypass the locality cache")
    ap.add_argument("--out", default="reports/results/site_planning")
    args = ap.parse_args()

    all_sites = sites.fetch_cod_localities(force=args.refresh)

    if args.list_operators:
        counts: dict[str, int] = {}
        for s in all_sites:
            for op in (o.strip() for o in s.operators.split(",") if o.strip()):
                counts[op] = counts.get(op, 0) + 1
        print(f"{len(counts)} operators hold Norwegian cod licences:\n")
        for op, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {n:3d}  {op}")
        return

    if not args.operator:
        ap.error("Provide --operator, or use --list-operators.")

    selected = sites.by_operator(args.operator, all_sites)
    if not selected:
        raise SystemExit(f"No cod localities found for operator {args.operator!r}. "
                         "Try --list-operators.")

    summary = sites.summarise(selected)
    print("\n" + "=" * 82)
    print(f"COD LOCALITIES OPERATED BY {args.operator.upper()} "
          "(Fiskeridirektoratet, open data)")
    print("=" * 82)
    print(f"{summary['sites']} localities · {summary['active']} active · "
          f"{summary['sea_sites']} in sea · {summary['land_sites']} on land")
    print(f"Licensed sea capacity: {summary['licensed_tonnes_total']:,.0f} tonnes")
    if summary["licensed_individuals_total"]:
        print(f"Licensed land capacity: "
              f"{summary['licensed_individuals_total']:,} individuals")
    print(f"Counties: {', '.join(summary['counties'])}")
    print(f"Production forms: {', '.join(summary['production_forms'])}\n")

    print(f"  {'loknr':>6s} {'name':24s} {'placement':9s} {'capacity':>12s} "
          f"{'county':18s} {'form'}")
    for s in sorted(selected, key=lambda x: (not x.is_sea_site, -x.capacity)):
        cap = (f"{s.capacity:,.0f} t" if s.capacity_unit == sites.UNIT_TONNES
               else f"{s.capacity:,.0f} stk")
        print(f"  {s.loknr:6d} {s.name[:24]:24s} {s.placement[:9]:9s} {cap:>12s} "
              f"{s.county[:18]:18s} {s.production_form[:26]}")

    payload = {
        "operator_query": args.operator,
        "summary": summary,
        "sites": [s.to_dict() for s in selected],
        "sources": {
            "localities": "Fiskeridirektoratet Yggdrasil ArcGIS (NLOD)",
            "ocean": "MET Norway oceanforecast 2.0 (NLOD + CC BY 4.0)",
        },
    }

    if args.forecast:
        sea = [s for s in selected if s.is_sea_site and s.has_coordinates
               and s.is_active][:args.max_sites]
        print("\n" + "=" * 82)
        print("INSPECTION WINDOWS AND THERMAL STATE  (live MET ocean forecast)")
        print("=" * 82)
        plans = []
        for s in sea:
            try:
                fc = ocean.fetch_forecast(s.lat, s.lon)
            except Exception as exc:
                LOGGER.warning("%s: forecast unavailable (%s)", s.name, exc)
                plans.append({"loknr": s.loknr, "name": s.name, "error": str(exc)})
                continue
            windows = ocean.inspection_windows(fc)
            wsum = ocean.summarise_windows(windows)
            temps = [x.sea_water_temperature_c for x in fc
                     if x.sea_water_temperature_c is not None]
            now_t = temps[0] if temps else None
            stress = (ocean.thermal_stress(args.fish_weight_g, now_t)
                      if now_t is not None else None)
            plans.append({"loknr": s.loknr, "name": s.name,
                          "lat": s.lat, "lon": s.lon,
                          "windows": wsum,
                          "sea_temperature_c": now_t,
                          "thermal_state": stress,
                          "hourly": [w.to_dict() for w in windows[:48]]})

            print(f"\n  {s.name} ({s.loknr}) — {s.county}")
            print(f"    forecast    {wsum['hours']}h from {wsum['from']}")
            print(f"    ratings     {wsum['ratings']}  "
                  f"(good {wsum['good_fraction']:.0%})")
            print(f"    next window {wsum['first_good_window']} "
                  f"[{wsum['first_good_rating']}]")
            if stress:
                print(f"    sea temp    {stress['temperature_c']} degC · "
                      f"optimum for {args.fish_weight_g:.0f} g cod "
                      f"{stress['thermal_optimum_c']} degC "
                      f"({stress['delta_c']:+.1f}) — {stress['state']}")
        payload["inspection_plans"] = plans

        print("\n  Note: window thresholds are reference points derived from this")
        print("  repo's own ROV telemetry and ordinary small-ROV practice — they")
        print("  indicate similarity to the evaluated conditions, not certified")
        print("  operating limits or guaranteed detection performance.")

    out_dir = ensure_dir(args.out)
    write_json(payload, out_dir / f"{args.operator.lower().replace(' ', '_')}_plan.json")
    print(f"\nWrote {out_dir}/{args.operator.lower().replace(' ', '_')}_plan.json")


if __name__ == "__main__":
    main()
