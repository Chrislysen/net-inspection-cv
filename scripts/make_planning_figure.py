"""Build the README figure for site planning: when is an inspection flyable?

Joins two open Norwegian sources — the Fiskeridirektoratet cod-locality register
and the MET Norway ocean forecast — into the operational question the detection
models cannot answer on their own: *when is it worth flying, and what is the
water doing to the fish?*

Kept employer-neutral on purpose. The data is a real operator's licensed sites
from the public register, but the figure is labelled generically: this repository
is a capability demonstration, not a pitch aimed at one company.

Run: ``python scripts/make_planning_figure.py --operator ODE``
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _common  # noqa: F401
import numpy as np

from netinspect import ocean, sites
from netinspect.utils import ensure_dir, get_logger

LOGGER = get_logger()

RATING_COLOURS = {"good": "#1baf7a", "marginal": "#eda100", "poor": "#eb6834",
                  "unknown": "#b9b8ae"}
SURFACE, INK, INK_SOFT, INK_FAINT, NEUTRAL = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#8a8981", "#b9b8ae")


def _style(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_FAINT)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_SOFT, labelsize=9, length=3, width=0.8)
    ax.grid(True, axis=grid_axis, color=NEUTRAL, alpha=0.35, linewidth=0.7)
    ax.set_axisbelow(True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--operator", default="ODE")
    ap.add_argument("--max-sites", type=int, default=6)
    ap.add_argument("--fish-weight-g", type=float, default=2000.0)
    ap.add_argument("--out", default="docs/images/site_planning.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = sites.by_operator(args.operator)
    sea = [s for s in selected if s.is_sea_site and s.is_active and s.has_coordinates]
    sea = sorted(sea, key=lambda s: -s.capacity)[:args.max_sites]
    if not sea:
        raise SystemExit(f"No sea sites for operator {args.operator!r}.")

    plans = []
    for s in sea:
        try:
            fc = ocean.fetch_forecast(s.lat, s.lon)
        except Exception as exc:
            LOGGER.warning("%s: %s", s.name, exc)
            continue
        windows = ocean.inspection_windows(fc)
        temps = [x.sea_water_temperature_c for x in fc
                 if x.sea_water_temperature_c is not None]
        plans.append({"site": s, "windows": windows,
                      "summary": ocean.summarise_windows(windows),
                      "temp": temps[0] if temps else None})
    if not plans:
        raise SystemExit("No forecasts retrieved.")

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.0), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.55, 1.0]})
    fig.subplots_adjust(wspace=0.30, top=0.72, bottom=0.22, left=0.075, right=0.98)

    # ---- Left: hourly inspection windows per site --------------------------
    ax = axes[0]
    _style(ax, grid_axis="x")
    ax.grid(False)
    hours = min(len(p["windows"]) for p in plans)
    for row, p in enumerate(plans):
        for h, w in enumerate(p["windows"][:hours]):
            ax.barh(row, 1, left=h, height=0.66,
                    color=RATING_COLOURS.get(w.rating, NEUTRAL),
                    edgecolor="none")
    ax.set_yticks(range(len(plans)))
    ax.set_yticklabels([p["site"].name.title()[:18] for p in plans],
                       fontsize=9, color=INK_SOFT)
    ax.invert_yaxis()
    ax.set_xlim(0, hours)
    step = 24
    ax.set_xticks(range(0, hours + 1, step))
    ax.set_xticklabels([f"+{d}d" if d else "now"
                        for d in range(0, hours // step + 1)],
                       fontsize=9, color=INK_SOFT)
    ax.set_xlabel("hours ahead (MET Norway ocean forecast)", fontsize=9.5,
                  color=INK_SOFT)
    ax.set_title("A · When each site is flyable for ROV net inspection",
                 fontsize=11, color=INK, fontweight="bold", pad=10, loc="left")
    for row, p in enumerate(plans):
        ax.text(hours + 2, row, f"{p['summary']['good_fraction']:.0%} good",
                va="center", fontsize=8.5, color=INK_SOFT)
    handles = [plt.Rectangle((0, 0), 1, 1, color=RATING_COLOURS[k])
               for k in ("good", "marginal", "poor")]
    ax.legend(handles, ["good", "marginal", "poor"], frameon=False, fontsize=8.5,
              ncol=3, loc="lower left", bbox_to_anchor=(0, -0.30),
              labelcolor=INK_SOFT, handlelength=1.4, handleheight=0.9)

    # ---- Right: thermal state vs the size-dependent optimum ----------------
    ax = axes[1]
    _style(ax)
    weights = np.logspace(np.log10(2), np.log10(6000), 200)
    opt = [ocean.thermal_optimum_c(w) for w in weights]
    ax.plot(weights, opt, color="#2a78d6", lw=2.4, zorder=3,
            label="cod thermal optimum")
    ax.set_xscale("log")

    temps = [p["temp"] for p in plans if p["temp"] is not None]
    if temps:
        t_lo, t_hi = min(temps), max(temps)
        ax.axhspan(t_lo, t_hi, color="#eb6834", alpha=0.16, zorder=1)
        ax.axhline((t_lo + t_hi) / 2, color="#eb6834", lw=2.0, ls="--", zorder=2,
                   label="sea temperature now, these sites")
        ax.text(2.4, (t_lo + t_hi) / 2 + 0.32,
                f"{t_lo:.1f}–{t_hi:.1f} °C today", fontsize=8.5, color="#b8461a")

        w = args.fish_weight_g
        o = ocean.thermal_optimum_c(w)
        ax.plot([w], [o], marker="o", markersize=10, color="#2a78d6",
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
        ax.annotate(f"{w / 1000:.0f} kg fish · optimum {o:.1f} °C\n"
                    f"{(t_lo + t_hi) / 2 - o:+.1f} °C above optimum today",
                    xy=(w, o), xytext=(w * 0.030, o + 1.0), fontsize=8.5,
                    color=INK_SOFT, ha="left",
                    arrowprops=dict(arrowstyle="->", color=INK_FAINT, lw=0.9))
    ax.set_xlabel("fish weight (g, log scale)", fontsize=9.5, color=INK_SOFT)
    ax.set_ylabel("temperature (°C)", fontsize=9.5, color=INK_SOFT)
    ax.set_title("B · Warm water helps a hatchery and hurts a grow-out pen",
                 fontsize=11, color=INK, fontweight="bold", pad=10, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=INK_SOFT)

    fig.suptitle("Planning inspections from open data — licensed sites × live ocean forecast",
                 fontsize=13.5, color=INK, fontweight="bold", x=0.075, ha="left", y=0.955)
    fig.text(0.075, 0.885,
             "Sites: Fiskeridirektoratet cod-locality register (NLOD).  Ocean: MET Norway oceanforecast 2.0 (NLOD + CC BY 4.0).  "
             "Neither needs a key.\nWindow ratings compare forecast wave height and current against the flight conditions this repo's models were "
             "characterised in — a similarity check, not a certified operating limit.",
             fontsize=8.8, color=INK_SOFT, ha="left", va="top", linespacing=1.5)

    out = Path(args.out)
    ensure_dir(out.parent)
    fig.savefig(out, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
