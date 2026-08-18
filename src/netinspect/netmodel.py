"""Place an inspected strip on a schematic sea cage, with a landmark to orient by.

A position like "3.1 m along the sweep, 1.7 m deep" is precise and unusable: it
is relative to wherever the pass happened to start. This module converts it into
something a person can act on — a bearing around the pen and a distance from a
**fixed landmark**, the feed barge — and reports what fraction of the net the
pass actually covered.

The honest part
---------------
The pen here is **declared, not measured**. Nothing in one SOLAQUA clip
identifies a pen radius: over a 5.5 m arc the deviation from a straight line is
inside USBL noise. So the operator supplies the cage dimensions (they know them —
a pen is a purchased object with a stated circumference) and this module places
the *measured* strip onto that *declared* shell.

Two kinds of number therefore come out of here, and they are kept separate
everywhere, including in the API responses and the UI:

* **measured** — along/across/depth of the strip and of each site, from
  :mod:`netinspect.mapping`.
* **declared** — pen circumference, cylinder depth, cone depth, the bearing the
  pass started at, and where the barge is moored.

A bearing is only as good as the declared start bearing. Say so rather than
printing a compass heading to one decimal and hoping nobody asks.

Cage shape
----------
A Norwegian sea cage is not a plain cylinder: a floating HDPE collar carries a
cylindrical net wall down to a depth, and below that the net tapers through a
**cone** to a bottom ring and centre weight, which is what keeps the net open
against current. Both sections are modelled, because a cone-section hole is a
different repair job — and a different escape risk — from a wall hole.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from .utils import get_logger

LOGGER = get_logger()

# A typical Norwegian production cage. Circumference is how pens are actually
# specified and sold, so it is the input rather than radius.
DEFAULT_CIRCUMFERENCE_M = 160.0
DEFAULT_CYLINDER_DEPTH_M = 15.0
DEFAULT_CONE_DEPTH_M = 10.0
DEFAULT_BARGE_BEARING_DEG = 0.0          # feed barge moored due north of the pen
DEFAULT_BARGE_OFFSET_M = 18.0            # from the pen wall, outward


@dataclass
class PenGeometry:
    """Declared dimensions of a cage. None of this is measured from the footage.

    Bearings are compass-style: 0 deg = north, increasing clockwise, so they read
    the same way as every other bearing on a farm.
    """
    circumference_m: float = DEFAULT_CIRCUMFERENCE_M
    cylinder_depth_m: float = DEFAULT_CYLINDER_DEPTH_M
    cone_depth_m: float = DEFAULT_CONE_DEPTH_M
    barge_bearing_deg: float = DEFAULT_BARGE_BEARING_DEG
    barge_offset_m: float = DEFAULT_BARGE_OFFSET_M
    start_bearing_deg: float = 0.0       # bearing of the pass's first frame
    clockwise: bool = True               # direction the vehicle travelled

    def __post_init__(self) -> None:
        if self.circumference_m <= 0:
            raise ValueError("circumference_m must be positive")
        if self.cylinder_depth_m < 0 or self.cone_depth_m < 0:
            raise ValueError("depths must be non-negative")
        if self.cylinder_depth_m + self.cone_depth_m <= 0:
            raise ValueError("a cage needs some depth")

    @property
    def radius_m(self) -> float:
        return self.circumference_m / (2.0 * math.pi)

    @property
    def total_depth_m(self) -> float:
        return self.cylinder_depth_m + self.cone_depth_m

    @property
    def cone_slant_m(self) -> float:
        """Slant height of the cone — the netting distance, not the vertical drop.

        With ``cone_depth_m == 0`` this degrades to the radius, which is the
        right answer rather than a special case: the taper collapses to a flat
        bottom panel and ``pi * r * slant`` becomes ``pi * r**2``. A flat bottom
        is still netting a fish can leave through, so it still counts as area.
        """
        return math.hypot(self.radius_m, self.cone_depth_m)

    @property
    def net_area_m2(self) -> float:
        """Total netting area: cylindrical wall plus the conical bottom.

        This is the denominator that makes a coverage number mean something. A
        pass that sounds thorough in metres is usually a rounding error of the
        net once you divide by it.
        """
        wall = self.circumference_m * self.cylinder_depth_m
        cone = math.pi * self.radius_m * self.cone_slant_m
        return wall + cone

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(radius_m=round(self.radius_m, 3),
                 total_depth_m=round(self.total_depth_m, 2),
                 net_area_m2=round(self.net_area_m2, 1),
                 declared=True,
                 note=("Cage dimensions are DECLARED by the operator, not measured "
                       "from the footage. One clip's arc is far too straight to "
                       "identify a pen radius."))
        return d


def _wrap_deg(deg: float) -> float:
    return deg % 360.0


def bearing_at(along_m: float, geom: PenGeometry) -> float:
    """Compass bearing of a point `along_m` into the pass, around the pen."""
    sweep = 360.0 * along_m / geom.circumference_m
    return _wrap_deg(geom.start_bearing_deg + (sweep if geom.clockwise else -sweep))


def radius_at_depth(depth_m: float, geom: PenGeometry) -> float:
    """Net radius at a given depth — constant down the wall, shrinking in the cone."""
    if depth_m <= geom.cylinder_depth_m or geom.cone_depth_m <= 0:
        return geom.radius_m
    into_cone = min(depth_m - geom.cylinder_depth_m, geom.cone_depth_m)
    return geom.radius_m * (1.0 - into_cone / geom.cone_depth_m)


@dataclass
class PenPosition:
    """A measured strip position expressed on the declared cage."""
    along_m: float
    depth_m: float
    bearing_deg: float
    x_m: float                  # east
    y_m: float                  # north
    z_m: float                  # up (negative below the surface)
    section: str                # "wall" or "cone"
    arc_from_barge_m: float
    side: str                   # "clockwise" / "anticlockwise" / "at the barge"

    def describe(self) -> str:
        where = ("level with the feed barge" if self.side == "at the barge"
                 else f"{self.arc_from_barge_m:.0f} m {self.side} around the ring "
                      f"from the feed barge")
        return (f"{where}, {self.depth_m:.1f} m deep on the {self.section} "
                f"(bearing {self.bearing_deg:.0f}°)")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["description"] = self.describe()
        return d


def place_on_pen(along_m: float, depth_m: float, geom: PenGeometry) -> PenPosition:
    """Put one measured (along, depth) position onto the declared cage."""
    bearing = bearing_at(along_m, geom)
    r = radius_at_depth(depth_m, geom)
    rad = math.radians(bearing)
    section = "cone" if depth_m > geom.cylinder_depth_m and geom.cone_depth_m > 0 else "wall"

    # Signed arc from the barge, taking the shorter way round — which is how
    # someone actually walks or drives it.
    delta = (bearing - geom.barge_bearing_deg + 540.0) % 360.0 - 180.0
    arc = abs(delta) / 360.0 * geom.circumference_m
    if arc < 0.5:
        side = "at the barge"
    else:
        side = "clockwise" if delta > 0 else "anticlockwise"

    return PenPosition(
        along_m=round(along_m, 3), depth_m=round(depth_m, 3),
        bearing_deg=round(bearing, 1),
        x_m=round(r * math.sin(rad), 3), y_m=round(r * math.cos(rad), 3),
        z_m=round(-depth_m, 3), section=section,
        arc_from_barge_m=round(arc, 1), side=side)


def barge_anchor(geom: PenGeometry) -> dict[str, Any]:
    """Where the feed barge sits, as the fixed thing everything else refers to.

    A farm has one unmistakable landmark and this is it: a moored platform with
    the feed silos, generators and the control room. "40 m anticlockwise from the
    barge" is a direction a person can follow; a raw compass bearing is not.
    """
    rad = math.radians(geom.barge_bearing_deg)
    dist = geom.radius_m + geom.barge_offset_m
    return {"bearing_deg": round(_wrap_deg(geom.barge_bearing_deg), 1),
            "x_m": round(dist * math.sin(rad), 2),
            "y_m": round(dist * math.cos(rad), 2),
            "distance_from_centre_m": round(dist, 2),
            "label": "feed barge",
            "declared": True}


def coverage_of_net(swept_area_m2: float, along_extent_m: float,
                    geom: PenGeometry) -> dict[str, Any]:
    """What fraction of the cage this pass actually looked at.

    The number that keeps a demo honest. A pass reported as "5.5 m of net swept"
    sounds substantial until it is divided by the netting area of a real cage,
    at which point it is a fraction of one percent — which is the true reason a
    single pass cannot support a clean bill of health.
    """
    area_fraction = swept_area_m2 / geom.net_area_m2 if geom.net_area_m2 else 0.0
    ring_fraction = along_extent_m / geom.circumference_m if geom.circumference_m else 0.0
    return {
        "swept_area_m2": round(swept_area_m2, 2),
        "net_area_m2": round(geom.net_area_m2, 1),
        "area_fraction": round(area_fraction, 6),
        "area_percent": round(100.0 * area_fraction, 3),
        "ring_fraction": round(ring_fraction, 4),
        "ring_percent": round(100.0 * ring_fraction, 2),
        "passes_to_cover_ring": max(1, math.ceil(1.0 / ring_fraction)) if ring_fraction else None,
        "note": ("One pass covers a band at one depth. Full coverage needs the "
                 "ring circled and the wall stacked in depth; the area fraction "
                 "is why a single clean pass is not a clean net."),
    }


def project_sites(sites: Sequence[Any], geom: PenGeometry) -> list[dict[str, Any]]:
    """Place mapped defect sites on the cage, keeping their measured evidence.

    Accepts :class:`netinspect.mapping.DefectSite` objects or the dicts they
    serialise to, so the API and the CLI can share one path.
    """
    out: list[dict[str, Any]] = []
    for s in sites:
        d = s if isinstance(s, dict) else s.to_dict()
        depth = d.get("depth_m")
        if depth is None:
            depth = 0.0
        pos = place_on_pen(float(d.get("along_m", 0.0)), float(depth), geom)
        out.append({
            "site_id": d.get("site_id"),
            "sightings": d.get("sightings"),
            "evidence": d.get("evidence") or d.get("confidence"),
            "median_width_mm": d.get("median_width_mm"),
            "median_height_mm": d.get("median_height_mm"),
            "max_score": d.get("max_score"),
            "measured": {"along_m": d.get("along_m"), "across_m": d.get("across_m"),
                         "depth_m": depth},
            "placed": pos.to_dict(),
        })
    return out


def build_scene(sites: Sequence[Any], track: Sequence[Any], coverage: dict[str, Any],
                geom: PenGeometry) -> dict[str, Any]:
    """Everything a viewer needs to draw the cage and what was inspected on it.

    Deliberately returns *parameters plus measured points*, not a mesh: the cage
    shell is cheap to generate client-side, and shipping a mesh would blur the
    line between the declared shell and the measured strip. The ``provenance``
    block states which is which, and the UI reads it rather than hard-coding the
    distinction.
    """
    pts = [p if isinstance(p, dict) else p.to_dict() for p in track]
    band = []
    for p in pts:
        depth = p.get("depth_m")
        if depth is None:
            depth = 0.0
        band.append({"along_m": p.get("along_m"), "depth_m": depth,
                     "bearing_deg": round(bearing_at(float(p.get("along_m", 0.0)), geom), 2),
                     "standoff_m": p.get("standoff_m"),
                     "footprint_m": round(float(p.get("mm_per_px") or 0.0) * 1280 / 1000.0, 3)})

    along_extent = float(coverage.get("along_extent_m") or 0.0)
    swept = float(coverage.get("swept_area_m2") or 0.0)
    return {
        "pen": geom.to_dict(),
        "barge": barge_anchor(geom),
        "band": band,
        "sites": project_sites(sites, geom),
        "coverage": coverage_of_net(swept, along_extent, geom),
        "gaps": coverage.get("gaps", []),
        "provenance": {
            "measured": ["band positions along the sweep", "band depth and standoff",
                         "site positions along the sweep", "site sizes in mm"],
            "declared": ["pen circumference", "cylinder depth", "cone depth",
                         "start bearing of the pass", "feed barge bearing"],
            "warning": ("The cage shell and the barge are a declared reference "
                        "frame, not a reconstruction. Bearings are only as good "
                        "as the declared start bearing."),
        },
    }
