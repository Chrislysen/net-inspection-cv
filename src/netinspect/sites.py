"""Norwegian aquaculture site registry (Fiskeridirektoratet, open data).

The Directorate of Fisheries publishes every licensed aquaculture locality as an
open ArcGIS feature service — no authentication, no key, NLOD-licensed. This
module wraps the cod-locality layer so a site list can be reconstructed from
public record: coordinates, licensed capacity, permit numbers, production form
and operator.

Why an inspection project cares
-------------------------------
An inspection system is only useful attached to real sites. Coordinates let the
ocean forecast in :mod:`netinspect.ocean` be queried per locality, which turns
"here is a model" into "here is when this site is inspectable and what the water
is doing". Licensed capacity and production form say which sites are grow-out
pens (where net integrity carries escape risk) versus land-based hatcheries
(where it does not).

Escape risk is the reason net inspection is regulated rather than optional:
``akvakulturdriftsforskriften`` requires nets to be checked regularly during
operation and requires inspections to be journalled, and NYTEK23 sets a
certified service cycle for the net itself. A site list is the first column of
that journal.

Licence
-------
Data is published under the Norwegian Licence for Open Government Data (NLOD).
Attribute it to Fiskeridirektoratet when reproducing.

Examples
--------
>>> from netinspect import sites
>>> ode = sites.by_operator("ODE")
>>> [s.name for s in ode if s.is_sea_site][:3]
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .utils import get_logger, read_json, write_json

LOGGER = get_logger()

ARCGIS_BASE = "https://gis.fiskeridir.no/server/rest/services/Yggdrasil"
COD_LAYER = f"{ARCGIS_BASE}/Akvakulturlokaliteter_torsk/MapServer/0/query"

CACHE_PATH = Path("data/raw/sites/cod_localities.json")

# Capacity unit codes used by the register.
UNIT_TONNES = "TN"
UNIT_COUNT = "STK"


@dataclass
class Site:
    """One licensed aquaculture locality.

    Attributes
    ----------
    loknr : int
        The locality number — the stable public identifier, also the key used by
        BarentsWatch and the Akvakulturregisteret web pages.
    capacity : float
        Licensed capacity in ``capacity_unit``: tonnes (``TN``) for grow-out
        pens, individual fish (``STK``) for hatcheries. Comparing the two
        directly is meaningless, which is why the unit is kept.
    placement : str
        ``SJØ`` (in sea) or ``LAND``. Only sea sites have nets to inspect.
    species, operators, permits, production_form : str
        As recorded in the register, comma-separated.
    """
    loknr: int
    name: str
    status: str
    lat: float | None
    lon: float | None
    county: str
    municipality: str
    capacity: float
    capacity_unit: str
    placement: str
    species: str
    operators: str
    permits: str
    production_form: str
    url: str = ""

    @property
    def is_sea_site(self) -> bool:
        """True for in-sea localities — the ones with a net to inspect."""
        return self.placement.upper().startswith("SJ")

    @property
    def is_active(self) -> bool:
        return self.status.upper() == "AKTIV"

    @property
    def has_coordinates(self) -> bool:
        return self.lat is not None and self.lon is not None

    @property
    def capacity_tonnes(self) -> float | None:
        """Licensed biomass in tonnes, or None when the unit is a fish count."""
        return self.capacity if self.capacity_unit == UNIT_TONNES else None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["is_sea_site"] = self.is_sea_site
        d["is_active"] = self.is_active
        return d


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _to_site(attrs: dict[str, Any]) -> Site:
    return Site(
        loknr=int(attrs.get("loknr") or 0),
        name=_clean(attrs.get("navn")),
        status=_clean(attrs.get("status_lokalitet")),
        lat=float(attrs["lat"]) if attrs.get("lat") is not None else None,
        lon=float(attrs["lon"]) if attrs.get("lon") is not None else None,
        county=_clean(attrs.get("fylke")),
        municipality=_clean(attrs.get("kommune")),
        capacity=float(attrs.get("kapasitet_lok") or 0.0),
        capacity_unit=_clean(attrs.get("kapasitet_unittype")),
        placement=_clean(attrs.get("plassering")),
        species=_clean(attrs.get("til_arter")),
        operators=_clean(attrs.get("til_innehavere")),
        permits=_clean(attrs.get("til_tillatelser")),
        production_form=_clean(attrs.get("til_produksjonsform")),
        url=_clean(attrs.get("lokalitet_url_ekstern") or attrs.get("lokalitet_url")),
    )


def fetch_cod_localities(timeout: int = 90, cache: str | Path | None = CACHE_PATH,
                         force: bool = False) -> list[Site]:
    """Fetch every licensed cod locality in Norway.

    Results are cached to JSON because the register changes on the order of
    weeks, not minutes, and an interview demo should not depend on a live
    network call succeeding.
    """
    cache_path = Path(cache) if cache else None
    if cache_path and cache_path.exists() and not force:
        raw = read_json(cache_path)
        LOGGER.info("Loaded %d cod localities from cache", len(raw))
        return [Site(**{k: v for k, v in s.items()
                        if k in Site.__dataclass_fields__}) for s in raw]

    params = urllib.parse.urlencode({
        "where": "1=1", "outFields": "*", "f": "json", "resultRecordCount": 4000,
    })
    req = urllib.request.Request(f"{COD_LAYER}?{params}",
                                 headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    if "error" in payload:
        raise RuntimeError(f"Fiskeridirektoratet API error: {payload['error']}")
    sites = [_to_site(f["attributes"]) for f in payload.get("features", [])]
    LOGGER.info("Fetched %d cod localities from Fiskeridirektoratet", len(sites))

    if cache_path:
        write_json([s.to_dict() for s in sites], cache_path)
    return sites


def by_operator(name: str, sites: Iterable[Site] | None = None) -> list[Site]:
    """Sites whose operator field contains ``name`` (case-insensitive).

    Matching is a substring test against the register's free-text operator
    field, which lists every permit holder on a locality. A site shared between
    two companies therefore matches both — that is the register's own model, not
    an approximation introduced here.
    """
    pool = list(sites) if sites is not None else fetch_cod_localities()
    needle = name.strip().lower()
    return [s for s in pool if needle in s.operators.lower()]


def summarise(sites: Iterable[Site]) -> dict[str, Any]:
    """Aggregate a site list into a value-chain summary."""
    sites = list(sites)
    sea = [s for s in sites if s.is_sea_site]
    land = [s for s in sites if not s.is_sea_site]
    tonnage = [s.capacity for s in sea if s.capacity_unit == UNIT_TONNES]
    counts = [s.capacity for s in land if s.capacity_unit == UNIT_COUNT]
    by_county: dict[str, int] = {}
    for s in sites:
        by_county[s.county] = by_county.get(s.county, 0) + 1
    return {
        "sites": len(sites),
        "active": sum(1 for s in sites if s.is_active),
        "sea_sites": len(sea),
        "land_sites": len(land),
        "licensed_tonnes_total": round(sum(tonnage), 1),
        "licensed_individuals_total": int(sum(counts)),
        "counties": dict(sorted(by_county.items(), key=lambda kv: -kv[1])),
        "production_forms": sorted({s.production_form for s in sites if s.production_form}),
        "note": ("Sea sites carry nets and therefore escape risk; land sites do "
                 "not. Tonnage and individual counts are different units and are "
                 "reported separately rather than summed."),
    }


__all__ = ["Site", "COD_LAYER", "CACHE_PATH", "UNIT_TONNES", "UNIT_COUNT",
           "fetch_cod_localities", "by_operator", "summarise"]
