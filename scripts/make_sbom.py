"""Emit a CycloneDX SBOM, and say which components are copyleft.

Every corporate deployment review asks for a bill of materials, and for this
project the question behind that request is not paperwork. The shipped YOLO path
depends on Ultralytics, which is **AGPL-3.0**, and AGPL is viral over a network —
serving a model over HTTP is exactly the use that triggers it. The whole reason
:mod:`netinspect.permissive_baseline` exists is to offer a path without that
obligation. An SBOM is how you *prove* that claim rather than asserting it.

So this does two things:

* writes ``reports/results/sbom.cyclonedx.json`` — CycloneDX 1.5, the format
  procurement and vulnerability scanners actually consume;
* classifies every component's licence, and exits non-zero under ``--fail-on
  copyleft`` if any strong-copyleft package is present.

    python scripts/make_sbom.py                     # write it, print the summary
    python scripts/make_sbom.py --fail-on copyleft  # gate a permissive build

Licence text comes from the installed distributions' own metadata via
``importlib.metadata``, not from a hand-maintained table — a hardcoded list goes
stale silently, which is the one failure mode that makes an SBOM worse than
useless. Packages whose metadata declares nothing are reported as ``UNKNOWN``
rather than guessed at; that is a finding, not a blank.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import _common  # noqa: F401

from netinspect.utils import ensure_dir, get_logger

LOGGER = get_logger()
REPO = _common.REPO_ROOT

# Strong copyleft: the obligation reaches your code. AGPL additionally reaches
# it across a network, which is the case that matters for an inference service.
COPYLEFT = re.compile(r"\bAGPL|\bGPL-[23]|GNU General Public|GNU Affero", re.I)
WEAK_COPYLEFT = re.compile(r"\bLGPL|Mozilla Public|\bMPL-|\bEPL-", re.I)
PERMISSIVE = re.compile(r"\bMIT\b|\bBSD\b|Apache|\bISC\b|Python Software Foundation|"
                        r"\bPSF\b|Unlicense|\bZlib\b|HPND", re.I)


def _licence_of(dist: metadata.Distribution) -> str:
    """Best available licence string for one installed distribution.

    Three places carry it and none is reliable alone: the ``License`` field, the
    ``License-Expression`` field (PEP 639, newer), and the trove classifiers.
    Prefer the SPDX expression, then classifiers, then the free-text field --
    which is sometimes an entire licence document, so it is truncated.
    """
    meta = dist.metadata
    expr = meta.get("License-Expression")
    if expr:
        return expr.strip()

    classifiers = [c for c in meta.get_all("Classifier") or []
                   if c.startswith("License ::")]
    if classifiers:
        # "License :: OSI Approved :: MIT License" -> "MIT License"
        return "; ".join(sorted({c.split("::")[-1].strip() for c in classifiers}))

    raw = (meta.get("License") or "").strip()
    if raw and "\n" not in raw and len(raw) < 100:
        return raw
    if raw:
        return raw.splitlines()[0][:100] + " (truncated)"
    return "UNKNOWN"


def classify(licence: str) -> str:
    if licence == "UNKNOWN":
        return "unknown"
    if COPYLEFT.search(licence):
        return "copyleft"
    if WEAK_COPYLEFT.search(licence):
        return "weak-copyleft"
    if PERMISSIVE.search(licence):
        return "permissive"
    return "other"


def collect() -> list[dict]:
    seen, comps = set(), []
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        version = dist.version or "0"
        licence = _licence_of(dist)
        comps.append({
            "type": "library",
            "name": name,
            "version": version,
            "purl": f"pkg:pypi/{name.lower()}@{version}",
            "licenses": [{"license": {"name": licence}}],
            # Not part of the CycloneDX schema's required fields; carried as a
            # property so the classification travels with the document.
            "properties": [{"name": "netinspect:licence_class",
                            "value": classify(licence)}],
        })
    return sorted(comps, key=lambda c: c["name"].lower())


def build_document(comps: list[dict]) -> dict:
    from netinspect import __version__
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {
            "timestamp": now,
            "tools": [{"vendor": "net-inspection-cv", "name": "make_sbom.py",
                       "version": __version__}],
            "component": {"type": "application", "name": "net-inspection-cv",
                          "version": __version__},
        },
        "components": comps,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="reports/results/sbom.cyclonedx.json")
    ap.add_argument("--fail-on", choices=["copyleft", "unknown", "none"], default="none",
                    help="exit non-zero if any component falls in this class")
    args = ap.parse_args()

    comps = collect()
    by_class: dict[str, list[str]] = {}
    for c in comps:
        cls = c["properties"][0]["value"]
        by_class.setdefault(cls, []).append(f"{c['name']} {c['version']}")

    out = Path(args.out)
    ensure_dir(out.parent)
    out.write_text(json.dumps(build_document(comps), indent=2), encoding="utf-8")

    print(f"\n{len(comps)} components -> {out}\n")
    for cls in ("copyleft", "weak-copyleft", "unknown", "other", "permissive"):
        names = by_class.get(cls, [])
        if names:
            print(f"{cls:15s} {len(names):4d}")
    print()

    copyleft = by_class.get("copyleft", [])
    if copyleft:
        print("STRONG COPYLEFT — these reach your code, and AGPL reaches it over a")
        print("network, which is what serving a model over HTTP is:")
        for n in copyleft:
            print(f"  - {n}")
        print("\nThe permissive path (netinspect.permissive_baseline, torchvision")
        print("BSD-3-Clause) exists precisely to avoid this. See README 'Licensing'.")
    else:
        print("No strong-copyleft components in this environment.")

    unknown = by_class.get("unknown", [])
    if unknown:
        print(f"\n{len(unknown)} components declare no licence metadata "
              "(reported, not guessed):")
        for n in unknown[:10]:
            print(f"  - {n}")
        if len(unknown) > 10:
            print(f"  ... and {len(unknown) - 10} more")

    if args.fail_on != "none" and by_class.get(args.fail_on):
        raise SystemExit(f"\nFAILED: {len(by_class[args.fail_on])} "
                         f"{args.fail_on} component(s) present.")


if __name__ == "__main__":
    main()
