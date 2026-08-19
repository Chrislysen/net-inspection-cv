"""The licence classifier behind the SBOM.

An SBOM is only worth generating if its classification is right, and the one
classification this project genuinely depends on is AGPL. The shipped YOLO path
uses Ultralytics (AGPL-3.0), which is viral over a network — serving a model over
HTTP is the triggering use — and the permissive torchvision path exists to avoid
exactly that. A classifier that quietly called AGPL "permissive" would turn the
document into false assurance, which is worse than having none.

The substring traps are real: "AGPL" contains "GPL", and "LGPL" contains "GPL"
while carrying a materially weaker obligation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

make_sbom = pytest.importorskip("make_sbom")


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("licence", [
    "AGPL-3.0",
    "AGPL-3.0-or-later",
    "GNU Affero General Public License v3",
    "GPL-3.0",
    "GPL-2.0-only",
    "GNU General Public License v3 (GPLv3)",
])
def test_strong_copyleft_is_recognised(licence):
    assert make_sbom.classify(licence) == "copyleft", (
        f"{licence!r} must be flagged: this is the finding the document exists for")


@pytest.mark.parametrize("licence", [
    "LGPL-3.0",
    "GNU Lesser General Public License v3 (LGPLv3)",
    "MPL-2.0",
    "Mozilla Public License 2.0 (MPL 2.0)",
])
def test_weak_copyleft_is_not_confused_with_strong(licence):
    """LGPL contains the substring 'GPL' but does not reach your code the same way."""
    assert make_sbom.classify(licence) == "weak-copyleft", (
        f"{licence!r} was misclassified — LGPL/MPL obligations are file- or "
        "library-level, and calling them strong copyleft would block a build "
        "that is actually fine")


@pytest.mark.parametrize("licence", [
    "MIT",
    "MIT License",
    "BSD-3-Clause",
    "Apache-2.0",
    "Apache Software License",
    "ISC",
    "Python Software Foundation License",
])
def test_permissive_is_recognised(licence):
    assert make_sbom.classify(licence) == "permissive"


def test_missing_metadata_is_reported_not_guessed():
    """A blank licence is a finding. Guessing one is how an SBOM lies."""
    assert make_sbom.classify("UNKNOWN") == "unknown"


def test_an_unrecognised_licence_is_not_silently_called_permissive():
    assert make_sbom.classify("Sleepycat") == "other"
    assert make_sbom.classify("Commercial, all rights reserved") == "other"


# --------------------------------------------------------------------------- #
# the document
# --------------------------------------------------------------------------- #
def test_the_document_is_valid_cyclonedx():
    doc = make_sbom.build_document(make_sbom.collect())
    assert doc["bomFormat"] == "CycloneDX"
    assert doc["specVersion"] == "1.5"
    assert doc["metadata"]["component"]["name"] == "net-inspection-cv"
    assert doc["components"], "no components collected"
    assert json.dumps(doc), "document must be JSON-serialisable"


def test_every_component_carries_a_purl_and_a_classification():
    for c in make_sbom.collect():
        assert c["purl"].startswith("pkg:pypi/"), c
        assert c["properties"][0]["name"] == "netinspect:licence_class"
        assert c["properties"][0]["value"] in (
            "copyleft", "weak-copyleft", "permissive", "other", "unknown")


def test_ultralytics_is_flagged_when_it_is_installed():
    """The project's central licence claim, checked against the real environment."""
    comps = {c["name"].lower(): c for c in make_sbom.collect()}
    if "ultralytics" not in comps:
        pytest.skip("ultralytics is not installed in this environment")
    assert comps["ultralytics"]["properties"][0]["value"] == "copyleft", (
        "Ultralytics is AGPL-3.0 and the SBOM did not flag it — the permissive "
        "path exists because of this, and the document is the evidence for it")
