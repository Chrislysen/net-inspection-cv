"""Shared helpers for the CLI scripts (path bootstrap, config loading).

Importing this module makes the ``netinspect`` package importable even without
``pip install -e .`` by adding the ``src/`` directory to ``sys.path``.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load_yaml(path: str | Path) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
