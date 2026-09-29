"""Ensure repo imports work regardless of how the dashboard is launched."""

from __future__ import annotations

import sys
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parent
ROOT = DASHBOARD_DIR.parent
SRC = ROOT / "src"


def ensure_repo_paths() -> Path:
    for path in (SRC, ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return ROOT


ensure_repo_paths()
