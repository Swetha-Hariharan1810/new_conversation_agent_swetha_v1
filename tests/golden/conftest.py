"""
conftest.py — path setup for the hermetic golden suite.

Ensures both the repo root (so ``tests.golden.*`` imports resolve) and ``src``
(so ``agent.*`` imports resolve) are on sys.path before any test module loads,
independent of how pytest is invoked or its rootdir/import-mode detection.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"

for _p in (str(_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
