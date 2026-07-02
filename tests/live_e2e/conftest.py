"""
conftest.py — path setup for the live E2E suite.

Ensures both the repo root (so ``tests.live_e2e.*`` imports resolve) and ``src``
(so ``agent.*`` imports resolve) are on sys.path before any test module loads,
independent of how pytest is invoked. Without this, collecting
``tests/live_e2e/test_live.py`` fails with ``ModuleNotFoundError: No module
named 'tests'`` because pyproject only puts ``src`` on the path.

The live tests themselves are gated behind the ``live`` marker (excluded by
default), so adding this path setup does not run any live services on a normal
``pytest`` invocation — it only makes the module collectable.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"

for _p in (str(_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
