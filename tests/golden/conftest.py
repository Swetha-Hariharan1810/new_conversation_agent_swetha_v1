"""
conftest.py — path setup for the hermetic golden suite.

Ensures both the repo root (so ``tests.golden.*`` imports resolve) and ``src``
(so ``agent.*`` imports resolve) are on sys.path before any test module loads,
independent of how pytest is invoked or its rootdir/import-mode detection.
"""

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"

for _p in (str(_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture(autouse=True)
def _understanding_decode_default():
    """Pin the promoted (Phase 3B) understanding decode before each golden test
    and restore it after, so per-test toggles (kill-switch / shadow capture) can
    never leak the decoder state across tests."""
    from agent.orchestration import shadow

    shadow.set_shadow_decoder(shadow.heuristic_decoder)
    try:
        yield
    finally:
        shadow.set_shadow_decoder(shadow.heuristic_decoder)
