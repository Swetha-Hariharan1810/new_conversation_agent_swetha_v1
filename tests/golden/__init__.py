"""
tests.golden — Deterministic, hermetic golden baseline for the multi-intent
(context-retention) defect described in pending_action_items (UAT-007).

Unlike tests/live_e2e (which drives the real graph against live Azure OpenAI +
Salesforce), this package is fully deterministic: the LLM and storage layers are
replaced with scripted fakes so the *current, broken* behavior is locked under
test with zero secrets and zero network. Assertions describe today's behavior and
are designed to be flipped, one phase at a time, as the architecture is rebuilt.
"""

import sys
from pathlib import Path

# Mirror tests/live_e2e: ensure src/ is importable when this package is imported
# directly (e.g. `pytest tests/golden`), independent of the pyproject pythonpath.
_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
