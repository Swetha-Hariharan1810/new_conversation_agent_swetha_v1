"""
test_care_wellness_agent.py — Live tests for CareWellnessAgent.

Requires AZURE_OPENAI_API_KEY to be set; all tests are skipped when absent.
The dispatch tool is mocked — no real Salesforce calls are made.

CareWellnessAgent makes no LLM call in its primary path, so latency
is dominated only by Python overhead and the (mocked) tool call.

Run all:    pytest src/agent/tests/test_care_wellness_agent.py -v
By marker:  pytest src/agent/tests/test_care_wellness_agent.py -v -m happy
"""

from __future__ import annotations

import os
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.tests.helpers import load_test_env as _load_env

_load_env()

CREDS_AVAILABLE = bool(os.getenv("AZURE_OPENAI_API_KEY"))
pytestmark = pytest.mark.skipif(not CREDS_AVAILABLE, reason="AZURE_OPENAI_API_KEY not set")

from agent.agents.care_wellness.agent import CareWellnessAgent  # noqa: E402
from agent.tests.fixtures import make_verified_state  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FAX_ON_FILE = "6175554101"
EMAIL_ON_FILE = "emily@example.com"


def _msg(role: str, text: str) -> dict:
    return {"role": role, "content": text}


def make_cw_state(**overrides) -> dict:
    defaults: dict = {
        "call_intent": "provider_services",
        "member_id": "M907503",
        "member_status_verify": True,
        "delivery_method": "fax",
        "fax": FAX_ON_FILE,
        "email": EMAIL_ON_FILE,
        "care_coach_details_sent": False,
        "care_coach_offered": False,
        "rewards_portal_shared": False,
    }
    defaults.update(overrides)
    return make_verified_state(**defaults)


async def _run(state: dict) -> dict:
    return await CareWellnessAgent.from_state(state).execute(state)


def is_complete(result: dict) -> bool:
    return result.get("next_node") == "orchestrator" and result.get("is_interrupt") is False


def is_escalation(result: dict) -> bool:
    return result.get("next_node") == "escalation_agent" and result.get("is_interrupt") is False


def _p(data: list[float], pct: float) -> float:
    s = sorted(data)
    n = len(s)
    k = (pct / 100) * (n - 1)
    lo, hi = int(k), min(int(k) + 1, n - 1)
    return s[lo] + (k - lo) * (s[hi] - s[lo])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_dispatch(monkeypatch) -> MagicMock:
    """Patch dispatch_care_coach_details tool so no real Salesforce call is made."""
    tool = MagicMock()
    tool.ainvoke = AsyncMock(return_value=True)
    monkeypatch.setattr("agent.storage.tools.dispatch_care_coach_details", tool)
    return tool


# ---------------------------------------------------------------------------
# SECTION 1 — Happy path (marker: happy)
# ---------------------------------------------------------------------------


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_fax_dispatch(mock_dispatch) -> None:
    """Fax path: agent dispatches details and returns complete."""
    state = make_cw_state(
        delivery_method="fax",
        messages=[_msg("user", "yes please send me the details")],
    )
    result = await _run(state)
    assert is_complete(result), f"Expected complete, got next_node={result.get('next_node')!r}"
    assert result.get("care_coach_details_sent") is True
    mock_dispatch.ainvoke.assert_called_once()


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_email_dispatch(mock_dispatch) -> None:
    """Email path: agent dispatches details and returns complete."""
    state = make_cw_state(
        delivery_method="email",
        fax="",
        messages=[_msg("user", "yes please email me the details")],
    )
    result = await _run(state)
    assert is_complete(result), f"Expected complete, got next_node={result.get('next_node')!r}"
    assert result.get("care_coach_details_sent") is True
    mock_dispatch.ainvoke.assert_called_once()


# ---------------------------------------------------------------------------
# SECTION 2 — Latency (marker: latency)
# ---------------------------------------------------------------------------


@pytest.mark.latency
@pytest.mark.asyncio
async def test_latency_care_wellness_dispatch(mock_dispatch) -> None:
    """p95 latency for a care wellness dispatch must be < 2 s (no LLM call)."""
    elapsed_list: list[float] = []
    samples = 5
    for _ in range(samples):
        state = make_cw_state(messages=[_msg("user", "yes please send the details")])
        t0 = time.perf_counter()
        result = await _run(state)
        elapsed_list.append(time.perf_counter() - t0)
        assert not is_escalation(result), "Dispatch must not escalate"

    p95 = _p(elapsed_list, 95)
    print(f"\nCare wellness dispatch p95={p95 * 1000:.0f}ms (budget=2000ms)")
    assert p95 < 2.0, f"p95 {p95 * 1000:.0f}ms exceeds 2000ms budget"
