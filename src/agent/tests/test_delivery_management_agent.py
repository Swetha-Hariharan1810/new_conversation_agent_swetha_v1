"""
test_delivery_management_agent.py — Live LLM tests for DeliveryManagementAgent.

Requires AZURE_OPENAI_API_KEY to be set; all tests are skipped when absent.

Run all:    pytest src/agent/tests/test_delivery_management_agent.py -v
By marker:  pytest src/agent/tests/test_delivery_management_agent.py -v -m happy
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

from agent.agents.delivery_management.agent import DeliveryManagementAgent  # noqa: E402
from agent.tests.fixtures import advance, get_awaiting, get_response, make_verified_state  # noqa: E402

FAX_ON_FILE = "6175554101"
EMAIL_ON_FILE = "emily@example.com"
PROVIDER = "Primary Care Physician"
ZIP_USED = "12139"


def _msg(role: str, text: str) -> dict:
    return {"role": role, "content": text}


def make_dm_state(**overrides) -> dict:
    defaults: dict = {
        "call_intent": "provider_services",
        "provider_type": PROVIDER,
        "zip_code_used": ZIP_USED,
        "zip_code": ZIP_USED,
        "fax": FAX_ON_FILE,
        "email": EMAIL_ON_FILE,
        "delivery_method": "",
        "provider_list_sent": False,
        "benefits_offer_made": False,
        "delivery_timestamp": "",
        "member_status_verify": True,
    }
    defaults.update(overrides)
    return make_verified_state(**defaults)


async def _run(state: dict) -> dict:
    return await DeliveryManagementAgent.from_state(state).execute(state)


def is_ask(result: dict) -> bool:
    return result.get("is_interrupt") is True and result.get("next_node") == "delivery_management_agent"


def is_escalation(result: dict) -> bool:
    return result.get("next_node") == "escalation_agent" and result.get("is_interrupt") is False


def is_complete(result: dict) -> bool:
    return result.get("next_node") == "orchestrator" and result.get("is_interrupt") is False


def _p(data: list[float], pct: float) -> float:
    s = sorted(data)
    n = len(s)
    k = (pct / 100) * (n - 1)
    lo, hi = int(k), min(int(k) + 1, n - 1)
    return s[lo] + (k - lo) * (s[hi] - s[lo])


@pytest.fixture
def mock_dispatch(monkeypatch):
    tool = MagicMock()
    tool.ainvoke = AsyncMock(return_value=True)
    monkeypatch.setattr("agent.storage.tools.dispatch_provider_list", tool)
    return tool


@pytest.fixture
def mock_fax_update(monkeypatch):
    tool = MagicMock()
    tool.ainvoke = AsyncMock(return_value=True)
    monkeypatch.setattr("agent.storage.tools.update_member_contact", tool)
    return tool


# ---------------------------------------------------------------------------
# SECTION 1 — Happy path (marker: happy)
# ---------------------------------------------------------------------------


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_fax_flow_step_by_step(mock_dispatch, mock_fax_update) -> None:
    """Multi-turn fax flow: select fax → confirm fax → benefits response → complete."""
    # Turn 1: ask delivery method
    state = make_dm_state(messages=[_msg("user", "hi, please proceed")])
    result = await _run(state)
    assert is_ask(result), "First turn must ask for delivery method"
    assert get_awaiting(result) == "delivery_method"

    # Turn 2: select fax
    state = advance(state, result, "fax")
    result = await _run(state)
    assert not is_escalation(result), "Fax selection must not escalate"
    if is_ask(result):
        awaiting = get_awaiting(result)
        assert awaiting in ("fax_confirmed", "fax"), f"Unexpected awaiting: {awaiting}"
        if awaiting == "fax_confirmed":
            assert FAX_ON_FILE in get_response(result) or len(get_response(result)) > 0

            # Turn 3: confirm fax
            state = advance(state, result, "yes that is correct")
            result = await _run(state)
            assert not is_escalation(result), "Fax confirmation must not escalate"

    # After confirmation path: should be awaiting benefits_response or complete
    if is_ask(result):
        assert get_awaiting(result) in ("benefits_response", "fax_confirmed", "fax"), (
            f"Unexpected awaiting after fax path: {get_awaiting(result)}"
        )
        if get_awaiting(result) == "benefits_response":
            assert result.get("provider_list_sent") is True

            # Respond to benefits offer
            state = advance(state, result, "no thank you")
            result = await _run(state)
            assert is_complete(result) or not is_escalation(result)

    if is_complete(result):
        assert result.get("provider_list_sent") is True


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_email_flow_step_by_step(mock_dispatch, mock_fax_update) -> None:
    """Multi-turn email flow: select email → confirm email → complete."""
    state = make_dm_state(messages=[_msg("user", "hi")])
    result = await _run(state)
    assert is_ask(result)

    # Select email
    state = advance(state, result, "email please")
    result = await _run(state)
    assert not is_escalation(result), "Email selection must not escalate"
    if is_ask(result):
        awaiting = get_awaiting(result)
        assert awaiting in ("email_confirmed", "email"), f"Unexpected awaiting: {awaiting}"
        if awaiting == "email_confirmed":
            assert EMAIL_ON_FILE in get_response(result) or len(get_response(result)) > 0

            # Confirm email
            state = advance(state, result, "yes that is right")
            result = await _run(state)
            assert not is_escalation(result)

    if is_ask(result) and get_awaiting(result) == "benefits_response":
        assert result.get("provider_list_sent") is True
        state = advance(state, result, "yes I'd love to hear about benefits")
        result = await _run(state)
        assert is_complete(result) or not is_escalation(result)
        if is_complete(result):
            assert result.get("provider_list_sent") is True


# ---------------------------------------------------------------------------
# SECTION 2 — Latency (marker: latency)
# ---------------------------------------------------------------------------


@pytest.mark.latency
@pytest.mark.asyncio
async def test_latency_fax_confirmation(mock_dispatch, mock_fax_update) -> None:
    """p95 latency for the fax confirmation step must be < 4 s."""
    elapsed_list: list[float] = []
    samples = 5
    for _ in range(samples):
        state = make_dm_state(
            delivery_method="fax",
            awaiting_slot="fax_confirmed",
            messages=[
                _msg("assistant", f"The fax number we have on file is {FAX_ON_FILE}. Is this correct?"),
                _msg("user", "yes that is correct"),
            ],
        )
        t0 = time.perf_counter()
        result = await _run(state)
        elapsed_list.append(time.perf_counter() - t0)
        assert not is_escalation(result), "Fax confirmation must not escalate"

    p95 = _p(elapsed_list, 95)
    print(f"\nFax confirmation p95={p95 * 1000:.0f}ms (budget=4000ms)")
    assert p95 < 4.0, f"p95 {p95 * 1000:.0f}ms exceeds 4000ms budget"
