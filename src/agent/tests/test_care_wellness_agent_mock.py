"""
test_care_wellness_agent_mock.py — Mock test suite for CareWellnessAgent.

No external credentials required. Covers happy paths, response content,
no-contact escalation, dispatch failure, and regression checks.

Run all:    pytest src/agent/tests/test_care_wellness_agent_mock.py -v
By marker:  pytest src/agent/tests/test_care_wellness_agent_mock.py -v -m happy
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from agent.agents.care_wellness.agent import CareWellnessAgent
from agent.tests.fixtures import make_verified_state

# ---------------------------------------------------------------------------
# State helper
# ---------------------------------------------------------------------------

FAX_ON_FILE = "6175554101"
EMAIL_ON_FILE = "emily@example.com"


def make_cw_state(**overrides) -> dict:
    """Verified state with delivery contact confirmed."""
    defaults: dict = {
        "call_intent": "provider_services",
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


def _msg(role: str, text: str) -> dict:
    return {"role": role, "content": text}


# ---------------------------------------------------------------------------
# Runner + assertion helpers
# ---------------------------------------------------------------------------


async def _run(state: dict) -> dict:
    return await CareWellnessAgent.from_state(state).execute(state)


def is_complete(result: dict) -> bool:
    return result.get("next_node") == "orchestrator" and result.get("is_interrupt") is False


def is_escalation(result: dict) -> bool:
    return result.get("next_node") == "escalation_agent" and result.get("is_interrupt") is False


def get_response(result: dict) -> str:
    msg = result.get("messages", {})
    if isinstance(msg, dict):
        return msg.get("content", "")
    if isinstance(msg, list) and msg:
        last = msg[-1]
        return last.get("content", "") if isinstance(last, dict) else str(last)
    return ""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_dispatch(monkeypatch) -> AsyncMock:
    mock = AsyncMock(return_value=None)
    monkeypatch.setattr("agent.agents.care_wellness.agent.dispatch_care_coach", mock)
    return mock


@pytest.fixture(autouse=True)
def _base_mocks(mock_dispatch):
    pass


# ---------------------------------------------------------------------------
# SECTION 1 — Happy path (marker: happy)
# ---------------------------------------------------------------------------


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_fax_dispatch_and_complete(mock_dispatch) -> None:
    """Fax path: dispatches Care Coach details and signals complete."""
    state = make_cw_state(
        delivery_method="fax",
        messages=[_msg("user", "yes please send it")],
    )
    result = await _run(state)
    assert is_complete(result)
    assert result.get("care_coach_details_sent") is True
    mock_dispatch.assert_called_once()


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_email_dispatch_and_complete(mock_dispatch) -> None:
    """Email path: dispatches Care Coach details and signals complete."""
    state = make_cw_state(
        delivery_method="email",
        messages=[_msg("user", "yes please send it")],
    )
    result = await _run(state)
    assert is_complete(result)
    assert result.get("care_coach_details_sent") is True
    mock_dispatch.assert_called_once()


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_early_exit_already_sent(mock_dispatch) -> None:
    """care_coach_details_sent=True → immediate complete, no dispatch call."""
    state = make_cw_state(care_coach_details_sent=True)
    result = await _run(state)
    assert is_complete(result)
    mock_dispatch.assert_not_called()


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_fax_preferred_over_email_when_delivery_method_fax(mock_dispatch) -> None:
    """delivery_method=fax with both fax and email set → dispatch uses fax."""
    state = make_cw_state(
        delivery_method="fax",
        fax=FAX_ON_FILE,
        email=EMAIL_ON_FILE,
        messages=[_msg("user", "yes")],
    )
    result = await _run(state)
    assert is_complete(result)
    call_kwargs = mock_dispatch.call_args
    _, kwargs_positional = call_kwargs[0], call_kwargs[1] if call_kwargs[1] else {}
    # dispatch_care_coach is called as dispatch_care_coach(self, state, method, contact)
    # positional args: (agent_self, state, method, contact)
    args = call_kwargs[0]
    assert args[2] == "fax", f"Expected method='fax', got {args[2]!r}"
    assert args[3] == FAX_ON_FILE, f"Expected contact={FAX_ON_FILE!r}, got {args[3]!r}"


# ---------------------------------------------------------------------------
# SECTION 2 — Response content (marker: response_check)
# ---------------------------------------------------------------------------


@pytest.mark.response_check
@pytest.mark.asyncio
async def test_response_contains_30_minutes(mock_dispatch) -> None:
    """Confirmation message must mention the 30-minute delivery window."""
    state = make_cw_state(messages=[_msg("user", "yes please")])
    result = await _run(state)
    assert is_complete(result)
    assert "30 minutes" in get_response(result), (
        f"Expected '30 minutes' in response: {get_response(result)!r}"
    )


@pytest.mark.response_check
@pytest.mark.asyncio
async def test_response_contains_contact_detail(mock_dispatch) -> None:
    """Confirmation message must mention the delivery contact (fax or email)."""
    state = make_cw_state(
        delivery_method="fax",
        fax=FAX_ON_FILE,
        messages=[_msg("user", "yes")],
    )
    result = await _run(state)
    assert is_complete(result)
    response = get_response(result)
    assert FAX_ON_FILE in response or "fax" in response.lower(), (
        f"Expected fax contact in response: {response!r}"
    )


@pytest.mark.response_check
@pytest.mark.asyncio
async def test_completion_flags_set(mock_dispatch) -> None:
    """After dispatch, care_coach_details_sent and care_coach_offered must be True."""
    state = make_cw_state(messages=[_msg("user", "yes")])
    result = await _run(state)
    assert is_complete(result)
    assert result.get("care_coach_details_sent") is True
    assert result.get("care_coach_offered") is True


# ---------------------------------------------------------------------------
# SECTION 3 — No contact escalation (marker: unhappy)
# ---------------------------------------------------------------------------


@pytest.mark.unhappy
@pytest.mark.asyncio
async def test_no_contact_on_file_escalates(mock_dispatch) -> None:
    """No fax and no email in state → escalate instead of dispatch."""
    state = make_cw_state(
        fax="",
        email="",
        delivery_method="",
        messages=[_msg("user", "yes please")],
    )
    result = await _run(state)
    assert is_escalation(result)
    mock_dispatch.assert_not_called()


# ---------------------------------------------------------------------------
# SECTION 4 — Dispatch failure (marker: dispatch_fail)
# ---------------------------------------------------------------------------


@pytest.mark.dispatch_fail
@pytest.mark.asyncio
async def test_dispatch_failure_escalates(monkeypatch) -> None:
    """dispatch_care_coach returning an escalation dict → agent returns escalation."""
    fake_escalation = {
        "next_node": "escalation_agent",
        "is_interrupt": False,
        "messages": {"role": "assistant", "content": "Could not dispatch Care Coach details."},
        "slot_attempts": {},
        "metadata_events": [],
        "app_run_id": "",
        "awaiting_slot": "",
        "last_agent_signal": {},
        "active_agent": "care_wellness_agent",
    }
    monkeypatch.setattr(
        "agent.agents.care_wellness.agent.dispatch_care_coach",
        AsyncMock(return_value=fake_escalation),
    )
    state = make_cw_state(messages=[_msg("user", "yes please")])
    result = await _run(state)
    assert is_escalation(result)


# ---------------------------------------------------------------------------
# SECTION 5 — Regression (marker: regression)
# ---------------------------------------------------------------------------


@pytest.mark.regression
@pytest.mark.asyncio
async def test_delivery_method_fallback_to_fax_when_method_unset(mock_dispatch) -> None:
    """delivery_method="" but fax is set → resolver falls back to fax."""
    state = make_cw_state(
        delivery_method="",
        fax=FAX_ON_FILE,
        email="",
        messages=[_msg("user", "yes")],
    )
    result = await _run(state)
    assert is_complete(result)
    args = mock_dispatch.call_args[0]
    assert args[2] == "fax", f"Expected fallback to 'fax', got {args[2]!r}"
    assert args[3] == FAX_ON_FILE
