"""
test_delivery_management_agent_mock.py — Mock test suite for DeliveryManagementAgent.

No external credentials required. Covers happy paths (fax + email), guard triggers,
contact confirmation flows, benefits offer, and retry exhaustion.

Run all:    pytest src/agent/tests/test_delivery_management_agent_mock.py -v
By marker:  pytest src/agent/tests/test_delivery_management_agent_mock.py -v -m happy
"""

from __future__ import annotations

import asyncio
import re
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.agents.delivery_management.agent import DeliveryManagementAgent
from agent.llm.schema import EventType, GuardType, WorkerResult
from agent.tests.fixtures import make_verified_state

# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

FAX_ON_FILE = "6175554101"
EMAIL_ON_FILE = "emily@example.com"
PROVIDER = "Primary Care Physician"
ZIP_USED = "12139"


def make_dm_state(**overrides) -> dict:
    """Verified state ready for delivery management."""
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
    }
    defaults.update(overrides)
    member_status_verify = defaults.pop("member_status_verify", True)
    state = make_verified_state(**defaults)
    state["member_status_verify"] = member_status_verify
    return state


def _msg(role: str, text: str) -> dict:
    return {"role": role, "content": text}


# ---------------------------------------------------------------------------
# WorkerResult factories
# ---------------------------------------------------------------------------

EMPTY_ANSWERED = WorkerResult(event_type=EventType.ANSWERED)


def answered_method(method: str) -> WorkerResult:
    return WorkerResult(
        extracted={"delivery_method": method},
        event_type=EventType.ANSWERED,
        guard=GuardType.NONE,
        guard_confidence=0.0,
    )


def answered_contact_yes() -> WorkerResult:
    return WorkerResult(
        extracted={"contact_confirmed": "yes"},
        event_type=EventType.ANSWERED,
        guard=GuardType.NONE,
        guard_confidence=0.0,
    )


def answered_contact_no() -> WorkerResult:
    return WorkerResult(
        extracted={"contact_confirmed": "no"},
        event_type=EventType.ANSWERED,
        guard=GuardType.NONE,
        guard_confidence=0.0,
    )


def answered_new_fax(fax: str = "4155553211") -> WorkerResult:
    return WorkerResult(
        extracted={"fax": fax},
        event_type=EventType.ANSWERED,
        guard=GuardType.NONE,
        guard_confidence=0.0,
    )


def answered_new_email(email: str = "new@example.com") -> WorkerResult:
    return WorkerResult(
        extracted={"email": email},
        event_type=EventType.ANSWERED,
        guard=GuardType.NONE,
        guard_confidence=0.0,
    )


def answered_benefits(response: str) -> WorkerResult:
    return WorkerResult(
        extracted={"benefits_response": response},
        event_type=EventType.ANSWERED,
        guard=GuardType.NONE,
        guard_confidence=0.0,
    )


def make_guard(guard: GuardType) -> WorkerResult:
    return WorkerResult(extracted=None, event_type=EventType.NONE, guard=guard, guard_confidence=0.95)


# ---------------------------------------------------------------------------
# Runner + assertion helpers
# ---------------------------------------------------------------------------


async def _run(state: dict) -> dict:
    return await DeliveryManagementAgent.from_state(state).execute(state)


def is_ask(result: dict) -> bool:
    return result.get("is_interrupt") is True and result.get("next_node") == "delivery_management_agent"


def is_escalation(result: dict) -> bool:
    return result.get("next_node") == "escalation_agent" and result.get("is_interrupt") is False


def is_complete(result: dict) -> bool:
    return result.get("next_node") == "orchestrator" and result.get("is_interrupt") is False


def get_awaiting(result: dict) -> str:
    return result.get("awaiting_slot", "")


def get_response(result: dict) -> str:
    msg = result.get("messages", {})
    if isinstance(msg, dict):
        return msg.get("content", "")
    if isinstance(msg, list) and msg:
        last = msg[-1]
        return last.get("content", "") if isinstance(last, dict) else str(last)
    return ""


def advance(state: dict, result: dict, user_text: str | None = None) -> dict:
    new_state = {**state}
    for key, val in result.items():
        if key == "messages":
            continue
        new_state[key] = val
    messages = list(state.get("messages") or [])
    if isinstance(result.get("messages"), dict):
        messages.append(result["messages"])
    if user_text is not None:
        messages.append({"role": "user", "content": user_text})
    new_state["messages"] = messages
    return new_state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_extraction(monkeypatch) -> AsyncMock:
    fake_llm = MagicMock()
    monkeypatch.setattr("agent.agents.delivery_management.agent.get_extraction_llm", lambda: fake_llm)
    mock = AsyncMock(return_value=EMPTY_ANSWERED)
    monkeypatch.setattr("agent.agents.delivery_management.agent.extract_delivery_management_decision", mock)
    return mock


@pytest.fixture
def mock_dispatch(monkeypatch) -> AsyncMock:
    mock = AsyncMock(return_value=None)
    monkeypatch.setattr("agent.agents.delivery_management.agent.dispatch_provider_list", mock)
    return mock


@pytest.fixture
def mock_fax_update(monkeypatch) -> AsyncMock:
    mock = AsyncMock(return_value=None)
    monkeypatch.setattr("agent.agents.delivery_management.agent.update_fax_in_salesforce", mock)
    return mock


@pytest.fixture
def mock_email_update(monkeypatch) -> AsyncMock:
    mock = AsyncMock(return_value=None)
    monkeypatch.setattr("agent.agents.delivery_management.agent.update_email_in_salesforce", mock)
    return mock


@pytest.fixture
def mock_recovery(monkeypatch) -> AsyncMock:
    async def _fn(*, slot_name, attempt, guard, last_messages, **kwargs):
        return f"[RECOVERY:{slot_name}:attempt{attempt}:guard{guard}]"

    mock = AsyncMock(side_effect=_fn)
    monkeypatch.setattr("agent.llm.response_generator.generate_recovery_message", mock)
    return mock


@pytest.fixture(autouse=True)
def _base_mocks(mock_extraction, mock_dispatch, mock_fax_update, mock_email_update, mock_recovery):
    pass


# ---------------------------------------------------------------------------
# SECTION 1 — Happy path: FAX (marker: happy)
# ---------------------------------------------------------------------------


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_first_turn_asks_delivery_method(mock_extraction) -> None:
    mock_extraction.return_value = EMPTY_ANSWERED
    state = make_dm_state(messages=[_msg("user", "hi")])
    result = await _run(state)
    assert is_ask(result)
    assert get_awaiting(result) == "delivery_method"


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_fax_confirmed_yes_full_flow(mock_extraction, mock_dispatch) -> None:
    """FAX happy path: fax → confirm yes → dispatch → benefits offer."""
    # Turn 1: collect delivery method
    mock_extraction.return_value = answered_method("fax")
    state = make_dm_state(messages=[_msg("user", "fax")])
    result1 = await _run(state)
    assert is_ask(result1)
    assert get_awaiting(result1) == "fax_confirmed"
    assert FAX_ON_FILE in get_response(result1)

    # Turn 2: confirm fax
    mock_extraction.return_value = answered_contact_yes()
    state2 = advance(state, result1, "yes")
    result2 = await _run(state2)
    assert is_ask(result2)
    assert get_awaiting(result2) == "benefits_response"
    assert result2.get("provider_list_sent") is True
    assert result2.get("benefits_offer_made") is True
    mock_dispatch.assert_called_once()

    # Turn 3: yes to benefits
    mock_extraction.return_value = answered_benefits("yes")
    state3 = advance(state2, result2, "yes")
    result3 = await _run(state3)
    assert is_complete(result3)
    assert result3.get("proactive_offer_available") is True


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_fax_confirmed_no_then_new_fax(mock_extraction, mock_dispatch, mock_fax_update) -> None:
    """FAX path: fax on file → no → collect new fax → dispatch → benefits."""
    state = make_dm_state(
        delivery_method="fax",
        awaiting_slot="fax_confirmed",
        messages=[_msg("assistant", f"Fax is {FAX_ON_FILE}?"), _msg("user", "no")],
    )
    # Turn 1: decline fax
    mock_extraction.return_value = answered_contact_no()
    result1 = await _run(state)
    assert is_ask(result1)
    assert get_awaiting(result1) == "fax"

    # Turn 2: provide new fax
    mock_extraction.return_value = answered_new_fax("4155553211")
    state2 = advance(state, result1, "4155553211")
    result2 = await _run(state2)
    assert is_ask(result2)
    assert get_awaiting(result2) == "benefits_response"
    assert result2.get("fax") == "4155553211"
    mock_fax_update.assert_called_once()
    mock_dispatch.assert_called_once()


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_fax_inline_new_number(mock_extraction, mock_dispatch, mock_fax_update) -> None:
    """Member says 'no, it's 4155553211' — inline new fax with the no."""
    state = make_dm_state(
        delivery_method="fax",
        awaiting_slot="fax_confirmed",
        messages=[_msg("assistant", f"Fax is {FAX_ON_FILE}?"), _msg("user", "no it's 4155553211")],
    )
    mock_extraction.return_value = WorkerResult(
        extracted={"fax": "4155553211"},
        event_type=EventType.ANSWERED,
        guard=GuardType.NONE,
        guard_confidence=0.0,
    )
    result = await _run(state)
    assert is_ask(result)
    assert get_awaiting(result) == "benefits_response"
    assert result.get("fax") == "4155553211"
    mock_fax_update.assert_called_once()
    mock_dispatch.assert_called_once()


# ---------------------------------------------------------------------------
# SECTION 2 — Happy path: EMAIL (marker: happy)
# ---------------------------------------------------------------------------


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_email_confirmed_yes(mock_extraction, mock_dispatch) -> None:
    state = make_dm_state(
        delivery_method="email",
        awaiting_slot="email_confirmed",
        messages=[_msg("assistant", f"Email is {EMAIL_ON_FILE}?"), _msg("user", "yes")],
    )
    mock_extraction.return_value = answered_contact_yes()
    result = await _run(state)
    assert is_ask(result)
    assert get_awaiting(result) == "benefits_response"
    assert result.get("email") == EMAIL_ON_FILE
    mock_dispatch.assert_called_once()


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_email_confirmed_no_then_new_email(
    mock_extraction, mock_dispatch, mock_email_update
) -> None:
    state = make_dm_state(
        delivery_method="email",
        awaiting_slot="email_confirmed",
        messages=[_msg("assistant", f"Email is {EMAIL_ON_FILE}?"), _msg("user", "no")],
    )
    mock_extraction.return_value = answered_contact_no()
    result1 = await _run(state)
    assert is_ask(result1) and get_awaiting(result1) == "email"

    mock_extraction.return_value = answered_new_email("new@example.com")
    state2 = advance(state, result1, "new@example.com")
    result2 = await _run(state2)
    assert is_ask(result2) and get_awaiting(result2) == "benefits_response"
    assert result2.get("email") == "new@example.com"
    mock_email_update.assert_called_once()
    mock_dispatch.assert_called_once()


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_benefits_no_completes(mock_extraction) -> None:
    state = make_dm_state(
        delivery_method="fax",
        provider_list_sent=True,
        benefits_offer_made=True,
        awaiting_slot="benefits_response",
        messages=[_msg("assistant", "Would you like benefits info?"), _msg("user", "no thanks")],
    )
    mock_extraction.return_value = answered_benefits("no")
    result = await _run(state)
    assert is_complete(result)
    assert result.get("proactive_offer_available") is False


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_early_exit_both_flags_set(mock_extraction) -> None:
    """Re-entry with provider_list_sent=True + benefits_offer_made=True → complete immediately."""
    state = make_dm_state(
        delivery_method="fax",
        provider_list_sent=True,
        benefits_offer_made=True,
        awaiting_slot="",
    )
    result = await _run(state)
    assert is_complete(result)
    mock_extraction.assert_not_called()


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_recovery_dispatched_offer_not_made(mock_extraction) -> None:
    """Recovery: dispatched but benefits_offer_made=False → make offer."""
    state = make_dm_state(
        delivery_method="fax",
        provider_list_sent=True,
        benefits_offer_made=False,
        awaiting_slot="",
        messages=[_msg("user", "hi")],
    )
    mock_extraction.return_value = EMPTY_ANSWERED
    result = await _run(state)
    assert is_ask(result)
    assert get_awaiting(result) == "benefits_response"
    assert result.get("benefits_offer_made") is True


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_no_contact_on_file_collects_directly(mock_extraction, mock_dispatch) -> None:
    """When no fax on file after fax selected, jump straight to collecting fax."""
    state = make_dm_state(
        delivery_method="fax",
        fax="",
        awaiting_slot="delivery_method",
        messages=[_msg("user", "fax")],
    )
    mock_extraction.return_value = answered_method("fax")
    result = await _run(state)
    assert is_ask(result)
    assert get_awaiting(result) == "fax"


# ---------------------------------------------------------------------------
# SECTION 3 — Guards (marker: guards)
# ---------------------------------------------------------------------------


@pytest.mark.guards
@pytest.mark.asyncio
async def test_guard_transfer_request(mock_extraction) -> None:
    mock_extraction.return_value = make_guard(GuardType.TRANSFER_REQUEST)
    state = make_dm_state(messages=[_msg("user", "transfer me")])
    assert is_escalation(await _run(state))


@pytest.mark.guards
@pytest.mark.asyncio
async def test_guard_abuse(mock_extraction) -> None:
    mock_extraction.return_value = make_guard(GuardType.ABUSE)
    state = make_dm_state(messages=[_msg("user", "you're useless")])
    assert is_escalation(await _run(state))


@pytest.mark.guards
@pytest.mark.asyncio
async def test_guard_self_harm(mock_extraction) -> None:
    mock_extraction.return_value = make_guard(GuardType.SELF_HARM)
    state = make_dm_state(messages=[_msg("user", "I want to end it")])
    assert is_escalation(await _run(state))


# ---------------------------------------------------------------------------
# SECTION 4 — Retry exhaustion (marker: retry)
# ---------------------------------------------------------------------------


@pytest.mark.retry
@pytest.mark.asyncio
async def test_fax_confirmed_first_failure_retries(mock_extraction) -> None:
    mock_extraction.return_value = EMPTY_ANSWERED
    state = make_dm_state(
        delivery_method="fax",
        awaiting_slot="fax_confirmed",
        messages=[_msg("assistant", f"Fax is {FAX_ON_FILE}?"), _msg("user", "uh")],
    )
    result = await _run(state)
    assert is_ask(result)
    assert get_awaiting(result) == "fax_confirmed"
    assert not is_escalation(result)


@pytest.mark.retry
@pytest.mark.asyncio
async def test_fax_confirmed_second_failure_escalates(mock_extraction) -> None:
    mock_extraction.return_value = EMPTY_ANSWERED
    state = make_dm_state(
        delivery_method="fax",
        awaiting_slot="fax_confirmed",
        slot_attempts={"fax_confirmed": {"attempt_count": 1, "confirmed": False, "last_value": None}},
        messages=[_msg("assistant", f"Fax is {FAX_ON_FILE}?"), _msg("user", "uh")],
    )
    result = await _run(state)
    assert is_escalation(result)


@pytest.mark.retry
@pytest.mark.asyncio
async def test_email_confirmed_second_failure_escalates(mock_extraction) -> None:
    mock_extraction.return_value = EMPTY_ANSWERED
    state = make_dm_state(
        delivery_method="email",
        awaiting_slot="email_confirmed",
        slot_attempts={"email_confirmed": {"attempt_count": 1, "confirmed": False, "last_value": None}},
        messages=[_msg("assistant", f"Email is {EMAIL_ON_FILE}?"), _msg("user", "uh")],
    )
    result = await _run(state)
    assert is_escalation(result)


@pytest.mark.retry
@pytest.mark.asyncio
async def test_delivery_method_retry_exhaustion_escalates(mock_extraction) -> None:
    mock_extraction.return_value = EMPTY_ANSWERED
    state = make_dm_state(
        awaiting_slot="delivery_method",
        slot_attempts={"delivery_method": {"attempt_count": 2, "confirmed": False, "last_value": None}},
        messages=[_msg("assistant", "Fax or email?"), _msg("user", "???")],
    )
    result = await _run(state)
    assert is_escalation(result)


@pytest.mark.retry
@pytest.mark.asyncio
async def test_benefits_retry_exhaustion_completes_gracefully(mock_extraction) -> None:
    """When benefits_response exhausts retries, complete with proactive_offer_available=False."""
    mock_extraction.return_value = EMPTY_ANSWERED
    state = make_dm_state(
        delivery_method="fax",
        provider_list_sent=True,
        benefits_offer_made=True,
        awaiting_slot="benefits_response",
        slot_attempts={"benefits_response": {"attempt_count": 2, "confirmed": False, "last_value": None}},
        messages=[_msg("assistant", "Benefits?"), _msg("user", "??")],
    )
    result = await _run(state)
    assert is_complete(result)
    assert result.get("proactive_offer_available") is False


# ---------------------------------------------------------------------------
# SECTION 5 — Dispatch failure (marker: dispatch_fail)
# ---------------------------------------------------------------------------


@pytest.mark.dispatch_fail
@pytest.mark.asyncio
async def test_dispatch_failure_escalates(mock_extraction, monkeypatch) -> None:
    fake_escalation = {
        "next_node": "escalation_agent",
        "is_interrupt": False,
        "messages": {"role": "assistant", "content": "Could not dispatch"},
        "slot_attempts": {},
        "metadata_events": [],
        "app_run_id": "",
        "awaiting_slot": "",
        "last_agent_signal": {},
        "active_agent": "delivery_management_agent",
    }
    monkeypatch.setattr(
        "agent.agents.delivery_management.agent.dispatch_provider_list",
        AsyncMock(return_value=fake_escalation),
    )
    state = make_dm_state(
        delivery_method="fax",
        awaiting_slot="fax_confirmed",
        messages=[_msg("assistant", f"Fax is {FAX_ON_FILE}?"), _msg("user", "yes")],
    )
    mock_extraction.return_value = answered_contact_yes()
    result = await _run(state)
    assert is_escalation(result)


# ---------------------------------------------------------------------------
# SECTION 6 — Response content (marker: response_check)
# ---------------------------------------------------------------------------


@pytest.mark.response_check
@pytest.mark.asyncio
async def test_fax_readback_contains_fax_number(mock_extraction) -> None:
    """After selecting fax, the readback must include the fax number on file."""
    mock_extraction.return_value = answered_method("fax")
    state = make_dm_state(messages=[_msg("user", "fax")])
    result = await _run(state)
    assert is_ask(result)
    assert FAX_ON_FILE in get_response(result)


@pytest.mark.response_check
@pytest.mark.asyncio
async def test_email_readback_contains_email(mock_extraction) -> None:
    """After selecting email, the readback must include the email on file."""
    mock_extraction.return_value = answered_method("email")
    state = make_dm_state(messages=[_msg("user", "email")])
    result = await _run(state)
    assert is_ask(result)
    assert EMAIL_ON_FILE in get_response(result)


@pytest.mark.response_check
@pytest.mark.asyncio
async def test_benefits_offer_message_contains_provider_type(mock_extraction, mock_dispatch) -> None:
    """Benefits offer message must contain the provider type."""
    state = make_dm_state(
        delivery_method="fax",
        awaiting_slot="fax_confirmed",
        messages=[_msg("assistant", f"Fax is {FAX_ON_FILE}?"), _msg("user", "yes")],
    )
    mock_extraction.return_value = answered_contact_yes()
    result = await _run(state)
    assert is_ask(result) and get_awaiting(result) == "benefits_response"
    # Provider type should appear in the combined delivery + benefits message
    provider_type_lower = PROVIDER.lower()
    assert any(word in get_response(result).lower() for word in provider_type_lower.split()), (
        f"Benefits offer must mention provider type. Got: {get_response(result)!r}"
    )


@pytest.mark.response_check
@pytest.mark.asyncio
async def test_completion_sets_all_context_fields(mock_extraction) -> None:
    """signal_complete must include all required context fields."""
    state = make_dm_state(
        delivery_method="fax",
        provider_list_sent=True,
        benefits_offer_made=True,
        awaiting_slot="benefits_response",
        delivery_timestamp="2026-05-26T00:00:00",
        messages=[_msg("assistant", "Benefits?"), _msg("user", "yes")],
    )
    mock_extraction.return_value = answered_benefits("yes")
    result = await _run(state)
    assert is_complete(result)
    assert result.get("provider_list_sent") is True
    assert result.get("delivery_method") == "fax"
    assert result.get("benefits_offer_made") is True
    assert result.get("proactive_offer_available") is True
    assert "delivery_timestamp" in result


# ---------------------------------------------------------------------------
# SECTION 7 — Unhappy: invalid fax retries (marker: unhappy)
# ---------------------------------------------------------------------------


@pytest.mark.unhappy
@pytest.mark.asyncio
async def test_unhappy_invalid_fax_retries(mock_extraction) -> None:
    """Member provides invalid fax digits (too short) → agent asks for a proper fax."""
    mock_extraction.return_value = WorkerResult(
        extracted={"fax": "41555"},  # only 5 digits — invalid
        event_type=EventType.ANSWERED,
        guard=GuardType.NONE,
        guard_confidence=0.0,
    )
    state = make_dm_state(
        delivery_method="fax",
        awaiting_slot="fax_confirmed",
        messages=[_msg("assistant", f"Fax is {FAX_ON_FILE}?"), _msg("user", "no, four one five five five")],
    )
    result = await _run(state)
    assert is_ask(result)
    assert get_awaiting(result) == "fax"
    assert not is_escalation(result)


# ---------------------------------------------------------------------------
# SECTION 8 — Guard: interruption re-asks (marker: guards)
# ---------------------------------------------------------------------------


@pytest.mark.guards
@pytest.mark.asyncio
async def test_guard_interruption_reasks(mock_extraction) -> None:
    """INTERRUPTION guard → re-ask current slot, do not escalate."""
    mock_extraction.return_value = make_guard(GuardType.INTERRUPTION)
    state = make_dm_state(messages=[_msg("user", "wait, can we do something else?")])
    result = await _run(state)
    assert is_ask(result)
    assert not is_escalation(result)


# ---------------------------------------------------------------------------
# SECTION 9 — Corrections: fax mid-collection (marker: corrections)
# ---------------------------------------------------------------------------


@pytest.mark.corrections
@pytest.mark.asyncio
async def test_correct_fax_mid_collection(mock_extraction, mock_dispatch, mock_fax_update) -> None:
    """Member provides wrong fax, then corrects to a valid 10-digit number."""
    # Turn 1: decline fax on file
    mock_extraction.return_value = answered_contact_no()
    state = make_dm_state(
        delivery_method="fax",
        awaiting_slot="fax_confirmed",
        messages=[_msg("assistant", f"Fax is {FAX_ON_FILE}?"), _msg("user", "no")],
    )
    result1 = await _run(state)
    assert is_ask(result1)
    assert get_awaiting(result1) == "fax"

    # Turn 2: provide correct 10-digit fax
    mock_extraction.return_value = answered_new_fax("4155553211")
    state2 = advance(state, result1, "four one five five five five three two one one")
    result2 = await _run(state2)
    assert is_ask(result2)
    assert get_awaiting(result2) == "benefits_response"
    assert result2.get("fax") == "4155553211"
    mock_fax_update.assert_called_once()


# ---------------------------------------------------------------------------
# SECTION 10 — Regression (marker: regression)
# ---------------------------------------------------------------------------


@pytest.mark.regression
@pytest.mark.asyncio
async def test_delivery_timestamp_set_on_dispatch(mock_extraction, mock_dispatch) -> None:
    """After dispatch, result must contain a non-empty delivery_timestamp string."""
    mock_extraction.return_value = answered_contact_yes()
    state = make_dm_state(
        delivery_method="fax",
        awaiting_slot="fax_confirmed",
        messages=[_msg("assistant", f"Fax is {FAX_ON_FILE}?"), _msg("user", "yes")],
    )
    result = await _run(state)
    assert is_ask(result)
    assert get_awaiting(result) == "benefits_response"
    ts = result.get("delivery_timestamp", "")
    assert isinstance(ts, str) and len(ts) > 0, f"delivery_timestamp must be a non-empty string, got {ts!r}"


@pytest.mark.regression
@pytest.mark.asyncio
async def test_fax_in_state_after_update(mock_extraction, mock_dispatch, mock_fax_update) -> None:
    """After fax update, new fax number must be persisted in the result."""
    mock_extraction.return_value = answered_new_fax("4155553211")
    state = make_dm_state(
        delivery_method="fax",
        awaiting_slot="fax",
        fax="",
        messages=[_msg("assistant", "Please provide the new fax number."), _msg("user", "4155553211")],
    )
    result = await _run(state)
    assert is_ask(result)
    assert get_awaiting(result) == "benefits_response"
    assert result.get("fax") == "4155553211", f"New fax must be stored in result, got {result.get('fax')!r}"
    mock_fax_update.assert_called_once()


# ---------------------------------------------------------------------------
# SECTION 11 — Email inline new address (marker: happy)
# ---------------------------------------------------------------------------


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_email_inline_new_address(mock_extraction, mock_dispatch, mock_email_update) -> None:
    """Member says new email inline with confirmation turn"""
    """→ SF update, dispatch, awaiting=benefits_response."""
    state = make_dm_state(
        delivery_method="email",
        awaiting_slot="email_confirmed",
        messages=[
            _msg("assistant", f"Email is {EMAIL_ON_FILE}?"),
            _msg("user", "no, it's new@example.com"),
        ],
    )
    mock_extraction.return_value = answered_new_email("new@example.com")
    result = await _run(state)
    assert is_ask(result), "Inline email update must produce ask (benefits_response)"
    assert get_awaiting(result) == "benefits_response", (
        f"Expected awaiting='benefits_response', got {get_awaiting(result)!r}"
    )
    assert result.get("email") == "new@example.com"
    mock_email_update.assert_called_once()
    mock_dispatch.assert_called_once()


# ---------------------------------------------------------------------------
# SECTION 12 — Benefits retry re-ask (marker: retry)
# ---------------------------------------------------------------------------


@pytest.mark.retry
@pytest.mark.asyncio
async def test_benefits_retry_once_then_reasks(mock_extraction) -> None:
    """First failure on benefits_response (attempt_count=0 → slot_fail → 1 < MAX=2) → re-ask."""
    # Note: attempt_count=0 means no prior failures; after slot_fail count becomes 1,
    # which is NOT exhausted (1 < MAX_SLOT_ATTEMPTS=2). The spec comment says "attempt_count=1"
    # but that would exhaust the slot — we use 0 here to get the re-ask path.
    mock_extraction.return_value = EMPTY_ANSWERED
    state = make_dm_state(
        delivery_method="fax",
        provider_list_sent=True,
        benefits_offer_made=True,
        awaiting_slot="benefits_response",
        messages=[
            _msg("assistant", "Would you like to also get the benefits?"),
            _msg("user", "hmm not sure"),
        ],
    )
    result = await _run(state)
    assert is_ask(result), "First benefits failure must re-ask, not exhaust"
    assert get_awaiting(result) == "benefits_response", (
        f"Expected awaiting='benefits_response', got {get_awaiting(result)!r}"
    )
    assert not is_escalation(result)


# ---------------------------------------------------------------------------
# SECTION 13 — Invalid email retries (marker: unhappy)
# ---------------------------------------------------------------------------


@pytest.mark.unhappy
@pytest.mark.asyncio
async def test_unhappy_invalid_email_retries(mock_extraction) -> None:
    """Member provides email missing '@' → agent asks for email update (awaiting=email)."""
    mock_extraction.return_value = WorkerResult(
        extracted={"email": "invalidemail"},  # missing '@' — fails validate_email
        event_type=EventType.ANSWERED,
        guard=GuardType.NONE,
        guard_confidence=0.0,
    )
    state = make_dm_state(
        delivery_method="email",
        awaiting_slot="email_confirmed",
        messages=[
            _msg("assistant", f"Email is {EMAIL_ON_FILE}?"),
            _msg("user", "no, invalidemail"),
        ],
    )
    result = await _run(state)
    assert is_ask(result), "Invalid email must produce a re-ask for proper address"
    assert get_awaiting(result) == "email", f"Expected awaiting='email', got {get_awaiting(result)!r}"
    assert not is_escalation(result)


# ---------------------------------------------------------------------------
# SECTION 14 — Fallback branch reasks confirmation (marker: happy)
# ---------------------------------------------------------------------------


@pytest.mark.happy
@pytest.mark.asyncio
async def test_fallback_branch_reasks_confirmation(mock_extraction) -> None:
    """delivery_method=fax + awaiting_slot='' → falls through to _ask_contact_confirmation → fax_confirmed."""
    # awaiting_slot="" → current_awaiting="delivery_method" (from `or "delivery_method"` fallback)
    # delivery_method is set → skips "collect delivery method" block
    # no fax_confirmed/fax/email_confirmed/email match → fallback branch triggers
    mock_extraction.return_value = EMPTY_ANSWERED
    state = make_dm_state(
        delivery_method="fax",
        fax=FAX_ON_FILE,
        awaiting_slot="",
        messages=[_msg("user", "hi")],
    )
    result = await _run(state)
    assert is_ask(result), "Fallback must produce a re-ask"
    assert get_awaiting(result) == "fax_confirmed", (
        f"Fallback with fax on file must ask for fax_confirmed, got {get_awaiting(result)!r}"
    )


# ---------------------------------------------------------------------------
# SECTION 15 — Delivery timestamp ISO format (marker: regression)
# ---------------------------------------------------------------------------


@pytest.mark.regression
@pytest.mark.asyncio
async def test_delivery_timestamp_is_iso_format(mock_extraction, mock_dispatch) -> None:
    """delivery_timestamp after dispatch must match ISO 8601 datetime pattern."""
    mock_extraction.return_value = answered_contact_yes()
    state = make_dm_state(
        delivery_method="fax",
        awaiting_slot="fax_confirmed",
        messages=[_msg("assistant", f"Fax is {FAX_ON_FILE}?"), _msg("user", "yes")],
    )
    result = await _run(state)
    assert is_ask(result) and get_awaiting(result) == "benefits_response"
    ts = result.get("delivery_timestamp", "")
    assert re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", str(ts)), (
        f"delivery_timestamp must be ISO 8601 format, got {ts!r}"
    )


# ---------------------------------------------------------------------------
# SECTION 16 — Email update SF failure (marker: dispatch_fail)
# ---------------------------------------------------------------------------


@pytest.mark.dispatch_fail
@pytest.mark.asyncio
async def test_email_update_sf_failure_escalates(mock_extraction, monkeypatch) -> None:
    """update_email_in_salesforce returns escalation dict → agent returns escalation."""
    fake_escalation = {
        "next_node": "escalation_agent",
        "is_interrupt": False,
        "messages": {"role": "assistant", "content": "Could not update email."},
        "slot_attempts": {},
        "metadata_events": [],
        "app_run_id": "",
        "awaiting_slot": "",
        "last_agent_signal": {},
        "active_agent": "delivery_management_agent",
    }
    monkeypatch.setattr(
        "agent.agents.delivery_management.agent.update_email_in_salesforce",
        AsyncMock(return_value=fake_escalation),
    )
    state = make_dm_state(
        delivery_method="email",
        awaiting_slot="email_confirmed",
        messages=[
            _msg("assistant", f"Email is {EMAIL_ON_FILE}?"),
            _msg("user", "no, it's new@example.com"),
        ],
    )
    mock_extraction.return_value = answered_new_email("new@example.com")
    result = await _run(state)
    assert is_escalation(result), f"SF email update failure must escalate, got {result.get('next_node')!r}"


# ---------------------------------------------------------------------------
# SECTION 17 — Concurrent stress: mock early-exit (marker: stress)
# ---------------------------------------------------------------------------


@pytest.mark.stress
@pytest.mark.asyncio
async def test_stress_10_concurrent_mock(mock_extraction) -> None:
    """10 concurrent early-exit runs (no LLM) — all must return is_complete=True."""

    async def _one_run() -> dict:
        state = make_dm_state(
            delivery_method="fax",
            provider_list_sent=True,
            benefits_offer_made=True,
            awaiting_slot="",
        )
        return await _run(state)

    results = await asyncio.gather(*[_one_run() for _ in range(10)], return_exceptions=True)
    failures = [r for r in results if isinstance(r, Exception) or not is_complete(r)]  # type: ignore[arg-type]
    assert len(failures) == 0, (
        f"All 10 concurrent early-exit runs must complete; {len(failures)} failed: {failures}"
    )
