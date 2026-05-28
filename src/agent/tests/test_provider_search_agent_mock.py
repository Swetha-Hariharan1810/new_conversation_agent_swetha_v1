"""
test_provider_search_agent_mock.py — Mock test suite for ProviderSearchAgent.

No external credentials required. Covers happy paths, guard triggers,
ZIP confirmation flow, ZIP update flow, and retry exhaustion.

Run all:    pytest src/agent/tests/test_provider_search_agent_mock.py -v
By marker:  pytest src/agent/tests/test_provider_search_agent_mock.py -v -m happy
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.agents.provider_search.agent import ProviderSearchAgent
from agent.llm.schema import EventType, GuardType, WorkerResult
from agent.tests.fixtures import make_verified_state

# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

ZIP_ON_FILE = "12139"
PROVIDER = "Primary Care Physician"


def make_ps_state(**overrides) -> dict:
    """Verified state ready for provider search."""
    defaults: dict = {
        "call_intent": "provider_services",
        "provider_type": "",
        "zip_code_used": "",
        "zip_code": ZIP_ON_FILE,
    }
    defaults.update(overrides)
    # member_status_verify may be overridden (e.g. to False); extract it before calling
    # make_verified_state so it doesn't collide with the hardcoded True in that helper.
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


def answered_provider(pt: str = "Primary Care Physician") -> WorkerResult:
    return WorkerResult(
        extracted={"provider_type": pt},
        event_type=EventType.ANSWERED,
        guard=GuardType.NONE,
        guard_confidence=0.0,
    )


def answered_zip_yes() -> WorkerResult:
    return WorkerResult(
        extracted={"zip_confirmed": "yes"},
        event_type=EventType.ANSWERED,
        guard=GuardType.NONE,
        guard_confidence=0.0,
    )


def answered_zip_no() -> WorkerResult:
    return WorkerResult(
        extracted={"zip_confirmed": "no"},
        event_type=EventType.ANSWERED,
        guard=GuardType.NONE,
        guard_confidence=0.0,
    )


def answered_new_zip(zip_code: str = "90210") -> WorkerResult:
    return WorkerResult(
        extracted={"zip_code": zip_code},
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
    return await ProviderSearchAgent.from_state(state).execute(state)


def is_ask(result: dict) -> bool:
    return result.get("is_interrupt") is True and result.get("next_node") == "provider_search_agent"


def is_escalation(result: dict) -> bool:
    return result.get("next_node") == "escalation_agent" and result.get("is_interrupt") is False


def is_done(result: dict) -> bool:
    """
    Result signals provider search complete → delivery_management_agent.
    _signal_done now uses ask_member (is_interrupt=True) so the graph pauses
    for the user's fax/email answer before delivery_management_agent runs.
    """
    return result.get("next_node") == "delivery_management_agent" and result.get("is_interrupt") is True


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
    monkeypatch.setattr("agent.agents.provider_search.agent.get_extraction_llm", lambda: fake_llm)
    mock = AsyncMock(return_value=EMPTY_ANSWERED)
    monkeypatch.setattr("agent.agents.provider_search.agent.extract_provider_search_decision", mock)
    return mock


@pytest.fixture
def mock_zip_update(monkeypatch) -> AsyncMock:
    mock = AsyncMock(return_value=None)
    monkeypatch.setattr("agent.agents.provider_search.agent.update_zip_in_salesforce", mock)
    return mock


@pytest.fixture
def mock_recovery(monkeypatch) -> AsyncMock:
    async def _fn(*, slot_name, attempt, guard, last_messages, **kwargs):
        return f"[RECOVERY:{slot_name}:attempt{attempt}:guard{guard}]"

    mock = AsyncMock(side_effect=_fn)
    monkeypatch.setattr("agent.llm.response_generator.generate_recovery_message", mock)
    return mock


@pytest.fixture(autouse=True)
def _base_mocks(mock_extraction, mock_zip_update, mock_recovery):
    pass


# ---------------------------------------------------------------------------
# SECTION 1 — Happy path (marker: happy)
# ---------------------------------------------------------------------------


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_not_verified_escalates(mock_extraction) -> None:
    state = make_ps_state(member_status_verify=False)
    result = await _run(state)
    assert is_escalation(result)
    mock_extraction.assert_not_called()


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_first_turn_asks_provider_type(mock_extraction) -> None:
    state = make_ps_state(messages=[_msg("user", "hi")])
    result = await _run(state)
    assert is_ask(result), "First turn must ask for provider type"
    assert get_awaiting(result) == "provider_type"
    # First-entry fast path skips extraction entirely
    mock_extraction.assert_not_called()


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_provider_then_zip_confirm_yes(mock_extraction, mock_zip_update) -> None:
    """Full happy path: provider_type → zip confirm yes → done."""
    # Turn 1: ask provider type
    mock_extraction.return_value = EMPTY_ANSWERED
    state = make_ps_state(messages=[_msg("user", "I need a primary care doctor")])
    result1 = await _run(state)
    assert is_ask(result1)
    assert get_awaiting(result1) == "provider_type"

    # Turn 2: provide provider type
    mock_extraction.return_value = answered_provider()
    state2 = advance(state, result1, "primary care physician")
    result2 = await _run(state2)
    assert is_ask(result2), "After provider_type, should ask to confirm ZIP"
    assert get_awaiting(result2) == "zip_confirmed"
    assert ZIP_ON_FILE in get_response(result2)

    # Turn 3: confirm ZIP
    mock_extraction.return_value = answered_zip_yes()
    state3 = advance(state2, result2, "yes")
    result3 = await _run(state3)
    assert is_done(result3), "After zip confirmed=yes, should complete"
    assert result3.get("provider_type") == PROVIDER
    assert result3.get("zip_code_used") == ZIP_ON_FILE


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_zip_confirm_no_then_provide_new_zip(mock_extraction, mock_zip_update) -> None:
    """Happy path: zip confirm no → provide new zip → done."""
    state = make_ps_state(
        provider_type=PROVIDER,
        awaiting_slot="zip_confirmed",
        messages=[_msg("assistant", f"Your ZIP is {ZIP_ON_FILE}, correct?"), _msg("user", "no")],
    )
    # Turn 1: decline ZIP
    mock_extraction.return_value = answered_zip_no()
    result1 = await _run(state)
    assert is_ask(result1)
    assert get_awaiting(result1) == "zip_code"

    # Turn 2: provide new ZIP
    mock_extraction.return_value = answered_new_zip("90210")
    state2 = advance(state, result1, "nine oh two one oh")
    result2 = await _run(state2)
    assert is_done(result2)
    assert result2.get("zip_code_used") == "90210"
    mock_zip_update.assert_called_once()


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_zip_inline_with_no(mock_extraction, mock_zip_update) -> None:
    """Member says 'no, it's 90210' — inline ZIP provided with the no."""
    state = make_ps_state(
        provider_type=PROVIDER,
        awaiting_slot="zip_confirmed",
        messages=[_msg("assistant", f"ZIP is {ZIP_ON_FILE}?"), _msg("user", "no it's 90210")],
    )
    mock_extraction.return_value = WorkerResult(
        extracted={"zip_code": "90210"},
        event_type=EventType.ANSWERED,
        guard=GuardType.NONE,
        guard_confidence=0.0,
    )
    result = await _run(state)
    assert is_done(result)
    assert result.get("zip_code_used") == "90210"
    mock_zip_update.assert_called_once()


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_early_exit_both_slots_collected(mock_extraction) -> None:
    """Re-entry with both slots already set — emit done immediately, no LLM call."""
    state = make_ps_state(
        provider_type=PROVIDER,
        zip_code_used=ZIP_ON_FILE,
    )
    result = await _run(state)
    assert is_done(result)
    mock_extraction.assert_not_called()


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_no_zip_on_file_goes_to_collect(mock_extraction) -> None:
    """When zip_code is blank after provider_type, ask for new ZIP directly."""
    state = make_ps_state(
        provider_type=PROVIDER,
        zip_code="",
        awaiting_slot="",
        messages=[_msg("user", "primary care")],
    )
    mock_extraction.return_value = EMPTY_ANSWERED
    result = await _run(state)
    assert is_ask(result)
    assert get_awaiting(result) == "zip_code"


# ---------------------------------------------------------------------------
# SECTION 2 — Guard triggers (marker: guards)
# ---------------------------------------------------------------------------


@pytest.mark.guards
@pytest.mark.asyncio
async def test_guard_transfer_request(mock_extraction) -> None:
    mock_extraction.return_value = make_guard(GuardType.TRANSFER_REQUEST)
    state = make_ps_state(messages=[_msg("user", "I want to speak to someone")])
    assert is_escalation(await _run(state))


@pytest.mark.guards
@pytest.mark.asyncio
async def test_guard_abuse(mock_extraction) -> None:
    mock_extraction.return_value = make_guard(GuardType.ABUSE)
    state = make_ps_state(messages=[_msg("user", "you're terrible")])
    assert is_escalation(await _run(state))


@pytest.mark.guards
@pytest.mark.asyncio
async def test_guard_self_harm(mock_extraction) -> None:
    mock_extraction.return_value = make_guard(GuardType.SELF_HARM)
    state = make_ps_state(messages=[_msg("user", "I want to end it all")])
    assert is_escalation(await _run(state))


@pytest.mark.guards
@pytest.mark.asyncio
async def test_guard_interruption_asks_again(mock_extraction) -> None:
    mock_extraction.return_value = make_guard(GuardType.INTERRUPTION)
    state = make_ps_state(messages=[_msg("user", "can we do something else")])
    result = await _run(state)
    assert is_ask(result)


# ---------------------------------------------------------------------------
# SECTION 3 — ZIP confirmation retry exhaustion (marker: zip_retry)
# ---------------------------------------------------------------------------


@pytest.mark.zip_retry
@pytest.mark.asyncio
async def test_zip_confirmed_first_failure_retries(mock_extraction) -> None:
    mock_extraction.return_value = EMPTY_ANSWERED
    state = make_ps_state(
        provider_type=PROVIDER,
        awaiting_slot="zip_confirmed",
        messages=[_msg("assistant", f"ZIP is {ZIP_ON_FILE}?"), _msg("user", "uh")],
    )
    result = await _run(state)
    assert is_ask(result)
    assert get_awaiting(result) == "zip_confirmed"
    assert not is_escalation(result)


@pytest.mark.zip_retry
@pytest.mark.asyncio
async def test_zip_confirmed_second_failure_escalates(mock_extraction) -> None:
    mock_extraction.return_value = EMPTY_ANSWERED
    state = make_ps_state(
        provider_type=PROVIDER,
        awaiting_slot="zip_confirmed",
        slot_attempts={"zip_confirmed": {"attempt_count": 1, "confirmed": False, "last_value": None}},
        messages=[_msg("assistant", f"ZIP is {ZIP_ON_FILE}?"), _msg("user", "uh")],
    )
    result = await _run(state)
    assert is_escalation(result)


@pytest.mark.zip_retry
@pytest.mark.asyncio
async def test_provider_type_retry_exhaustion_escalates(mock_extraction) -> None:
    mock_extraction.return_value = EMPTY_ANSWERED
    state = make_ps_state(
        awaiting_slot="provider_type",
        slot_attempts={"provider_type": {"attempt_count": 2, "confirmed": False, "last_value": None}},
        messages=[_msg("assistant", "What type of provider?"), _msg("user", "???")],
    )
    result = await _run(state)
    assert is_escalation(result)


# ---------------------------------------------------------------------------
# SECTION 4 — ZIP update Salesforce failure (marker: sf_fail)
# ---------------------------------------------------------------------------


@pytest.mark.sf_fail
@pytest.mark.asyncio
async def test_zip_update_sf_failure_escalates(mock_extraction, monkeypatch) -> None:
    """If update_zip_in_salesforce returns an escalation dict, agent must return it."""
    fake_escalation = {
        "next_node": "escalation_agent",
        "is_interrupt": False,
        "messages": {"role": "assistant", "content": "Could not update ZIP"},
        "slot_attempts": {},
        "metadata_events": [],
        "app_run_id": "",
        "awaiting_slot": "",
        "last_agent_signal": {},
        "active_agent": "provider_search_agent",
    }
    monkeypatch.setattr(
        "agent.agents.provider_search.agent.update_zip_in_salesforce",
        AsyncMock(return_value=fake_escalation),
    )
    mock_extraction.return_value = answered_zip_no()
    state = make_ps_state(
        provider_type=PROVIDER,
        awaiting_slot="zip_confirmed",
        messages=[_msg("assistant", f"ZIP is {ZIP_ON_FILE}?"), _msg("user", "no")],
    )
    result1 = await _run(state)
    assert is_ask(result1) and get_awaiting(result1) == "zip_code"

    # Now provide new zip — SF update fails
    mock_extraction.return_value = answered_new_zip("90210")
    state2 = advance(state, result1, "90210")
    result2 = await _run(state2)
    assert is_escalation(result2)


# ---------------------------------------------------------------------------
# SECTION 5 — Response content (marker: response_check)
# ---------------------------------------------------------------------------


@pytest.mark.response_check
@pytest.mark.asyncio
async def test_zip_confirm_message_contains_zip(mock_extraction) -> None:
    """ZIP confirmation question must contain the ZIP on file."""
    mock_extraction.return_value = answered_provider()
    state = make_ps_state(
        awaiting_slot="provider_type",
        messages=[_msg("assistant", "What type of provider?"), _msg("user", "primary care")],
    )
    result = await _run(state)
    assert is_ask(result)
    assert ZIP_ON_FILE in get_response(result), "ZIP confirmation must mention the ZIP on file"


@pytest.mark.response_check
@pytest.mark.asyncio
async def test_done_message_not_empty(mock_extraction, mock_zip_update) -> None:
    """Completion bridge message must be non-empty."""
    state = make_ps_state(
        provider_type=PROVIDER,
        awaiting_slot="zip_confirmed",
        messages=[_msg("assistant", f"ZIP is {ZIP_ON_FILE}?"), _msg("user", "yes")],
    )
    mock_extraction.return_value = answered_zip_yes()
    result = await _run(state)
    assert is_done(result)
    assert len(get_response(result)) > 0, "Completion bridge message must not be empty"


@pytest.mark.response_check
@pytest.mark.asyncio
async def test_done_result_has_all_context_fields(mock_extraction) -> None:
    state = make_ps_state(
        provider_type=PROVIDER,
        awaiting_slot="zip_confirmed",
        messages=[_msg("assistant", f"ZIP is {ZIP_ON_FILE}?"), _msg("user", "yes")],
    )
    mock_extraction.return_value = answered_zip_yes()
    result = await _run(state)
    assert is_done(result)
    assert result.get("provider_type") == PROVIDER
    assert result.get("zip_code") == ZIP_ON_FILE
    assert result.get("zip_code_used") == ZIP_ON_FILE
    assert result.get("awaiting_slot") == ""


# ---------------------------------------------------------------------------
# SECTION 6 — Spoken provider type normalization (marker: happy)
# ---------------------------------------------------------------------------


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_spoken_provider_type(mock_extraction) -> None:
    """'primary care physician' normalises to 'Primary Care Physician'."""
    mock_extraction.return_value = WorkerResult(
        extracted={"provider_type": "primary care physician"},
        event_type=EventType.ANSWERED,
        guard=GuardType.NONE,
        guard_confidence=0.0,
    )
    state = make_ps_state(
        awaiting_slot="provider_type",
        messages=[_msg("assistant", "What type of provider?"), _msg("user", "primary care physician")],
    )
    result = await _run(state)
    assert result.get("provider_type") == "Primary Care Physician"


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_already_has_provider_type(mock_extraction) -> None:
    """provider_type already in state → skips collection, goes to ZIP confirmation."""
    mock_extraction.return_value = EMPTY_ANSWERED
    state = make_ps_state(
        provider_type="Primary Care Physician",
        zip_code_used="",
        awaiting_slot="",
        messages=[_msg("user", "hi")],
    )
    result = await _run(state)
    assert is_ask(result), "Should ask for ZIP confirmation"
    assert get_awaiting(result) == "zip_confirmed"


# ---------------------------------------------------------------------------
# SECTION 7 — Unhappy paths (marker: unhappy)
# ---------------------------------------------------------------------------


@pytest.mark.unhappy
@pytest.mark.asyncio
async def test_unhappy_refuses_provider_type(mock_extraction) -> None:
    """Member gives invalid/empty input for provider type → retry ask."""
    mock_extraction.return_value = EMPTY_ANSWERED
    state = make_ps_state(
        awaiting_slot="provider_type",
        messages=[_msg("assistant", "What type of provider?"), _msg("user", "I don't know")],
    )
    result = await _run(state)
    assert is_ask(result)
    assert get_awaiting(result) == "provider_type"
    assert not is_escalation(result)


@pytest.mark.unhappy
@pytest.mark.asyncio
async def test_unhappy_refuses_zip(mock_extraction) -> None:
    """Member gives no yes/no to ZIP confirm → retry ask."""
    mock_extraction.return_value = EMPTY_ANSWERED
    state = make_ps_state(
        provider_type="Primary Care Physician",
        awaiting_slot="zip_confirmed",
        messages=[_msg("assistant", "Is your ZIP 12139?"), _msg("user", "uh what?")],
    )
    result = await _run(state)
    assert is_ask(result)
    assert get_awaiting(result) == "zip_confirmed"
    assert not is_escalation(result)


@pytest.mark.unhappy
@pytest.mark.asyncio
async def test_unhappy_max_retries_provider_type_escalates(mock_extraction) -> None:
    """slot_attempts.provider_type.attempt_count=1 → one more failure → escalate."""
    mock_extraction.return_value = EMPTY_ANSWERED
    state = make_ps_state(
        awaiting_slot="provider_type",
        slot_attempts={"provider_type": {"attempt_count": 1, "confirmed": False, "last_value": None}},
        messages=[_msg("assistant", "What type of provider?"), _msg("user", "???")],
    )
    result = await _run(state)
    assert is_escalation(result)


# ---------------------------------------------------------------------------
# SECTION 8 — Guard: offtopic redirects (marker: guards)
# ---------------------------------------------------------------------------


@pytest.mark.guards
@pytest.mark.asyncio
async def test_guard_offtopic_redirects(mock_extraction) -> None:
    """OFFTOPIC_AGENT guard re-asks the current slot, does not escalate."""
    mock_extraction.return_value = make_guard(GuardType.OFFTOPIC_AGENT)
    state = make_ps_state(messages=[_msg("user", "tell me about my benefits")])
    result = await _run(state)
    assert is_ask(result)
    assert not is_escalation(result)


# ---------------------------------------------------------------------------
# SECTION 9 — Corrections (marker: corrections)
# ---------------------------------------------------------------------------


@pytest.mark.corrections
@pytest.mark.asyncio
async def test_correct_provider_type(mock_extraction) -> None:
    """Member says different provider type before ZIP confirmed — provider_type updated."""
    mock_extraction.return_value = WorkerResult(
        extracted={"provider_type": "pediatrician"},
        event_type=EventType.ANSWERED,
        guard=GuardType.NONE,
        guard_confidence=0.0,
    )
    state = make_ps_state(
        provider_type="",
        awaiting_slot="provider_type",
        messages=[
            _msg("assistant", "What type of provider?"),
            _msg("user", "actually I need a pediatrician not a PCP"),
        ],
    )
    result = await _run(state)
    assert not is_escalation(result)
    assert result.get("provider_type") == "Pediatrician"


# ---------------------------------------------------------------------------
# SECTION 10 — Regression: routing (marker: regression)
# ---------------------------------------------------------------------------


@pytest.mark.regression
@pytest.mark.asyncio
async def test_routing_after_zip_confirm_is_delivery_management(mock_extraction) -> None:
    mock_extraction.return_value = answered_zip_yes()
    state = make_ps_state(
        provider_type="Primary Care Physician",
        awaiting_slot="zip_confirmed",
        messages=[_msg("assistant", "ZIP is 12139?"), _msg("user", "yes")],
    )
    result = await _run(state)
    assert is_done(result)
    assert result.get("next_node") == "delivery_management_agent"
    assert result.get("is_interrupt") is True  # pauses for fax/email answer
    assert result.get("awaiting_slot") == ""  # delivery_management starts fresh
    assert result.get("next_node") != "orchestrator"


# ---------------------------------------------------------------------------
# SECTION 11 — First-entry fast path: provider_type set, no zip (marker: happy)
# ---------------------------------------------------------------------------


@pytest.mark.happy
@pytest.mark.asyncio
async def test_first_entry_no_zip_pre_populated_provider(mock_extraction) -> None:
    """raw_awaiting="" + provider_type pre-set + zip_code="" → fast path asks for zip_code, no LLM."""
    state = make_ps_state(
        provider_type=PROVIDER,
        zip_code="",  # no ZIP on file
        awaiting_slot="",  # fresh first entry
        messages=[_msg("user", "hi")],
    )
    result = await _run(state)
    assert is_ask(result), "Must ask for zip_code directly"
    assert get_awaiting(result) == "zip_code", f"Expected awaiting='zip_code', got {get_awaiting(result)!r}"
    assert result.get("provider_type") == PROVIDER
    mock_extraction.assert_not_called()


# ---------------------------------------------------------------------------
# SECTION 12 — Spoken ZIP normalization (marker: happy)
# ---------------------------------------------------------------------------


@pytest.mark.happy
@pytest.mark.asyncio
async def test_spoken_zip_normalized(mock_extraction, mock_zip_update) -> None:
    """Extraction returns spoken ZIP 'nine oh two one oh' → agent normalizes to '90210' → done."""
    # zip_confirmed branch: inline new zip extracted in spoken form
    state = make_ps_state(
        provider_type=PROVIDER,
        awaiting_slot="zip_confirmed",
        messages=[
            _msg("assistant", f"Your ZIP is {ZIP_ON_FILE}, correct?"),
            _msg("user", "no, nine oh two one oh"),
        ],
    )
    mock_extraction.return_value = WorkerResult(
        extracted={"zip_code": "nine oh two one oh"},
        event_type=EventType.ANSWERED,
        guard=GuardType.NONE,
        guard_confidence=0.0,
    )
    result = await _run(state)
    assert is_done(result), "Normalized spoken ZIP must complete provider search"
    assert result.get("zip_code_used") == "90210", (
        f"Spoken ZIP must normalize to '90210', got {result.get('zip_code_used')!r}"
    )
    mock_zip_update.assert_called_once()


# ---------------------------------------------------------------------------
# SECTION 13 — Invalid ZIP after normalization (marker: unhappy)
# ---------------------------------------------------------------------------


@pytest.mark.unhappy
@pytest.mark.asyncio
async def test_zip_invalid_after_normalization_asks_update(mock_extraction) -> None:
    """Extracted zip '123' (only 3 digits, fails validate_zip_code) → asks for zip_code, not escalation."""
    state = make_ps_state(
        provider_type=PROVIDER,
        awaiting_slot="zip_confirmed",
        messages=[
            _msg("assistant", f"Your ZIP is {ZIP_ON_FILE}, correct?"),
            _msg("user", "no, 123"),
        ],
    )
    mock_extraction.return_value = WorkerResult(
        extracted={"zip_code": "123"},
        event_type=EventType.ANSWERED,
        guard=GuardType.NONE,
        guard_confidence=0.0,
    )
    result = await _run(state)
    assert is_ask(result), "Invalid ZIP must produce a re-ask, not escalation"
    assert get_awaiting(result) == "zip_code", (
        f"Invalid ZIP must route to 'zip_code' slot, got {get_awaiting(result)!r}"
    )
    assert not is_escalation(result)


# ---------------------------------------------------------------------------
# SECTION 14 — Correction: provider_type updated while awaiting zip_confirmed (marker: corrections)
# ---------------------------------------------------------------------------


@pytest.mark.corrections
@pytest.mark.asyncio
async def test_correct_provider_type_while_zip_pending(mock_extraction) -> None:
    """User provides provider_type while awaiting zip_confirmed → type stored, zip still pending.

    Note: provider_type must be empty in state so the pipeline collects the new value.
    If provider_type is already non-empty (e.g. 'PCP'), validate_provider_type accepts it
    and the pipeline skips the slot — no correction is applied.
    """
    state = make_ps_state(
        provider_type="",  # not yet set — pipeline will accept new value
        awaiting_slot="zip_confirmed",
        messages=[
            _msg("assistant", f"Your ZIP is {ZIP_ON_FILE}, correct?"),
            _msg("user", "actually a cardiologist"),
        ],
    )
    mock_extraction.return_value = WorkerResult(
        extracted={"provider_type": "Cardiologist"},
        event_type=EventType.ANSWERED,
        guard=GuardType.NONE,
        guard_confidence=0.0,
    )
    result = await _run(state)
    assert not is_escalation(result), "Provider type update must not escalate"
    assert result.get("provider_type") == "Cardiologist", (
        f"Expected provider_type='Cardiologist', got {result.get('provider_type')!r}"
    )
    assert get_awaiting(result) == "zip_confirmed", (
        f"Must still await zip_confirmed after providing provider_type, got {get_awaiting(result)!r}"
    )


# ---------------------------------------------------------------------------
# SECTION 15 — Bonus extraction: provider_type + zip in one turn (marker: happy)
# ---------------------------------------------------------------------------


@pytest.mark.happy
@pytest.mark.asyncio
async def test_bonus_zip_extracted_with_provider_type(mock_extraction) -> None:
    """LLM extracts both provider_type and zip_code in one turn → provider_type stored, asks zip_confirmed."""
    state = make_ps_state(
        provider_type="",
        awaiting_slot="provider_type",
        zip_code="90210",  # set on file so zip_confirmed question uses it
        messages=[_msg("assistant", "What type of provider?"), _msg("user", "cardiologist")],
    )
    mock_extraction.return_value = WorkerResult(
        extracted={"provider_type": "Cardiologist", "zip_code": "90210"},
        event_type=EventType.ANSWERED,
        guard=GuardType.NONE,
        guard_confidence=0.0,
    )
    result = await _run(state)
    assert is_ask(result), "After bonus extraction, must ask for zip_confirmed"
    assert get_awaiting(result) == "zip_confirmed", (
        f"Expected awaiting='zip_confirmed', got {get_awaiting(result)!r}"
    )
    assert result.get("provider_type") == "Cardiologist"
    assert "90210" in get_response(result), "Confirmation must mention zip from state"


# ---------------------------------------------------------------------------
# SECTION 16 — _signal_done result contract (marker: response_check)
# ---------------------------------------------------------------------------


@pytest.mark.response_check
@pytest.mark.asyncio
async def test_signal_done_awaiting_slot_is_empty(mock_extraction) -> None:
    """After _signal_done: awaiting_slot='' so delivery_management_agent starts fresh."""
    mock_extraction.return_value = answered_zip_yes()
    state = make_ps_state(
        provider_type=PROVIDER,
        awaiting_slot="zip_confirmed",
        messages=[_msg("assistant", f"ZIP is {ZIP_ON_FILE}?"), _msg("user", "yes")],
    )
    result = await _run(state)
    assert is_done(result)
    assert result.get("awaiting_slot") == "", (
        f"_signal_done must clear awaiting_slot, got {result.get('awaiting_slot')!r}"
    )


@pytest.mark.response_check
@pytest.mark.asyncio
async def test_signal_done_is_interrupt_true(mock_extraction) -> None:
    """_signal_done uses ask_member → is_interrupt=True so graph pauses for fax/email answer."""
    mock_extraction.return_value = answered_zip_yes()
    state = make_ps_state(
        provider_type=PROVIDER,
        awaiting_slot="zip_confirmed",
        messages=[_msg("assistant", f"ZIP is {ZIP_ON_FILE}?"), _msg("user", "yes")],
    )
    result = await _run(state)
    assert is_done(result)
    assert result.get("is_interrupt") is True, (
        "_signal_done must set is_interrupt=True (graph pauses for member's fax/email answer)"
    )


# ---------------------------------------------------------------------------
# SECTION 17 — OFFTOPIC_GLOBAL guard (marker: guards)
# ---------------------------------------------------------------------------


@pytest.mark.guards
@pytest.mark.asyncio
async def test_guard_offtopic_global_reasks(mock_extraction) -> None:
    """OFFTOPIC_GLOBAL guard → ask_member with static response, offtopic_global_count incremented."""
    mock_extraction.return_value = make_guard(GuardType.OFFTOPIC_GLOBAL)
    state = make_ps_state(
        offtopic_global_count=0,
        messages=[_msg("user", "what's the weather like?")],
    )
    result = await _run(state)
    assert is_ask(result), "OFFTOPIC_GLOBAL must re-ask, not escalate (first occurrence)"
    assert not is_escalation(result)
    assert result.get("offtopic_global_count", 0) >= 1, (
        "offtopic_global_count must be incremented on OFFTOPIC_GLOBAL guard"
    )
