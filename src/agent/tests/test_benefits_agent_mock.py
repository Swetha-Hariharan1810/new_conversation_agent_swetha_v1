"""
test_benefits_agent_mock.py — Mock test suite for BenefitsAgent.

No external credentials required. Covers happy paths, response content,
retry exhaustion, guards, SF fetch failure, and regression checks.

Run all:    pytest src/agent/tests/test_benefits_agent_mock.py -v
By marker:  pytest src/agent/tests/test_benefits_agent_mock.py -v -m happy
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.agents.benefits.agent import BenefitsAgent, _clean_amount
from agent.llm.schema import EventType, GuardType, WorkerResult
from agent.tests.fixtures import make_verified_state

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_BENEFITS = {
    "individual_deductible": "750",
    "family_deductible": "2500",
    "coinsurance_percent": "20",
    "individual_oop_max": "3000",
    "family_oop_max": "7000",
}

FAX_ON_FILE = "6175554101"
EMAIL_ON_FILE = "emily@example.com"

# ---------------------------------------------------------------------------
# State helper
# ---------------------------------------------------------------------------


def make_benefits_state(**overrides) -> dict:
    """Verified state ready for BenefitsAgent."""
    defaults: dict = {
        "call_intent": "benefits_inquiry",
        "member_id": "M907503",
        "benefits_explained": False,
        "care_coach_offered": False,
        "care_coach_offer_made": False,
        "individual_deductible": "",
        "family_deductible": "",
        "coinsurance_percent": "",
        "individual_oop_max": "",
        "family_oop_max": "",
        "proactive_offer_available": True,
        "delivery_method": "fax",
        "fax": FAX_ON_FILE,
        "email": EMAIL_ON_FILE,
    }
    defaults.update(overrides)
    return make_verified_state(**defaults)


def _msg(role: str, text: str) -> dict:
    return {"role": role, "content": text}


# ---------------------------------------------------------------------------
# WorkerResult factories
# ---------------------------------------------------------------------------

EMPTY_ANSWERED = WorkerResult(event_type=EventType.ANSWERED)


def answered_care_coach(response: str) -> WorkerResult:
    return WorkerResult(
        extracted={"care_coach_response": response},
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
    return await BenefitsAgent.from_state(state).execute(state)


def is_ask(result: dict) -> bool:
    return result.get("is_interrupt") is True and result.get("next_node") == "benefits_agent"


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
    monkeypatch.setattr("agent.agents.benefits.agent.get_extraction_llm", lambda: fake_llm)
    mock = AsyncMock(return_value=EMPTY_ANSWERED)
    monkeypatch.setattr("agent.agents.benefits.agent.extract_benefits_decision", mock)
    return mock


@pytest.fixture
def mock_benefits_fetch(monkeypatch) -> AsyncMock:
    mock = AsyncMock(return_value=(SAMPLE_BENEFITS, None))
    monkeypatch.setattr("agent.agents.benefits.agent.fetch_benefits", mock)
    return mock


@pytest.fixture
def mock_recovery(monkeypatch) -> AsyncMock:
    async def _fn(*, slot_name, attempt, guard, last_messages, **kwargs):
        return f"[RECOVERY:{slot_name}:attempt{attempt}:guard{guard}]"

    mock = AsyncMock(side_effect=_fn)
    monkeypatch.setattr("agent.llm.response_generator.generate_recovery_message", mock)
    return mock


@pytest.fixture(autouse=True)
def _base_mocks(mock_extraction, mock_benefits_fetch, mock_recovery):
    pass


# ---------------------------------------------------------------------------
# SECTION 1 — Happy path (marker: happy)
# ---------------------------------------------------------------------------


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_first_entry_explains_and_offers(mock_benefits_fetch) -> None:
    """First entry: fetch benefits, deliver explanation + care coach offer, await response."""
    state = make_benefits_state(messages=[_msg("user", "I'd like to know about my benefits")])
    result = await _run(state)
    assert is_ask(result)
    assert get_awaiting(result) == "care_coach_response"
    assert result.get("benefits_explained") is True
    mock_benefits_fetch.assert_called_once()


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_care_coach_yes_signals_complete_with_offer(mock_extraction) -> None:
    """Member accepts Care Coach → signal_complete with proactive_offer_available=True."""
    mock_extraction.return_value = answered_care_coach("yes")
    state = make_benefits_state(
        awaiting_slot="care_coach_response",
        benefits_explained=True,
        individual_deductible="750",
        family_deductible="2500",
        coinsurance_percent="20",
        individual_oop_max="3000",
        family_oop_max="7000",
        messages=[
            _msg("assistant", "Would you like Care Coach details?"),
            _msg("user", "yes please"),
        ],
    )
    result = await _run(state)
    assert is_complete(result)
    assert result.get("proactive_offer_available") is True


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_care_coach_no_signals_complete_no_offer(mock_extraction) -> None:
    """Member declines Care Coach → signal_complete with proactive_offer_available=False."""
    mock_extraction.return_value = answered_care_coach("no")
    state = make_benefits_state(
        awaiting_slot="care_coach_response",
        benefits_explained=True,
        individual_deductible="750",
        family_deductible="2500",
        coinsurance_percent="20",
        individual_oop_max="3000",
        family_oop_max="7000",
        messages=[
            _msg("assistant", "Would you like Care Coach details?"),
            _msg("user", "no thanks"),
        ],
    )
    result = await _run(state)
    assert is_complete(result)
    assert result.get("proactive_offer_available") is False


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_early_exit_already_explained_and_offered(mock_extraction) -> None:
    """Re-entry with both flags set → immediate complete, no LLM call."""
    state = make_benefits_state(
        benefits_explained=True,
        care_coach_offered=True,
        individual_deductible="750",
    )
    result = await _run(state)
    assert is_complete(result)
    mock_extraction.assert_not_called()


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_benefits_already_in_state_skips_sf(mock_benefits_fetch) -> None:
    """Benefits already in state — agent works correctly without an additional SF fetch."""
    state = make_benefits_state(
        individual_deductible="750",
        family_deductible="2500",
        coinsurance_percent="20",
        individual_oop_max="3000",
        family_oop_max="7000",
        messages=[_msg("user", "what are my benefits?")],
    )
    result = await _run(state)
    assert is_ask(result)
    assert get_awaiting(result) == "care_coach_response"
    assert result.get("benefits_explained") is True


# ---------------------------------------------------------------------------
# SECTION 2 — Response content (marker: response_check)
# ---------------------------------------------------------------------------


@pytest.mark.response_check
@pytest.mark.asyncio
async def test_explanation_contains_dollar_amounts(mock_benefits_fetch) -> None:
    """Explanation must include the individual and family deductible amounts."""
    state = make_benefits_state(messages=[_msg("user", "what are my benefits?")])
    result = await _run(state)
    response = get_response(result)
    assert "$750" in response, f"Expected $750 in response: {response!r}"
    assert "$2500" in response, f"Expected $2500 in response: {response!r}"


@pytest.mark.response_check
@pytest.mark.asyncio
async def test_explanation_contains_coinsurance_percent(mock_benefits_fetch) -> None:
    """Explanation must include the coinsurance percentage."""
    state = make_benefits_state(messages=[_msg("user", "what are my benefits?")])
    result = await _run(state)
    assert "20%" in get_response(result)


@pytest.mark.response_check
@pytest.mark.asyncio
async def test_explanation_contains_oop_max(mock_benefits_fetch) -> None:
    """Explanation must include individual and family OOP max values."""
    state = make_benefits_state(messages=[_msg("user", "what are my benefits?")])
    result = await _run(state)
    response = get_response(result)
    assert "$3000" in response, f"Expected $3000 in response: {response!r}"
    assert "$7000" in response, f"Expected $7000 in response: {response!r}"


@pytest.mark.response_check
@pytest.mark.asyncio
async def test_explanation_contains_100_percent_coverage(mock_benefits_fetch) -> None:
    """Explanation must mention 100% coverage once OOP max is reached."""
    state = make_benefits_state(messages=[_msg("user", "what are my benefits?")])
    result = await _run(state)
    assert "100%" in get_response(result)


# ---------------------------------------------------------------------------
# SECTION 3 — Retry exhaustion (marker: retry)
# ---------------------------------------------------------------------------


@pytest.mark.retry
@pytest.mark.asyncio
async def test_care_coach_retry_once_then_complete_gracefully(mock_extraction) -> None:
    """Two failed extractions (slot exhausted) → complete with proactive_offer_available=False."""
    mock_extraction.return_value = EMPTY_ANSWERED
    state = make_benefits_state(
        awaiting_slot="care_coach_response",
        benefits_explained=True,
        individual_deductible="750",
        family_deductible="2500",
        coinsurance_percent="20",
        individual_oop_max="3000",
        family_oop_max="7000",
        slot_attempts={"care_coach_response": {"attempt_count": 2, "confirmed": False, "last_value": None}},
        messages=[_msg("assistant", "Care Coach?"), _msg("user", "hmm")],
    )
    result = await _run(state)
    assert is_complete(result)
    assert result.get("proactive_offer_available") is False


# ---------------------------------------------------------------------------
# SECTION 4 — Guards (marker: guards)
# ---------------------------------------------------------------------------


@pytest.mark.guards
@pytest.mark.asyncio
async def test_guard_transfer_request_escalates(mock_extraction) -> None:
    mock_extraction.return_value = make_guard(GuardType.TRANSFER_REQUEST)
    state = make_benefits_state(
        awaiting_slot="care_coach_response",
        benefits_explained=True,
        individual_deductible="750",
        messages=[_msg("user", "transfer me please")],
    )
    assert is_escalation(await _run(state))


@pytest.mark.guards
@pytest.mark.asyncio
async def test_guard_abuse_escalates(mock_extraction) -> None:
    mock_extraction.return_value = make_guard(GuardType.ABUSE)
    state = make_benefits_state(
        awaiting_slot="care_coach_response",
        benefits_explained=True,
        individual_deductible="750",
        messages=[_msg("user", "this is ridiculous")],
    )
    assert is_escalation(await _run(state))


# ---------------------------------------------------------------------------
# SECTION 5 — SF fetch failure (marker: sf_fail)
# ---------------------------------------------------------------------------


@pytest.mark.sf_fail
@pytest.mark.asyncio
async def test_sf_fetch_fail_escalates(monkeypatch) -> None:
    """fetch_benefits returns (None, escalation_dict) → agent returns escalation."""
    escalation_dict = {
        "next_node": "escalation_agent",
        "is_interrupt": False,
        "messages": {"role": "assistant", "content": "I'm unable to retrieve your plan details."},
        "slot_attempts": {},
        "metadata_events": [],
        "app_run_id": "",
        "awaiting_slot": "",
        "last_agent_signal": {},
        "active_agent": "benefits_agent",
    }
    monkeypatch.setattr(
        "agent.agents.benefits.agent.fetch_benefits",
        AsyncMock(return_value=(None, escalation_dict)),
    )
    state = make_benefits_state(messages=[_msg("user", "what are my benefits?")])
    result = await _run(state)
    assert is_escalation(result)


# ---------------------------------------------------------------------------
# SECTION 6 — Regression (marker: regression)
# ---------------------------------------------------------------------------


@pytest.mark.regression
@pytest.mark.asyncio
async def test_completion_context_includes_all_benefit_fields(mock_extraction) -> None:
    """signal_complete after yes must carry all five benefit fields in the result."""
    mock_extraction.return_value = answered_care_coach("yes")
    state = make_benefits_state(
        awaiting_slot="care_coach_response",
        benefits_explained=True,
        individual_deductible="750",
        family_deductible="2500",
        coinsurance_percent="20",
        individual_oop_max="3000",
        family_oop_max="7000",
        messages=[
            _msg("assistant", "Would you like Care Coach details?"),
            _msg("user", "yes"),
        ],
    )
    result = await _run(state)
    assert is_complete(result)
    assert result.get("individual_deductible") == "750"
    assert result.get("family_deductible") == "2500"
    assert result.get("coinsurance_percent") == "20"
    assert result.get("individual_oop_max") == "3000"
    assert result.get("family_oop_max") == "7000"


# ---------------------------------------------------------------------------
# SECTION 7 — NO path (marker: happy)
# ---------------------------------------------------------------------------


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_no_path_skips_sf_fetch(mock_extraction, mock_benefits_fetch) -> None:
    """proactive_offer_available=False"""
    """→ skip SF fetch → NOEXPLANATION template, awaiting care_coach_response."""
    state = make_benefits_state(
        proactive_offer_available=False,
        messages=[_msg("user", "what are my benefits?")],
    )
    result = await _run(state)
    assert is_ask(result)
    assert get_awaiting(result) == "care_coach_response"
    assert result.get("benefits_explained") is False
    mock_benefits_fetch.assert_not_called()
    mock_extraction.assert_not_called()


@pytest.mark.happy
@pytest.mark.asyncio
async def test_early_exit_care_coach_offered_with_offer_true(mock_extraction) -> None:
    """care_coach_offered=True + proactive_offer_available=True → immediate complete, proactive=True."""
    state = make_benefits_state(care_coach_offered=True, proactive_offer_available=True)
    result = await _run(state)
    assert is_complete(result)
    assert result.get("proactive_offer_available") is True
    mock_extraction.assert_not_called()


@pytest.mark.happy
@pytest.mark.asyncio
async def test_early_exit_care_coach_offered_with_offer_false(mock_extraction) -> None:
    """care_coach_offered=True + proactive_offer_available=False → immediate complete, proactive=False."""
    state = make_benefits_state(care_coach_offered=True, proactive_offer_available=False)
    result = await _run(state)
    assert is_complete(result)
    assert result.get("proactive_offer_available") is False
    mock_extraction.assert_not_called()


# ---------------------------------------------------------------------------
# SECTION 8 — _clean_amount unit tests (pure; no agent needed)
# ---------------------------------------------------------------------------


@pytest.mark.happy
def test_clean_amount_decimal() -> None:
    assert _clean_amount("600.0") == "600"


@pytest.mark.happy
def test_clean_amount_double_decimal() -> None:
    assert _clean_amount("600.00") == "600"


@pytest.mark.happy
def test_clean_amount_none() -> None:
    assert _clean_amount(None) == "0"


@pytest.mark.happy
def test_clean_amount_already_clean() -> None:
    assert _clean_amount("600") == "600"


# ---------------------------------------------------------------------------
# SECTION 9 — Care coach retry not yet exhausted (marker: retry)
# ---------------------------------------------------------------------------


@pytest.mark.retry
@pytest.mark.asyncio
async def test_care_coach_retry_not_exhausted_reasks(mock_extraction) -> None:
    """attempt_count=0 (first failure) → slot not exhausted → re-ask, benefits_explained stays True."""
    mock_extraction.return_value = EMPTY_ANSWERED
    state = make_benefits_state(
        awaiting_slot="care_coach_response",
        benefits_explained=True,
        individual_deductible="750",
        family_deductible="2500",
        coinsurance_percent="20",
        individual_oop_max="3000",
        family_oop_max="7000",
        slot_attempts={"care_coach_response": {"attempt_count": 0, "confirmed": False, "last_value": None}},
        messages=[_msg("assistant", "Would you like Care Coach details?"), _msg("user", "hmm not sure")],
    )
    result = await _run(state)
    assert is_ask(result), "Not-yet-exhausted slot must produce a re-ask"
    assert get_awaiting(result) == "care_coach_response"
    assert result.get("benefits_explained") is True
    assert not is_escalation(result)


# ---------------------------------------------------------------------------
# SECTION 10 — Guards: OFFTOPIC_AGENT + SELF_HARM (marker: guards)
# ---------------------------------------------------------------------------


@pytest.mark.guards
@pytest.mark.asyncio
async def test_guard_offtopic_agent_reasks(mock_extraction) -> None:
    """OFFTOPIC_AGENT guard during care_coach_response phase → re-ask (not escalation)."""
    mock_extraction.return_value = make_guard(GuardType.OFFTOPIC_AGENT)
    state = make_benefits_state(
        awaiting_slot="care_coach_response",
        benefits_explained=True,
        individual_deductible="750",
        messages=[
            _msg("assistant", "Would you like Care Coach details?"),
            _msg("user", "tell me about my deductible again"),
        ],
    )
    result = await _run(state)
    assert is_ask(result), "OFFTOPIC_AGENT must produce a re-ask, not escalation"
    assert not is_escalation(result)


@pytest.mark.guards
@pytest.mark.asyncio
async def test_guard_self_harm_escalates(mock_extraction) -> None:
    """SELF_HARM guard during care_coach_response phase → immediate escalation."""
    mock_extraction.return_value = make_guard(GuardType.SELF_HARM)
    state = make_benefits_state(
        awaiting_slot="care_coach_response",
        benefits_explained=True,
        individual_deductible="750",
        messages=[
            _msg("assistant", "Would you like Care Coach details?"),
            _msg("user", "I don't want to be here anymore"),
        ],
    )
    assert is_escalation(await _run(state))


# ---------------------------------------------------------------------------
# SECTION 11 — NO-path completion context + regression (marker: regression)
# ---------------------------------------------------------------------------


@pytest.mark.regression
@pytest.mark.asyncio
async def test_completion_context_no_path_has_empty_benefit_fields(mock_extraction) -> None:
    """On NO path (no SF fetch), all five benefit fields in signal_complete context are empty strings."""
    mock_extraction.return_value = answered_care_coach("no")
    state = make_benefits_state(
        proactive_offer_available=False,
        awaiting_slot="care_coach_response",
        benefits_explained=False,
        individual_deductible="",
        family_deductible="",
        coinsurance_percent="",
        individual_oop_max="",
        family_oop_max="",
        messages=[
            _msg("assistant", "Would you like Care Coach details?"),
            _msg("user", "no thanks"),
        ],
    )
    result = await _run(state)
    assert is_complete(result)
    assert result.get("benefits_explained") is False
    assert result.get("proactive_offer_available") is False
    assert result.get("individual_deductible") == ""
    assert result.get("family_deductible") == ""
    assert result.get("coinsurance_percent") == ""
    assert result.get("individual_oop_max") == ""
    assert result.get("family_oop_max") == ""


@pytest.mark.regression
@pytest.mark.asyncio
async def test_regression_benefits_explained_false_on_no_path(mock_extraction, mock_benefits_fetch) -> None:
    """After NO-path first entry, result['benefits_explained'] must be False (no explanation was given)."""
    state = make_benefits_state(
        proactive_offer_available=False,
        messages=[_msg("user", "I want to know about my coverage")],
    )
    result = await _run(state)
    assert is_ask(result)
    assert result.get("benefits_explained") is False, (
        f"NO path must set benefits_explained=False; got {result.get('benefits_explained')!r}"
    )
    mock_benefits_fetch.assert_not_called()
    mock_extraction.assert_not_called()
