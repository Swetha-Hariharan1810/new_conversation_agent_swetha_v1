"""
test_follow_up_agent_mock.py — Mock tests for FollowUpAgent.

No external credentials required. The single LLM call (generate_follow_up_answer)
is patched with a deterministic function that reads from state.

Run all:   pytest src/agent/tests/test_follow_up_agent_mock.py -v
By marker: pytest src/agent/tests/test_follow_up_agent_mock.py -v -m happy
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from agent.agents.follow_up.agent import FollowUpAgent
from agent.agents.follow_up.constants import BARE_AFFIRMATIONS, CLOSURE_KEYWORDS
from agent.agents.follow_up.llm import _build_session_snapshot
from agent.tests.fixtures import make_verified_state

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FAX_ON_FILE = "6175554101"


# ---------------------------------------------------------------------------
# State factory
# ---------------------------------------------------------------------------


def make_fu_state(**overrides) -> dict:
    """Fully loaded post-call state ready for FollowUpAgent."""
    defaults: dict = {
        "call_intent": "provider_services",
        "member_status_verify": True,
        "provider_type": "Primary Care Physician",
        "zip_code_used": "12139",
        "zip_code": "12139",
        "provider_list_sent": True,
        "benefits_explained": True,
        "care_coach_offered": True,
        "care_coach_details_sent": True,
        "delivery_method": "fax",
        "fax": FAX_ON_FILE,
        "individual_deductible": "750",
        "family_deductible": "2500",
        "coinsurance_percent": "20",
        "individual_oop_max": "3000",
        "family_oop_max": "7000",
        "follow_up_turn_count": 0,
        "follow_up_last_question": "",
    }
    defaults.update(overrides)
    # make_verified_state hard-codes member_status_verify=True; pop it first
    # so it is not passed twice (same pattern as make_delivery_ready_state).
    member_status_verify = defaults.pop("member_status_verify", True)
    state = make_verified_state(**defaults)
    state["member_status_verify"] = member_status_verify
    return state


def _msg(role: str, text: str) -> dict:
    return {"role": role, "content": text}


# ---------------------------------------------------------------------------
# Runner + assertion helpers
# ---------------------------------------------------------------------------


async def _run(state: dict) -> dict:
    return await FollowUpAgent.from_state(state).execute(state)


def is_ask(result: dict) -> bool:
    return result.get("is_interrupt") is True and result.get("next_node") == "follow_up_agent"


def is_closure(result: dict) -> bool:
    """Agent signalled closure — orchestrator will route to closure_agent."""
    signal = result.get("last_agent_signal") or {}
    return (
        result.get("is_interrupt") is False
        and result.get("next_node") == "orchestrator"
        and signal.get("closure_requested") is True
    )


def get_response(result: dict) -> str:
    msg = result.get("messages", {})
    if isinstance(msg, dict):
        return msg.get("content", "")
    if isinstance(msg, list) and msg:
        last = msg[-1]
        return last.get("content", "") if isinstance(last, dict) else str(last)
    return ""


def advance(state: dict, result: dict, user_text: str) -> dict:
    new_state = {**state}
    for key, val in result.items():
        if key == "messages":
            continue
        new_state[key] = val
    messages = list(state.get("messages") or [])
    if isinstance(result.get("messages"), dict):
        messages.append(result["messages"])
    messages.append({"role": "user", "content": user_text})
    new_state["messages"] = messages
    return new_state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm(monkeypatch) -> AsyncMock:
    """
    Patch generate_follow_up_answer to return a deterministic answer
    derived from state — no real LLM call.
    """

    async def _fn(state, question):
        indv = state.get("individual_deductible", "")
        fam = state.get("family_deductible", "")
        coins = state.get("coinsurance_percent", "")
        indv_oop = state.get("individual_oop_max", "")
        fam_oop = state.get("family_oop_max", "")
        if indv:
            return (
                f"Your individual deductible is ${indv} per year, and the family "
                f"deductible is ${fam}. After meeting the deductible, you pay "
                f"{coins}% coinsurance. Your individual out-of-pocket max is "
                f"${indv_oop} and the family max is ${fam_oop}. After that, your "
                f"plan covers 100% of in-network services."
            )
        return ""  # signals cannot-answer path

    mock = AsyncMock(side_effect=_fn)
    monkeypatch.setattr("agent.agents.follow_up.agent.generate_follow_up_answer", mock)
    return mock


@pytest.fixture(autouse=True)
def _base_mock(mock_llm):
    pass


# ---------------------------------------------------------------------------
# SECTION 1 — Closure detection (marker: happy)
# ---------------------------------------------------------------------------


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_no_thanks_signals_closure(mock_llm) -> None:
    state = make_fu_state(
        messages=[
            _msg("assistant", "Would you like help with anything else?"),
            _msg("user", "No thanks that was helpful"),
        ]
    )
    result = await _run(state)
    assert is_closure(result), f"Expected closure signal, got: {result.get('next_node')}"
    mock_llm.assert_not_called()


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_no_signals_closure(mock_llm) -> None:
    state = make_fu_state(
        messages=[_msg("assistant", "Would you like help with anything else?"), _msg("user", "No")]
    )
    result = await _run(state)
    assert is_closure(result)
    mock_llm.assert_not_called()


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_bye_signals_closure(mock_llm) -> None:
    state = make_fu_state(
        messages=[_msg("assistant", "Would you like help with anything else?"), _msg("user", "bye")]
    )
    result = await _run(state)
    assert is_closure(result)
    mock_llm.assert_not_called()


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_that_was_helpful_signals_closure(mock_llm) -> None:
    state = make_fu_state(
        messages=[
            _msg("assistant", "Would you like help with anything else?"),
            _msg("user", "that was helpful"),
        ]
    )
    result = await _run(state)
    assert is_closure(result)


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_thank_you_signals_closure(mock_llm) -> None:
    state = make_fu_state(
        messages=[_msg("assistant", "Would you like help with anything else?"), _msg("user", "thank you")]
    )
    result = await _run(state)
    assert is_closure(result)


# ---------------------------------------------------------------------------
# SECTION 2 — Question answered from session context (marker: happy)
# ---------------------------------------------------------------------------


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_question_triggers_llm(mock_llm) -> None:
    state = make_fu_state(
        messages=[
            _msg("assistant", "Would you like help with anything else?"),
            _msg("user", "Can you summarize the PCP benefits?"),
        ]
    )
    result = await _run(state)
    assert is_ask(result), f"Expected ask, got next_node={result.get('next_node')!r}"
    mock_llm.assert_called_once()


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_follow_up_turn_count_incremented(mock_llm) -> None:
    state = make_fu_state(
        follow_up_turn_count=1,
        messages=[_msg("assistant", "Anything else?"), _msg("user", "what is my deductible")],
    )
    result = await _run(state)
    assert result.get("follow_up_turn_count") == 2


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_last_question_stored(mock_llm) -> None:
    question = "Can you summarize the PCP benefits?"
    state = make_fu_state(messages=[_msg("assistant", "Anything else?"), _msg("user", question)])
    result = await _run(state)
    assert result.get("follow_up_last_question") == question


# ---------------------------------------------------------------------------
# SECTION 3 — Response content (marker: response_check)
# ---------------------------------------------------------------------------


@pytest.mark.response_check
@pytest.mark.asyncio
async def test_response_contains_individual_deductible(mock_llm) -> None:
    state = make_fu_state(
        messages=[_msg("assistant", "Anything else?"), _msg("user", "what is my deductible")]
    )
    result = await _run(state)
    assert "$750" in get_response(result), get_response(result)


@pytest.mark.response_check
@pytest.mark.asyncio
async def test_response_contains_family_deductible(mock_llm) -> None:
    state = make_fu_state(
        messages=[_msg("assistant", "Anything else?"), _msg("user", "summarize my benefits")]
    )
    result = await _run(state)
    assert "$2500" in get_response(result), get_response(result)


@pytest.mark.response_check
@pytest.mark.asyncio
async def test_response_contains_continuation_question(mock_llm) -> None:
    state = make_fu_state(
        messages=[_msg("assistant", "Anything else?"), _msg("user", "what is my deductible")]
    )
    result = await _run(state)
    response = get_response(result)
    continuation_phrases = [
        "anything else",
        "anything i can",
        "other questions",
        "i can assist",
    ]
    assert any(p in response.lower() for p in continuation_phrases), (
        f"Continuation question missing from: {response!r}"
    )


@pytest.mark.response_check
@pytest.mark.asyncio
async def test_response_closure_has_no_message(mock_llm) -> None:
    """Closure signal must produce an empty message — closure_agent speaks."""
    state = make_fu_state(messages=[_msg("assistant", "Anything else?"), _msg("user", "no")])
    result = await _run(state)
    assert is_closure(result)
    # message may be absent or empty string
    msg = result.get("messages")
    if msg:
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        assert content.strip() == "", f"Closure must not produce a message: {content!r}"


# ---------------------------------------------------------------------------
# SECTION 4 — Cannot-answer path (marker: unhappy)
# ---------------------------------------------------------------------------


@pytest.mark.unhappy
@pytest.mark.asyncio
async def test_cannot_answer_still_asks_continuation(monkeypatch) -> None:
    """When LLM returns '' (cannot answer), agent apologises + asks continuation."""

    async def _empty(state, question):
        return ""

    monkeypatch.setattr(
        "agent.agents.follow_up.agent.generate_follow_up_answer",
        AsyncMock(side_effect=_empty),
    )
    state = make_fu_state(
        individual_deductible="",  # no benefits in state
        messages=[_msg("assistant", "Anything else?"), _msg("user", "what are my vision benefits")],
    )
    result = await _run(state)
    assert is_ask(result), "Cannot-answer must still re-ask, not close"
    response = get_response(result)
    continuation_phrases = [
        "anything else",
        "anything i can",
        "other questions",
        "i can assist",
    ]
    assert any(p in response.lower() for p in continuation_phrases), (
        f"Continuation question missing even on cannot-answer: {response!r}"
    )


# ---------------------------------------------------------------------------
# SECTION 5 — Max turns exhaustion (marker: retry)
# ---------------------------------------------------------------------------


@pytest.mark.retry
@pytest.mark.asyncio
async def test_max_turns_routes_to_closure(mock_llm) -> None:
    """After MAX_FOLLOW_UP_TURNS, agent signals closure regardless of question."""
    from agent.agents.follow_up.constants import MAX_FOLLOW_UP_TURNS

    state = make_fu_state(
        follow_up_turn_count=MAX_FOLLOW_UP_TURNS,  # already at max
        messages=[_msg("assistant", "Anything else?"), _msg("user", "tell me more")],
    )
    result = await _run(state)
    assert is_closure(result), "Max-turn exhaustion must signal closure"
    mock_llm.assert_not_called()


# ---------------------------------------------------------------------------
# SECTION 6 — Multi-turn follow-up flow (marker: happy)
# ---------------------------------------------------------------------------


@pytest.mark.happy
@pytest.mark.asyncio
async def test_multi_turn_question_then_close(mock_llm) -> None:
    """Full two-turn flow: question answered → member says no → closure."""
    state = make_fu_state(
        messages=[
            _msg("assistant", "Is there anything else?"),
            _msg("user", "Can you summarize the PCP benefits?"),
        ]
    )

    # Turn 1: question
    result1 = await _run(state)
    assert is_ask(result1)
    assert "$750" in get_response(result1)

    # Turn 2: closure
    state2 = advance(state, result1, "No thanks that was helpful")
    result2 = await _run(state2)
    assert is_closure(result2)


# ---------------------------------------------------------------------------
# SECTION 7 — Regression (marker: regression)
# ---------------------------------------------------------------------------


@pytest.mark.regression
@pytest.mark.asyncio
async def test_regression_empty_message_does_not_close(mock_llm) -> None:
    """An empty/whitespace utterance must not trigger closure."""
    state = make_fu_state(messages=[_msg("assistant", "Anything else?"), _msg("user", "   ")])
    result = await _run(state)
    # Empty utterance → LLM returns '' → cannot-answer path → ask again
    assert is_ask(result), "Empty utterance should not trigger closure"


@pytest.mark.regression
@pytest.mark.asyncio
async def test_regression_turn_count_not_incremented_on_closure(mock_llm) -> None:
    """Closure must not increment turn count (no further turns needed)."""
    state = make_fu_state(
        follow_up_turn_count=2, messages=[_msg("assistant", "Anything else?"), _msg("user", "no")]
    )
    result = await _run(state)
    assert is_closure(result)
    # turn count in context_updates should remain 2 (not 3)
    ctx = (result.get("last_agent_signal") or {}).get("context_updates") or {}
    stored_count = ctx.get("follow_up_turn_count", 2)
    assert stored_count == 2, f"Closure must not increment turn count, got {stored_count}"


# ---------------------------------------------------------------------------
# SECTION 8 — Bare affirmation nudge (Bug 1) (marker: happy / regression)
# ---------------------------------------------------------------------------


@pytest.mark.happy
@pytest.mark.asyncio
async def test_bare_affirmation_yes_returns_nudge(mock_llm) -> None:
    """'yes' must return a nudge question, never call the LLM."""
    state = make_fu_state(
        messages=[_msg("assistant", "Is there anything else I can help you with?"), _msg("user", "yes")]
    )
    result = await _run(state)
    assert is_ask(result), "Bare affirmation must re-ask, not close"
    mock_llm.assert_not_called()
    response = get_response(result)
    # Must offer summary or specific question — must not echo session content
    assert any(w in response.lower() for w in ["summary", "recap", "specific", "details"]), (
        f"Nudge must offer summary/details option, got: {response!r}"
    )
    # Must not contain dollar amounts (session data must not appear)
    assert "$" not in response, f"Nudge must not contain session data: {response!r}"


@pytest.mark.happy
@pytest.mark.asyncio
async def test_bare_affirmation_sure_returns_nudge(mock_llm) -> None:
    state = make_fu_state(messages=[_msg("assistant", "Is there anything else?"), _msg("user", "sure")])
    result = await _run(state)
    assert is_ask(result)
    mock_llm.assert_not_called()


@pytest.mark.happy
@pytest.mark.asyncio
async def test_bare_affirmation_okay_returns_nudge(mock_llm) -> None:
    state = make_fu_state(messages=[_msg("assistant", "Is there anything else?"), _msg("user", "okay")])
    result = await _run(state)
    assert is_ask(result)
    mock_llm.assert_not_called()


@pytest.mark.happy
@pytest.mark.asyncio
async def test_after_nudge_summary_request_calls_llm(mock_llm) -> None:
    """After the nudge, a real question must reach the LLM."""
    state = make_fu_state(
        messages=[
            _msg("assistant", "Is there anything else?"),
            _msg("user", "yes"),
            _msg("assistant", "Would you like a summary of what we covered today?"),
            _msg("user", "yes please give me a summary"),
        ]
    )
    result = await _run(state)
    assert is_ask(result)
    mock_llm.assert_called_once()


@pytest.mark.regression
@pytest.mark.asyncio
async def test_yes_does_not_repeat_previous_message(mock_llm) -> None:
    """'yes' must never produce a response that repeats the last assistant message."""
    last_ai = "Great! I've sent the Care Coach details to your email."
    state = make_fu_state(messages=[_msg("assistant", last_ai), _msg("user", "yes")])
    result = await _run(state)
    response = get_response(result)
    assert response != last_ai, "Response must not be identical to the previous message"
    assert "care coach" not in response.lower(), "Nudge must not repeat care coach content"


# ---------------------------------------------------------------------------
# SECTION 9 — _build_session_snapshot unit tests (marker: response_check)
# ---------------------------------------------------------------------------


@pytest.mark.response_check
def test_snapshot_includes_benefits_when_present() -> None:
    """Snapshot serialises benefit fields — individual deductible appears as $750."""
    state = {
        "individual_deductible": "750",
        "family_deductible": "2500",
        "coinsurance_percent": "20",
        "individual_oop_max": "3000",
        "family_oop_max": "7000",
    }
    snapshot = _build_session_snapshot(state)
    assert "$750" in snapshot, f"Expected $750 in snapshot: {snapshot!r}"
    assert "$2500" in snapshot, f"Expected $2500 in snapshot: {snapshot!r}"
    assert "20%" in snapshot, f"Expected coinsurance in snapshot: {snapshot!r}"


@pytest.mark.response_check
def test_snapshot_includes_provider_sent_flag() -> None:
    """provider_list_sent=True → snapshot contains 'provider list was sent'."""
    state = {
        "provider_type": "Primary Care Physician",
        "zip_code_used": "12139",
        "provider_list_sent": True,
    }
    snapshot = _build_session_snapshot(state)
    assert "provider list was sent" in snapshot, (
        f"Expected 'provider list was sent' in snapshot: {snapshot!r}"
    )


@pytest.mark.response_check
def test_snapshot_empty_when_no_data() -> None:
    """Empty state → snapshot is an empty string (nothing to summarise)."""
    snapshot = _build_session_snapshot({})
    assert snapshot == "", f"Expected empty snapshot, got: {snapshot!r}"


@pytest.mark.response_check
def test_snapshot_includes_care_coach_detail() -> None:
    """care_coach_details_sent=True + fax delivery → 'Care Coach details were sent' in snapshot."""
    state = {
        "care_coach_details_sent": True,
        "delivery_method": "fax",
        "fax": FAX_ON_FILE,
    }
    snapshot = _build_session_snapshot(state)
    assert "Care Coach details were sent" in snapshot, (
        f"Expected care coach sentence in snapshot: {snapshot!r}"
    )
    assert FAX_ON_FILE in snapshot, f"Expected fax number in snapshot: {snapshot!r}"


# ---------------------------------------------------------------------------
# SECTION 10 — No new-intent re-routing (N/A documentation test) (marker: happy)
# ---------------------------------------------------------------------------


@pytest.mark.happy
@pytest.mark.asyncio
async def test_question_not_rerouted_to_domain_agent(mock_llm) -> None:
    """FollowUpAgent never re-routes to domain agents; it answers everything from context.

    Note: FollowUpAgent has NO new-intent detection. Any question — even one that mentions
    finding a provider — is answered from the session snapshot, then the agent asks the
    continuation question. It never routes back to provider_search_agent or intake_agent.
    """
    state = make_fu_state(
        messages=[
            _msg("assistant", "Is there anything else?"),
            _msg("user", "I actually want to find another provider as well"),
        ]
    )
    result = await _run(state)
    assert result.get("next_node") not in ("provider_search_agent", "intake_agent", "verification_agent"), (
        f"FollowUpAgent must never re-route to domain agents; got {result.get('next_node')!r}"
    )
    assert is_ask(result) or is_closure(result), (
        f"FollowUpAgent must answer or close, not route elsewhere; got {result!r}"
    )


# ---------------------------------------------------------------------------
# SECTION 11 — Cannot-answer path with empty snapshot (marker: unhappy)
# ---------------------------------------------------------------------------


@pytest.mark.unhappy
@pytest.mark.asyncio
async def test_cannot_answer_empty_snapshot(mock_llm) -> None:
    """All benefit/provider fields empty → LLM mock returns '' → sorry + continuation re-ask."""
    state = make_fu_state(
        individual_deductible="",
        family_deductible="",
        coinsurance_percent="",
        individual_oop_max="",
        family_oop_max="",
        provider_type="",
        zip_code_used="",
        provider_list_sent=False,
        care_coach_details_sent=False,
        care_coach_offered=False,
        messages=[_msg("assistant", "Anything else?"), _msg("user", "what are my vision benefits")],
    )
    result = await _run(state)
    assert is_ask(result), "Cannot-answer path must still re-ask, not close"
    response = get_response(result)
    continuation_phrases = ["anything else", "anything i can", "other questions", "i can assist"]
    assert any(p in response.lower() for p in continuation_phrases), (
        f"Continuation question missing from cannot-answer response: {response!r}"
    )


# ---------------------------------------------------------------------------
# SECTION 12 — Bare affirmation set coverage (marker: happy)
# ---------------------------------------------------------------------------


@pytest.mark.happy
@pytest.mark.parametrize("affirmation", sorted(BARE_AFFIRMATIONS - CLOSURE_KEYWORDS))
@pytest.mark.asyncio
async def test_bare_affirmations_complete_coverage(mock_llm, affirmation) -> None:
    """Entries in BARE_AFFIRMATIONS that are not also CLOSURE_KEYWORDS must return a nudge.

    "ok" and "okay" appear in both sets; closure detection runs first so those entries
    correctly reach is_closure — they are covered by test_bare_affirmation_okay_returns_nudge
    (which documents a known constants conflict) and are excluded here to avoid redundant failures.
    """
    state = make_fu_state(
        messages=[_msg("assistant", "Is there anything else I can help you with?"), _msg("user", affirmation)]
    )
    result = await _run(state)
    assert is_ask(result), (
        f"Bare affirmation {affirmation!r} must return a nudge (is_ask), "
        f"got next_node={result.get('next_node')!r}, is_interrupt={result.get('is_interrupt')}"
    )
    mock_llm.assert_not_called()


# ---------------------------------------------------------------------------
# SECTION 13 — Closure keyword set coverage (marker: happy)
# ---------------------------------------------------------------------------

_CLOSURE_SAMPLE = [
    "no",
    "nope",
    "no thanks",
    "that's all",
    "done",
    "bye",
    "goodbye",
    "that was helpful",
    "nothing else",
    "i'm done",
]


@pytest.mark.happy
@pytest.mark.parametrize("keyword", _CLOSURE_SAMPLE)
@pytest.mark.asyncio
async def test_closure_keywords_complete_coverage(mock_llm, keyword) -> None:
    """A representative subset of CLOSURE_KEYWORDS must signal closure without calling the LLM."""
    assert keyword in CLOSURE_KEYWORDS, f"{keyword!r} is not in CLOSURE_KEYWORDS — update _CLOSURE_SAMPLE"
    state = make_fu_state(
        messages=[_msg("assistant", "Is there anything else?"), _msg("user", keyword)]
    )
    result = await _run(state)
    assert is_closure(result), (
        f"Closure keyword {keyword!r} must signal closure, "
        f"got next_node={result.get('next_node')!r}, is_interrupt={result.get('is_interrupt')}"
    )
    mock_llm.assert_not_called()
