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
        fam  = state.get("family_deductible", "")
        coins = state.get("coinsurance_percent", "")
        indv_oop = state.get("individual_oop_max", "")
        fam_oop  = state.get("family_oop_max", "")
        if indv:
            return (
                f"Your individual deductible is ${indv} per year, and the family "
                f"deductible is ${fam}. After meeting the deductible, you pay "
                f"{coins}% coinsurance. Your individual out-of-pocket max is "
                f"${indv_oop} and the family max is ${fam_oop}. After that, your "
                f"plan covers 100% of in-network services."
            )
        return ""   # signals cannot-answer path

    mock = AsyncMock(side_effect=_fn)
    monkeypatch.setattr(
        "agent.agents.follow_up.agent.generate_follow_up_answer", mock
    )
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
        messages=[_msg("assistant", "Would you like help with anything else?"),
                  _msg("user", "No thanks that was helpful")]
    )
    result = await _run(state)
    assert is_closure(result), f"Expected closure signal, got: {result.get('next_node')}"
    mock_llm.assert_not_called()


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_no_signals_closure(mock_llm) -> None:
    state = make_fu_state(
        messages=[_msg("assistant", "Would you like help with anything else?"),
                  _msg("user", "No")]
    )
    result = await _run(state)
    assert is_closure(result)
    mock_llm.assert_not_called()


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_bye_signals_closure(mock_llm) -> None:
    state = make_fu_state(
        messages=[_msg("assistant", "Would you like help with anything else?"),
                  _msg("user", "bye")]
    )
    result = await _run(state)
    assert is_closure(result)
    mock_llm.assert_not_called()


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_that_was_helpful_signals_closure(mock_llm) -> None:
    state = make_fu_state(
        messages=[_msg("assistant", "Would you like help with anything else?"),
                  _msg("user", "that was helpful")]
    )
    result = await _run(state)
    assert is_closure(result)


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_thank_you_signals_closure(mock_llm) -> None:
    state = make_fu_state(
        messages=[_msg("assistant", "Would you like help with anything else?"),
                  _msg("user", "thank you")]
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
        messages=[_msg("assistant", "Would you like help with anything else?"),
                  _msg("user", "Can you summarize the PCP benefits?")]
    )
    result = await _run(state)
    assert is_ask(result), f"Expected ask, got next_node={result.get('next_node')!r}"
    mock_llm.assert_called_once()


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_follow_up_turn_count_incremented(mock_llm) -> None:
    state = make_fu_state(
        follow_up_turn_count=1,
        messages=[_msg("assistant", "Anything else?"),
                  _msg("user", "what is my deductible")]
    )
    result = await _run(state)
    assert result.get("follow_up_turn_count") == 2


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_last_question_stored(mock_llm) -> None:
    question = "Can you summarize the PCP benefits?"
    state = make_fu_state(
        messages=[_msg("assistant", "Anything else?"), _msg("user", question)]
    )
    result = await _run(state)
    assert result.get("follow_up_last_question") == question


# ---------------------------------------------------------------------------
# SECTION 3 — Response content (marker: response_check)
# ---------------------------------------------------------------------------

@pytest.mark.response_check
@pytest.mark.asyncio
async def test_response_contains_individual_deductible(mock_llm) -> None:
    state = make_fu_state(
        messages=[_msg("assistant", "Anything else?"),
                  _msg("user", "what is my deductible")]
    )
    result = await _run(state)
    assert "$750" in get_response(result), get_response(result)


@pytest.mark.response_check
@pytest.mark.asyncio
async def test_response_contains_family_deductible(mock_llm) -> None:
    state = make_fu_state(
        messages=[_msg("assistant", "Anything else?"),
                  _msg("user", "summarize my benefits")]
    )
    result = await _run(state)
    assert "$2500" in get_response(result), get_response(result)


@pytest.mark.response_check
@pytest.mark.asyncio
async def test_response_contains_continuation_question(mock_llm) -> None:
    state = make_fu_state(
        messages=[_msg("assistant", "Anything else?"),
                  _msg("user", "what is my deductible")]
    )
    result = await _run(state)
    response = get_response(result)
    continuation_phrases = [
        "anything else", "anything i can", "other questions", "i can assist",
    ]
    assert any(p in response.lower() for p in continuation_phrases), (
        f"Continuation question missing from: {response!r}"
    )


@pytest.mark.response_check
@pytest.mark.asyncio
async def test_response_closure_has_no_message(mock_llm) -> None:
    """Closure signal must produce an empty message — closure_agent speaks."""
    state = make_fu_state(
        messages=[_msg("assistant", "Anything else?"),
                  _msg("user", "no")]
    )
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
        individual_deductible="",   # no benefits in state
        messages=[_msg("assistant", "Anything else?"),
                  _msg("user", "what are my vision benefits")]
    )
    result = await _run(state)
    assert is_ask(result), "Cannot-answer must still re-ask, not close"
    response = get_response(result)
    continuation_phrases = [
        "anything else", "anything i can", "other questions", "i can assist",
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
        messages=[_msg("assistant", "Anything else?"),
                  _msg("user", "tell me more")]
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
        messages=[_msg("assistant", "Is there anything else?"),
                  _msg("user", "Can you summarize the PCP benefits?")]
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
    state = make_fu_state(
        messages=[_msg("assistant", "Anything else?"), _msg("user", "   ")]
    )
    result = await _run(state)
    # Empty utterance → LLM returns '' → cannot-answer path → ask again
    assert is_ask(result), (
        "Empty utterance should not trigger closure"
    )


@pytest.mark.regression
@pytest.mark.asyncio
async def test_regression_turn_count_not_incremented_on_closure(mock_llm) -> None:
    """Closure must not increment turn count (no further turns needed)."""
    state = make_fu_state(
        follow_up_turn_count=2,
        messages=[_msg("assistant", "Anything else?"), _msg("user", "no")]
    )
    result = await _run(state)
    assert is_closure(result)
    # turn count in context_updates should remain 2 (not 3)
    ctx = (result.get("last_agent_signal") or {}).get("context_updates") or {}
    stored_count = ctx.get("follow_up_turn_count", 2)
    assert stored_count == 2, (
        f"Closure must not increment turn count, got {stored_count}"
    )
