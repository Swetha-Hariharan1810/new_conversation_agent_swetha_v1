"""
Mock tests — no external credentials required.
These mirror test_intake_agent.py but replace the LLM with deterministic mocks.
Run with: pytest src/agent/tests/test_intake_agent_mock.py -v
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.agents.intake.agent import IntakeAgent
from agent.agents.intake.constants import GREETING, INTENT_BRIDGE_MSG
from agent.llm.schema import GuardType, WorkerResult
from agent.orchestration.orchestration import AgentNode

GREETING_TEXT: str = GREETING
BRIDGE_TEXT: str = INTENT_BRIDGE_MSG

# ---------------------------------------------------------------------------
# Shared helpers (identical to live test)
# ---------------------------------------------------------------------------


def _make_state(messages=None, **overrides) -> dict:
    """Return the minimal state dict IntakeAgent needs."""
    state: dict = {
        "messages": messages if messages is not None else [],
        "metadata_events": [],
        "is_interrupt": False,
        "next_node": "",
        "app_run_id": str(uuid.uuid4()),
        "slot_attempts": {},
        "call_intent": "",
        "conversation_summary": None,
        "awaiting_slot": "",
        "active_agent": "",
    }
    state.update(overrides)
    return state


def _with_turn(state: dict, user_text: str, ai_text: str | None = None) -> dict:
    """Return a new state with messages extended by an optional AI turn then a user turn."""
    new_messages = list(state.get("messages") or [])
    if ai_text is not None:
        new_messages.append({"role": "assistant", "content": ai_text})
    new_messages.append({"role": "user", "content": user_text})
    return {**state, "messages": new_messages}


async def run_intake(state: dict) -> dict:
    """Convenience wrapper: build IntakeAgent from state and execute."""
    return await IntakeAgent.from_state(state).execute(state)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_RECOVERY_MSGS: dict[str, str] = {
    "first_name": "I'm sorry — could you repeat your first name?",
    "last_name": "I'm sorry — could you repeat your last name?",
    "member_id": "Could you provide your member ID again, please?",
    "dob": "Could you repeat your date of birth, please?",
    "relationship": "Are you the plan holder or a subscriber?",
    "phone_confirmed": "I'm sorry — could you confirm your phone number on file?",
    "intent": "I'm sorry, could you clarify what you need help with?",
}


@pytest.fixture
def mock_llm_extraction(monkeypatch):
    """Default: provider_services intent. Tests override as needed."""
    fake_llm = MagicMock()
    monkeypatch.setattr("agent.agents.intake.agent.get_extraction_llm", lambda: fake_llm)

    mock = AsyncMock(
        return_value=WorkerResult(
            extracted={"intent": "provider_services"},
            guard=GuardType.NONE,
            guard_confidence=0.0,
        )
    )
    monkeypatch.setattr("agent.agents.intake.agent.extract_intake_intent", mock)
    return mock


@pytest.fixture
def mock_recovery(monkeypatch):
    """Return a slot-specific recovery message instead of calling LLM 2."""

    async def _fn(*, slot_name, attempt, guard, last_messages, **kwargs):
        return _RECOVERY_MSGS.get(slot_name, "I'm sorry, could you repeat that?")

    mock = AsyncMock(side_effect=_fn)
    monkeypatch.setattr("agent.llm.response_generator.generate_recovery_message", mock)
    return mock


@pytest.fixture(autouse=True)
def all_mocks(mock_llm_extraction, mock_recovery):
    pass


# ---------------------------------------------------------------------------
# SECTION 2 — Happy path
# ---------------------------------------------------------------------------


@pytest.mark.happy
@pytest.mark.asyncio
async def test_greeting_returns_static_message() -> None:
    state = _make_state()  # no messages — LLM not called
    result = await run_intake(state)

    assert result["is_interrupt"] is True
    assert result["next_node"] == "intake_agent"
    assert GREETING_TEXT in result["messages"]["content"]
    assert result.get("call_intent", "") == ""


@pytest.mark.happy
@pytest.mark.asyncio
async def test_provider_services_explicit() -> None:
    state = _make_state()
    state = _with_turn(
        state,
        user_text="I need to find a cardiologist in my network.",
        ai_text=GREETING_TEXT,
    )
    result = await run_intake(state)

    assert result["call_intent"] == "provider_services"
    assert result["next_node"] == AgentNode.VERIFICATION.value
    assert result["is_interrupt"] is True
    assert BRIDGE_TEXT in result["messages"]["content"]


@pytest.mark.happy
@pytest.mark.asyncio
async def test_provider_services_short() -> None:
    state = _make_state()
    state = _with_turn(state, user_text="provider", ai_text=GREETING_TEXT)
    result = await run_intake(state)

    assert result["call_intent"] == "provider_services"
    assert result["next_node"] == AgentNode.VERIFICATION.value
    assert result["is_interrupt"] is True
    assert BRIDGE_TEXT in result["messages"]["content"]


@pytest.mark.happy
@pytest.mark.asyncio
async def test_provider_services_natural() -> None:
    state = _make_state()
    state = _with_turn(
        state,
        user_text="uh yeah so I need to find a doctor in my network",
        ai_text=GREETING_TEXT,
    )
    result = await run_intake(state)

    assert result["call_intent"] == "provider_services"
    assert result["next_node"] == AgentNode.VERIFICATION.value
    assert result["is_interrupt"] is True
    assert BRIDGE_TEXT in result["messages"]["content"]


@pytest.mark.happy
@pytest.mark.asyncio
async def test_claim_services_explicit(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent.agents.intake.agent.extract_intake_intent",
        AsyncMock(
            return_value=WorkerResult(
                extracted={"intent": "claim_services"},
                guard=GuardType.NONE,
                guard_confidence=0.0,
            )
        ),
    )
    state = _make_state()
    state = _with_turn(
        state,
        user_text="I want to check on my denied claim from last month.",
        ai_text=GREETING_TEXT,
    )
    result = await run_intake(state)

    assert result["call_intent"] == "claim_services"
    assert result["next_node"] == AgentNode.VERIFICATION.value


@pytest.mark.happy
@pytest.mark.asyncio
async def test_claim_services_short(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent.agents.intake.agent.extract_intake_intent",
        AsyncMock(
            return_value=WorkerResult(
                extracted={"intent": "claim_services"},
                guard=GuardType.NONE,
                guard_confidence=0.0,
            )
        ),
    )
    state = _make_state()
    state = _with_turn(state, user_text="claim status please", ai_text=GREETING_TEXT)
    result = await run_intake(state)

    assert result["call_intent"] == "claim_services"


@pytest.mark.happy
@pytest.mark.asyncio
async def test_metadata_event_emitted() -> None:
    state = _make_state()
    state = _with_turn(state, user_text="find a provider", ai_text=GREETING_TEXT)
    result = await run_intake(state)

    events = result["metadata_events"]
    assert isinstance(events, list)
    assert len(events) >= 1

    first = events[0]
    assert first["eventType"] == "CallAgentField"
    assert first["data"]["field"] == "call_intent"
    assert first["data"]["value"] in {"provider_services", "claim_services"}


@pytest.mark.happy
@pytest.mark.asyncio
async def test_next_node_is_verification_after_bridge() -> None:
    state = _make_state()
    state = _with_turn(state, user_text="provider services", ai_text=GREETING_TEXT)
    result = await run_intake(state)

    assert result["next_node"] == AgentNode.VERIFICATION.value


# ---------------------------------------------------------------------------
# SECTION 3 — Non-happy path
# ---------------------------------------------------------------------------


@pytest.mark.unhappy
@pytest.mark.asyncio
async def test_unclear_intent_first_attempt(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent.agents.intake.agent.extract_intake_intent",
        AsyncMock(
            return_value=WorkerResult(
                extracted={"intent": "unclear"},
                guard=GuardType.NONE,
                guard_confidence=0.0,
            )
        ),
    )
    state = _make_state()
    state = _with_turn(
        state,
        user_text="I have a question about my insurance.",
        ai_text=GREETING_TEXT,
    )
    result = await run_intake(state)

    assert result["is_interrupt"] is True
    assert result["next_node"] == "intake_agent"
    assert result.get("call_intent", "") == ""
    assert result["messages"]["content"] != ""


@pytest.mark.unhappy
@pytest.mark.asyncio
async def test_unclear_intent_second_attempt(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent.agents.intake.agent.extract_intake_intent",
        AsyncMock(
            return_value=WorkerResult(
                extracted={"intent": "unclear"},
                guard=GuardType.NONE,
                guard_confidence=0.0,
            )
        ),
    )
    state = _make_state(slot_attempts={"intent": {"attempt_count": 1}})
    state = _with_turn(
        state,
        user_text="I'm not sure what I need help with",
        ai_text=GREETING_TEXT,
    )
    result = await run_intake(state)

    assert result["is_interrupt"] is True
    assert result["next_node"] == "intake_agent"


@pytest.mark.unhappy
@pytest.mark.asyncio
async def test_unclear_intent_max_retries_escalates(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent.agents.intake.agent.extract_intake_intent",
        AsyncMock(
            return_value=WorkerResult(
                extracted={"intent": "unclear"},
                guard=GuardType.NONE,
                guard_confidence=0.0,
            )
        ),
    )
    state = _make_state(slot_attempts={"intent": {"attempt_count": 2}})
    state = _with_turn(state, user_text="umm I don't know", ai_text=GREETING_TEXT)
    result = await run_intake(state)

    assert result["next_node"] == AgentNode.ESCALATION.value
    assert result["is_interrupt"] is False


@pytest.mark.unhappy
@pytest.mark.asyncio
async def test_offtopic_first_attempt(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent.agents.intake.agent.extract_intake_intent",
        AsyncMock(
            return_value=WorkerResult(
                extracted=None,
                guard=GuardType.OFFTOPIC_GLOBAL,
                guard_confidence=0.9,
            )
        ),
    )
    state = _make_state()
    state = _with_turn(
        state,
        user_text="What's the weather like in Miami today?",
        ai_text=GREETING_TEXT,
    )
    result = await run_intake(state)

    assert result["is_interrupt"] is True
    assert result["next_node"] == "intake_agent"
    assert result.get("call_intent", "") == ""


@pytest.mark.unhappy
@pytest.mark.asyncio
async def test_offtopic_max_retries_escalates(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent.agents.intake.agent.extract_intake_intent",
        AsyncMock(
            return_value=WorkerResult(
                extracted=None,
                guard=GuardType.OFFTOPIC_GLOBAL,
                guard_confidence=0.9,
            )
        ),
    )
    # offtopic_global_count=1: guard increments to 2 >= MAX_SLOT_ATTEMPTS(2) → ESCALATE
    state = _make_state(offtopic_global_count=1)
    state = _with_turn(
        state,
        user_text="Can you recommend a good restaurant?",
        ai_text=GREETING_TEXT,
    )
    result = await run_intake(state)

    assert result["next_node"] == AgentNode.ESCALATION.value


@pytest.mark.unhappy
@pytest.mark.asyncio
async def test_transfer_request_escalates_immediately(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent.agents.intake.agent.extract_intake_intent",
        AsyncMock(
            return_value=WorkerResult(
                extracted=None,
                guard=GuardType.TRANSFER_REQUEST,
                guard_confidence=0.95,
            )
        ),
    )
    state = _make_state()
    state = _with_turn(
        state,
        user_text="Get me a real person right now.",
        ai_text=GREETING_TEXT,
    )
    result = await run_intake(state)

    assert result["next_node"] == AgentNode.ESCALATION.value
    assert result["is_interrupt"] is False


@pytest.mark.unhappy
@pytest.mark.asyncio
async def test_transfer_request_polite(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent.agents.intake.agent.extract_intake_intent",
        AsyncMock(
            return_value=WorkerResult(
                extracted=None,
                guard=GuardType.TRANSFER_REQUEST,
                guard_confidence=0.95,
            )
        ),
    )
    state = _make_state()
    state = _with_turn(
        state,
        user_text="Can I speak to a representative please?",
        ai_text=GREETING_TEXT,
    )
    result = await run_intake(state)

    assert result["next_node"] == AgentNode.ESCALATION.value


@pytest.mark.unhappy
@pytest.mark.asyncio
async def test_asr_empty_utterance(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent.agents.intake.agent.extract_intake_intent",
        AsyncMock(
            return_value=WorkerResult(
                extracted={"intent": "unclear"},
                guard=GuardType.NONE,
                guard_confidence=0.0,
            )
        ),
    )
    state = _make_state()
    state = _with_turn(state, user_text="", ai_text=GREETING_TEXT)
    result = await run_intake(state)

    assert result["is_interrupt"] is True
    assert result["next_node"] == "intake_agent"
    assert result.get("call_intent", "") == ""


@pytest.mark.unhappy
@pytest.mark.asyncio
async def test_asr_noise(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent.agents.intake.agent.extract_intake_intent",
        AsyncMock(
            return_value=WorkerResult(
                extracted={"intent": "unclear"},
                guard=GuardType.NONE,
                guard_confidence=0.0,
            )
        ),
    )
    state = _make_state()
    state = _with_turn(state, user_text="zzzshhh kkkk mmph", ai_text=GREETING_TEXT)
    result = await run_intake(state)

    assert result["is_interrupt"] is True
    assert result.get("call_intent", "") == ""


# ---------------------------------------------------------------------------
# SECTION 4 — Re-entry guard
#
# Verifies that when call_intent is already set the LLM is bypassed entirely
# and the agent signals complete without re-delivering the bridge message.
# ---------------------------------------------------------------------------


@pytest.mark.happy
@pytest.mark.asyncio
async def test_reentry_guard_skips_llm() -> None:
    state = _make_state()
    state["call_intent"] = "provider_services"  # already classified
    state["messages"] = [{"role": "user", "content": "its emily"}]
    result = await run_intake(state)

    assert result["is_interrupt"] is False
    assert result["next_node"] == "orchestrator"
    assert result.get("call_intent") == "provider_services"


# ---------------------------------------------------------------------------
# SECTION 5 — Bug regression tests (marker: regression)
# ---------------------------------------------------------------------------


@pytest.mark.regression
@pytest.mark.asyncio
async def test_unclear_first_attempt_uses_open_question(monkeypatch) -> None:
    """First unclear intent attempt must use a natural open-ended question, not a menu."""
    monkeypatch.setattr(
        "agent.agents.intake.agent.extract_intake_intent",
        AsyncMock(
            return_value=WorkerResult(
                extracted={"intent": "unclear"},
                guard=GuardType.NONE,
                guard_confidence=0.0,
            )
        ),
    )
    state = _make_state(slot_attempts={"intent": {"attempt_count": 0}})
    state = _with_turn(state, user_text="Hi", ai_text=GREETING_TEXT)
    result = await run_intake(state)

    response = result["messages"]["content"]
    assert "provider services" not in response.lower(), "First attempt must not list menu options"
    assert "claim services" not in response.lower(), "First attempt must not list menu options"
    assert "?" in response, "Response should still be a question"
    assert result["is_interrupt"] is True
    assert result["next_node"] == "intake_agent"


@pytest.mark.regression
@pytest.mark.asyncio
async def test_unclear_second_attempt_mentions_options(monkeypatch) -> None:
    """Second unclear intent attempt may mention provider or claim categories."""
    monkeypatch.setattr(
        "agent.agents.intake.agent.extract_intake_intent",
        AsyncMock(
            return_value=WorkerResult(
                extracted={"intent": "unclear"},
                guard=GuardType.NONE,
                guard_confidence=0.0,
            )
        ),
    )

    async def _recovery_with_options(
        *, slot_name, attempt, guard, last_messages, slot_label_override=None, **kwargs
    ):
        if slot_name == "intent" and slot_label_override and "provider" in (slot_label_override or ""):
            return "Are you calling about provider services or a claim?"
        return _RECOVERY_MSGS.get(slot_name, "I'm sorry, could you repeat that?")

    monkeypatch.setattr(
        "agent.llm.response_generator.generate_recovery_message",
        AsyncMock(side_effect=_recovery_with_options),
    )

    state = _make_state(slot_attempts={"intent": {"attempt_count": 1}})
    state = _with_turn(state, user_text="I'm not sure", ai_text=GREETING_TEXT)
    result = await run_intake(state)

    response = result["messages"]["content"]
    assert "provider" in response.lower() or "claim" in response.lower(), (
        "Second attempt should guide caller toward provider/claim categories"
    )
    assert result["is_interrupt"] is True
    assert result["next_node"] == "intake_agent"


# ---------------------------------------------------------------------------
# SECTION 6 — OFFTOPIC_AGENT guard (marker: guards)
# ---------------------------------------------------------------------------


@pytest.mark.guards
@pytest.mark.asyncio
async def test_offtopic_agent_guard_below_max_redirects(monkeypatch) -> None:
    """OFFTOPIC_AGENT + attempts < MAX_CLARIFICATION_ATTEMPTS → re-ask (redirect path)."""
    monkeypatch.setattr(
        "agent.agents.intake.agent.extract_intake_intent",
        AsyncMock(
            return_value=WorkerResult(
                extracted=None,
                guard=GuardType.OFFTOPIC_AGENT,
                guard_confidence=0.9,
            )
        ),
    )
    # attempt_count=0: guard fires, agent checks 0 < 2 → return interrupt
    state = _make_state(slot_attempts={"intent": {"attempt_count": 0}})
    state = _with_turn(
        state,
        user_text="Can you tell me the best restaurants in Boston?",
        ai_text=GREETING_TEXT,
    )
    result = await run_intake(state)

    assert result["is_interrupt"] is True
    assert result["next_node"] == "intake_agent"
    assert result.get("call_intent", "") == ""


@pytest.mark.guards
@pytest.mark.asyncio
async def test_offtopic_agent_guard_at_max_escalates(monkeypatch) -> None:
    """OFFTOPIC_AGENT + attempts >= MAX_CLARIFICATION_ATTEMPTS → escalation."""
    monkeypatch.setattr(
        "agent.agents.intake.agent.extract_intake_intent",
        AsyncMock(
            return_value=WorkerResult(
                extracted=None,
                guard=GuardType.OFFTOPIC_AGENT,
                guard_confidence=0.9,
            )
        ),
    )
    # attempt_count=2: guard fires, agent checks 2 >= 2 → escalate
    state = _make_state(slot_attempts={"intent": {"attempt_count": 2}})
    state = _with_turn(
        state,
        user_text="Actually tell me how to bake a chocolate cake.",
        ai_text=GREETING_TEXT,
    )
    result = await run_intake(state)

    assert result["next_node"] == AgentNode.ESCALATION.value
    assert result["is_interrupt"] is False


# ---------------------------------------------------------------------------
# SECTION 7 — Re-entry fast path: claim_services variant (marker: happy)
# ---------------------------------------------------------------------------


@pytest.mark.happy
@pytest.mark.asyncio
async def test_reentry_claim_services_skips_llm() -> None:
    """call_intent='claim_services' already set → signal_complete, no LLM call, no bridge re-delivery."""
    state = _make_state(call_intent="claim_services")
    state["messages"] = [{"role": "user", "content": "yes my name is emily"}]
    result = await run_intake(state)

    assert result["is_interrupt"] is False
    assert result["next_node"] == "orchestrator"
    assert result.get("call_intent") == "claim_services"


# ---------------------------------------------------------------------------
# SECTION 8 — app_run_id generated on first turn (marker: happy)
# ---------------------------------------------------------------------------


@pytest.mark.happy
@pytest.mark.asyncio
async def test_app_run_id_generated_if_absent() -> None:
    """state has empty app_run_id → result['app_run_id'] is a non-empty string (UUID)."""
    state = _make_state(app_run_id="")  # empty string is falsy → new UUID generated
    result = await run_intake(state)

    assert result.get("app_run_id"), "app_run_id must be non-empty in result"
    assert isinstance(result["app_run_id"], str)
    assert len(result["app_run_id"]) > 0


# ---------------------------------------------------------------------------
# SECTION 9 — Greeting emits no CallAgentField event (marker: happy)
# ---------------------------------------------------------------------------


@pytest.mark.happy
@pytest.mark.asyncio
async def test_greeting_has_no_call_agent_field_event() -> None:
    """Turn 0 (greeting, no messages): metadata_events must NOT contain a CallAgentField event."""
    state = _make_state()  # no messages → greeting path, call_intent not yet known
    result = await run_intake(state)

    events = result.get("metadata_events") or []
    call_agent_events = [e for e in events if e.get("eventType") == "CallAgentField"]
    assert len(call_agent_events) == 0, (
        f"Greeting turn must not emit CallAgentField events; found: {call_agent_events}"
    )


# ---------------------------------------------------------------------------
# SECTION 10 — SELF_HARM guard (marker: guards)
# ---------------------------------------------------------------------------


@pytest.mark.guards
@pytest.mark.asyncio
async def test_guard_self_harm_escalates_intake(monkeypatch) -> None:
    """SELF_HARM guard → immediate escalation regardless of intent attempts."""
    monkeypatch.setattr(
        "agent.agents.intake.agent.extract_intake_intent",
        AsyncMock(
            return_value=WorkerResult(
                extracted=None,
                guard=GuardType.SELF_HARM,
                guard_confidence=0.95,
            )
        ),
    )
    state = _make_state()
    state = _with_turn(
        state,
        user_text="I just don't want to be here anymore.",
        ai_text=GREETING_TEXT,
    )
    result = await run_intake(state)

    assert result["next_node"] == AgentNode.ESCALATION.value
    assert result["is_interrupt"] is False
