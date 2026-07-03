"""Phase 3 acceptance tests: WAIT handling in _collect_slot.

Covers:
(a) bare "give me a minute" → static wait ack, attempt_count unchanged,
    awaiting_slot unchanged, zero generation-LLM (Gemini) invocations
(b) "hold on, it's M451982" → slot confirmed via the normal answered path
(c) three consecutive waits → nudge message (still not a failure)
(d) "I don't have my card" → cannot-provide escalation, never a wait ack

Run with:  pytest tests/test_wait_phase3.py
"""

import sys
import types

import pytest

from agent.core.agent import BaseAgent
from agent.core.constants import INTERRUPTION_PATTERNS, MAX_WAIT_TURNS
from agent.core.slot_manager import _InternalSlotConfig
from agent.llm.schema import EventType, WorkerResult
from agent.responses.static import MSG_WAIT_ACK, MSG_WAIT_NUDGE
from agent.slots.normalizers import normalize_member_id
from agent.slots.validators import validate_member_id
from agent.utils import detect_wait_request


class _StubAgent(BaseAgent):
    AGENT_NAME = "test_agent"

    async def run(self, state):  # pragma: no cover - not exercised
        return {}


@pytest.fixture
def gemini_calls(monkeypatch):
    """Replace the lazily imported generation-LLM module with a call recorder."""
    calls: list[dict] = []
    stub = types.ModuleType("agent.llm.response_generator")

    async def generate_recovery_message(**kwargs):
        calls.append(kwargs)
        return "GENERATED"

    stub.generate_recovery_message = generate_recovery_message
    monkeypatch.setitem(sys.modules, "agent.llm.response_generator", stub)
    return calls


def _config() -> _InternalSlotConfig:
    return _InternalSlotConfig(
        slot_name="member_id",
        prompt="May I have your member ID?",
        normalizer=normalize_member_id,
        validator=validate_member_id,
    )


def _state(wait_count: int = 0) -> dict:
    return {
        "awaiting_slot": "member_id",
        "wait_count": wait_count,
        "ambiguous_counts": {},
        "slot_attempts": {},
        "member_id": "",
        "app_run_id": "test-run",
    }


def _messages(user_text: str) -> list:
    return [
        {"role": "assistant", "content": "May I have your member ID?"},
        {"role": "user", "content": user_text},
    ]


class TestWaitAck:
    async def test_bare_wait_via_llm_event(self, gemini_calls):
        agent = _StubAgent()
        decision = WorkerResult(event_type=EventType.WAIT)
        value, interrupt = await agent._collect_slot(
            _state(), _config(), _messages("give me a minute"), decision=decision
        )
        assert value is None
        assert interrupt["messages"]["content"] in MSG_WAIT_ACK
        assert interrupt["awaiting_slot"] == "member_id"
        assert interrupt["wait_count"] == 1
        assert agent.get_slot("member_id").attempt_count == 0
        assert "ambiguous_counts" not in interrupt
        assert gemini_calls == []

    async def test_wait_mislabelled_as_ambiguous_rescued_by_regex(self, gemini_calls):
        agent = _StubAgent()
        decision = WorkerResult(event_type=EventType.AMBIGUOUS)
        value, interrupt = await agent._collect_slot(
            _state(), _config(), _messages("give me a minute"), decision=decision
        )
        assert value is None
        assert interrupt["messages"]["content"] in MSG_WAIT_ACK
        assert interrupt["awaiting_slot"] == "member_id"
        assert interrupt["wait_count"] == 1
        assert agent.get_slot("member_id").attempt_count == 0
        assert gemini_calls == []


class TestWaitPlusValue:
    async def test_value_wins_over_wait_phrase(self, gemini_calls):
        agent = _StubAgent()
        decision = WorkerResult(extracted={"member_id": "M451982"}, event_type=EventType.ANSWERED)
        value, interrupt = await agent._collect_slot(
            _state(),
            _config(),
            _messages("hold on, it's M451982"),
            pre_extracted="M451982",
            decision=decision,
        )
        assert value == "M451982"
        assert interrupt is None
        assert agent.get_slot("member_id").confirmed is True
        assert gemini_calls == []

    def test_regex_defers_to_extraction_when_value_present(self):
        assert detect_wait_request("hold on, it's M451982") is False
        assert detect_wait_request("give me a minute") is True
        assert detect_wait_request("hold on, let me grab my card") is True


class TestWaitNudge:
    async def test_third_consecutive_wait_nudges(self, gemini_calls):
        agent = _StubAgent()
        decision = WorkerResult(event_type=EventType.WAIT)
        value, interrupt = await agent._collect_slot(
            _state(wait_count=MAX_WAIT_TURNS - 1),
            _config(),
            _messages("just a sec"),
            decision=decision,
        )
        assert value is None
        assert interrupt["wait_count"] == MAX_WAIT_TURNS
        expected = MSG_WAIT_NUDGE[0].format(slot_label="member id")
        assert interrupt["messages"]["content"] == expected
        assert agent.get_slot("member_id").attempt_count == 0  # still not a failure
        assert gemini_calls == []


class TestCannotProvideOutranksWait:
    async def test_cannot_provide_escalates_even_if_llm_says_wait(self, gemini_calls):
        agent = _StubAgent()
        decision = WorkerResult(event_type=EventType.WAIT)
        value, result = await agent._collect_slot(
            _state(), _config(), _messages("I don't have my card"), decision=decision
        )
        assert value is None
        assert result["escalation_pre_message"].startswith("No problem")
        assert "wait_count" not in result  # never took the WAIT branch
        assert gemini_calls == []

    def test_detect_wait_request_precedence(self):
        assert detect_wait_request("I don't have my card") is False
        assert detect_wait_request(None) is False
        assert detect_wait_request("M110781") is False


class TestNonWaitTurnsResetCounter:
    async def test_retry_interrupt_resets_wait_count(self, gemini_calls):
        agent = _StubAgent()
        decision = WorkerResult(event_type=EventType.ANSWERED)
        value, interrupt = await agent._collect_slot(
            _state(wait_count=2), _config(), _messages("umm what?"), decision=decision
        )
        assert value is None
        assert interrupt["wait_count"] == 0
        assert len(gemini_calls) == 1  # normal retry path still generates

    async def test_first_ask_resets_wait_count(self, gemini_calls):
        agent = _StubAgent()
        state = _state(wait_count=2)
        state["awaiting_slot"] = ""  # not yet awaiting this slot → first ask
        value, interrupt = await agent._collect_slot(state, _config(), [])
        assert value is None
        assert interrupt["wait_count"] == 0
        assert gemini_calls == []


class TestInterruptionGuardOverlap:
    def test_hold_on_removed_from_interruption_patterns(self):
        # WAIT must win over the INTERRUPTION keyword fallback: no pattern may
        # match a plain "hold on" wait phrase.
        assert all("hold on" not in p for p in INTERRUPTION_PATTERNS)
