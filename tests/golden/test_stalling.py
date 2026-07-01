"""
test_stalling.py — EventType.STALLING handling.

When the caller asks for time ("give me a few seconds", "hold on, let me grab
that"), the agent must acknowledge ONLY ("take your time") — it must NOT re-prompt
the slot question and must NOT count a failed attempt (so a few stalls can never
escalate the way the UAT log showed). EventType.STALLING is the primary signal;
detect_stalling() is the deterministic regex fallback.
"""

from __future__ import annotations

import re

import pytest

import tests.golden  # noqa: F401 — ensures src/ is on sys.path
from agent.core.agent import BaseAgent
from agent.core.slot_manager import MAX_STALLS, _InternalSlotConfig
from agent.llm.schema import EventType, WorkerResult
from agent.slots.normalizers import normalize_member_id
from agent.slots.types import SlotType
from agent.slots.validators import validate_member_id

pytestmark = pytest.mark.regression


class _Probe(BaseAgent):
    AGENT_NAME = "verification_agent"

    async def run(self, state):  # pragma: no cover
        return {}


def _config():
    return _InternalSlotConfig(
        slot_name="member_id",
        prompt="",
        normalizer=normalize_member_id,
        validator=validate_member_id,
        slot_type=SlotType.MEMBER_ID,
    )


def _messages(user: str):
    return [
        {"role": "assistant", "content": "Could I get your Member ID?"},
        {"role": "user", "content": user},
    ]


def _state():
    return {"awaiting_slot": "member_id", "slot_attempts": {}}


def _ack_text(interrupt: dict) -> str:
    msgs = interrupt.get("messages")
    return msgs["content"] if isinstance(msgs, dict) else ""


# ── detector (regex fallback) ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "utterance",
    [
        "Give me a few seconds. Let me just get that for you.",
        "Sure. Just give me a few seconds.",
        "I'm just grabbing that. Thank you.",
        "hold on, let me find my card",
        "one moment please",
        "bear with me",
        "let me check real quick",
    ],
)
def test_detect_stalling_true(utterance):
    from agent.utils import detect_stalling

    assert detect_stalling(utterance) is True


@pytest.mark.parametrize(
    "utterance",
    [
        "M310188",
        "James Wilson",
        "yes",
        "I don't have it",  # decline, not a stall
        "I don't have my member ID",
        "give me the list of providers",  # no time word
        "April twelfth nineteen eighty eight",
    ],
)
def test_detect_stalling_false(utterance):
    from agent.utils import detect_stalling

    assert detect_stalling(utterance) is False


# ── live behaviour in _collect_slot ──────────────────────────────────────────


async def test_stalling_via_event_type_acknowledges_without_reasking():
    agent = _Probe.from_state(_state())
    decision = WorkerResult(event_type=EventType.STALLING)
    value, interrupt = await agent._collect_slot(
        _state(), _config(), _messages("Give me a few seconds."), pre_extracted="", decision=decision
    )
    assert value is None
    assert interrupt is not None
    # Still waiting for the slot, but the message is a PURE acknowledgement —
    # no slot question, no "Member ID", no question mark.
    assert interrupt["awaiting_slot"] == "member_id"
    ack = _ack_text(interrupt)
    assert re.search(r"take your time|no rush", ack, re.IGNORECASE)
    assert "member id" not in ack.lower()
    assert "?" not in ack
    # The real slot attempt counter was NOT advanced (no failure counted).
    assert agent.get_slot("member_id").attempt_count == 0


async def test_stalling_via_regex_fallback_when_llm_mislabels():
    agent = _Probe.from_state(_state())
    # LLM returned the default "answered" (mislabel); regex fallback catches it.
    decision = WorkerResult(event_type=EventType.ANSWERED)
    value, interrupt = await agent._collect_slot(
        _state(),
        _config(),
        _messages("hold on, let me grab my card"),
        pre_extracted="",
        decision=decision,
    )
    assert value is None
    assert interrupt["awaiting_slot"] == "member_id"
    assert re.search(r"take your time|no rush", _ack_text(interrupt), re.IGNORECASE)
    assert agent.get_slot("member_id").attempt_count == 0


async def test_repeated_stalls_do_not_escalate_or_burn_attempts():
    agent = _Probe.from_state(_state())
    decision = WorkerResult(event_type=EventType.STALLING)
    for _ in range(MAX_STALLS):
        value, interrupt = await agent._collect_slot(
            _state(), _config(), _messages("just a moment"), pre_extracted="", decision=decision
        )
        assert value is None
        assert interrupt["awaiting_slot"] == "member_id"
    # Real slot never failed → never exhausted → no escalation.
    assert agent.get_slot("member_id").attempt_count == 0
    assert not agent.get_slot("member_id").is_exhausted()


async def test_answer_after_stalling_is_captured_normally():
    agent = _Probe.from_state(_state())
    # First a stall...
    await agent._collect_slot(
        _state(),
        _config(),
        _messages("give me a sec"),
        pre_extracted="",
        decision=WorkerResult(event_type=EventType.STALLING),
    )
    # ...then the member provides the value.
    value, interrupt = await agent._collect_slot(
        _state(),
        _config(),
        _messages("M310188"),
        pre_extracted="M310188",
        decision=WorkerResult(event_type=EventType.ANSWERED),
    )
    assert value == "M310188"
    assert interrupt is None


async def test_stall_cap_falls_through_to_normal_handling():
    """After MAX_STALLS acknowledged stalls, a further stall is treated as a
    normal non-answer (counts a real attempt) so it cannot loop forever."""
    agent = _Probe.from_state(_state())
    decision = WorkerResult(event_type=EventType.STALLING)
    for _ in range(MAX_STALLS):
        await agent._collect_slot(
            _state(), _config(), _messages("one moment"), pre_extracted="", decision=decision
        )
    assert agent.get_slot("member_id").attempt_count == 0
    # The next stall exceeds the cap → falls through → real attempt counted.
    await agent._collect_slot(
        _state(), _config(), _messages("one moment"), pre_extracted="", decision=decision
    )
    assert agent.get_slot("member_id").attempt_count == 1


# ── check_stalling() — shared guard for hand-coded confirmation flows ────────
#
# Several agents build yes/no confirmations (ZIP, fax, email, phone, name...)
# directly in run() instead of going through _collect_slot, so they need to
# call check_stalling() explicitly. These tests cover the helper itself plus
# the exact regression reported in production: provider_search_agent's ZIP
# read-back ("Just to confirm — your ZIP code is 12138?") burned a retry
# attempt and re-asked the question in the same breath when the caller said
# "give me a few seconds" instead of acknowledging and waiting.


def test_check_stalling_acknowledges_without_reasking_or_burning_attempt():
    agent = _Probe.from_state(_state())
    decision = WorkerResult(event_type=EventType.STALLING)
    interrupt = agent.check_stalling(
        _state(), _messages("give me a few seconds"), decision, "zip_confirmed"
    )
    assert interrupt is not None
    assert interrupt["awaiting_slot"] == "zip_confirmed"
    ack = _ack_text(interrupt)
    assert re.search(r"take your time|no rush", ack, re.IGNORECASE)
    assert "zip" not in ack.lower()
    assert "?" not in ack
    assert agent.get_slot("zip_confirmed").attempt_count == 0


def test_check_stalling_returns_none_for_a_real_answer():
    agent = _Probe.from_state(_state())
    decision = WorkerResult(event_type=EventType.ANSWERED)
    assert agent.check_stalling(_state(), _messages("yes that's correct"), decision, "zip_confirmed") is None


async def test_provider_search_zip_confirmed_stall_does_not_reask_or_burn_attempt(monkeypatch):
    """Reproduces the UAT transcript: caller stalls on the ZIP read-back and
    the agent must NOT combine the ack with a repeated confirmation question."""
    from agent.agents.provider_search.agent import ProviderSearchAgent
    from agent.agents.provider_search import agent as provider_search_agent_module

    async def _fake_extract(*args, **kwargs):
        return WorkerResult(event_type=EventType.STALLING)

    monkeypatch.setattr(provider_search_agent_module, "extract_provider_search_decision", _fake_extract)
    monkeypatch.setattr(provider_search_agent_module, "get_extraction_llm", lambda: None)

    state = {
        "member_status_verify": True,
        "provider_type": "Primary Care Physician",
        "zip_code": "12138",
        "zip_code_used": "",
        "awaiting_slot": "zip_confirmed",
        "slot_attempts": {},
        "messages": [
            {"role": "assistant", "content": "Just to confirm — your ZIP code is 12138?"},
            {"role": "user", "content": "give me a few seconds"},
        ],
    }
    agent = ProviderSearchAgent.from_state(state)
    interrupt = await agent.run(state)

    assert interrupt["awaiting_slot"] == "zip_confirmed"
    ack = _ack_text(interrupt)
    assert re.search(r"take your time|no rush", ack, re.IGNORECASE)
    assert "zip" not in ack.lower()
    assert "?" not in ack
    assert agent.get_slot("zip_confirmed").attempt_count == 0
