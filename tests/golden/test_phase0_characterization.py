"""
test_phase0_characterization.py — Phase 0 characterization tests.

These lock TODAY's behavior at two seams the rebuild will change, so the change
is measurable and the flip is explicit. They are GREEN now and describe current
behavior (marked with PHASE-FLIP where a later phase will invert the assertion).

Both drive the shared per-turn collector (``_collect_slot``) directly — the
"relevant agent" component — with the LLM-1 decision supplied verbatim, so there
is no graph, no network, and no dependence on which agent owns the slot. This is
the same probe style as ``test_golden_baseline.test_mid_verification_correction``.

Cases:
  1. answer + follow-up  ("member id is M123456, and is my card still active?")
     → the slot is confirmed and the follow-up is *acknowledged-only* (routed
       through LLM-2 generation on the ANSWERED_WITH_FOLLOWUP path), never
       answered with card data and never parked/queued as an actionable intent.
  2. a genuine retry turn
     → recovery goes through LLM-2 (Gemini) with guard "RETRY", using the
       ``generation/recovery.md`` system prompt.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

import tests.golden  # noqa: F401 — ensures src/ is on sys.path
from tests.golden.driver import build_result

pytestmark = pytest.mark.regression


def _member_id_config():
    from agent.core.slot_manager import _InternalSlotConfig
    from agent.slots.normalizers import normalize_member_id
    from agent.slots.types import SlotType
    from agent.slots.validators import validate_member_id

    return _InternalSlotConfig(
        slot_name="member_id",
        prompt="",
        normalizer=normalize_member_id,
        validator=validate_member_id,
        slot_type=SlotType.MEMBER_ID,
    )


def _probe_agent(state: dict):
    from agent.core.agent import BaseAgent

    class _ProbeAgent(BaseAgent):
        AGENT_NAME = "verification_agent"

        async def run(self, _state):  # pragma: no cover - not used
            return {}

    return _ProbeAgent.from_state(state)


# ──────────────────────────────────────────────────────────────────────────────
# 1. answer + follow-up → acknowledged-only (never actually answered)
# ──────────────────────────────────────────────────────────────────────────────


async def test_answer_with_followup_is_acknowledged_only():
    """The member answers the slot AND asks a side question in one breath. Today
    the answer is captured and the follow-up is *acknowledged only* — it is routed
    through LLM-2 generation on the ANSWERED_WITH_FOLLOWUP path (which per
    recovery.md acknowledges but does not answer it), and it is never parked or
    routed to an agent that could answer it.

    PHASE-FLIP: a later phase (PARK_ANSWERABLE) will park/answer the follow-up;
    this assertion flips from "acknowledged-only" to "parked/answered".
    """
    from agent.llm.schema import EventType

    state = {
        "awaiting_slot": "member_id",
        "member_id": "",
        "slot_attempts": {},
        "call_intent": "provider_services",
        "active_agent": "verification_agent",
    }
    agent = _probe_agent(state)
    config = _member_id_config()
    messages = [
        {"role": "assistant", "content": "Can I get your Member ID?"},
        {"role": "user", "content": "member id is M123456, and is my card still active?"},
    ]
    decision = build_result(
        {
            "extracted": {"member_id": "M123456"},
            "event_type": EventType.ANSWERED_WITH_FOLLOWUP.value,
        }
    )

    # Spy on the generation seam so we can assert the follow-up was handled via
    # generation (acknowledged), with the ANSWERED_WITH_FOLLOWUP guard.
    import agent.core.slot_manager as sm

    seen_guards: list[str] = []

    async def _fake_gen(*_a, guard: str = "RETRY", **_k):
        seen_guards.append(guard)
        return "[[acknowledged-only-followup]]"

    with patch.object(sm, "generate_recovery_message", _fake_gen, create=True):
        # slot_manager imports generate_recovery_message lazily inside the method,
        # so patch it at its source module too.
        import agent.llm.response_generator as rg

        with patch.object(rg, "generate_recovery_message", _fake_gen):
            value, interrupt = await agent._collect_slot(
                dict(state),
                config,
                messages,
                pre_extracted="M123456",
                decision=decision,
            )

    # The slot answer WAS captured.
    assert value == "M123456"

    # The follow-up was acknowledged only — via the ANSWERED_WITH_FOLLOWUP
    # generation path, not answered with real data.
    assert "ANSWERED_WITH_FOLLOWUP" in seen_guards, (
        f"expected the follow-up to be acknowledged via generation, guards seen: {seen_guards!r}"
    )

    # An interrupt was returned (the acknowledgement), and the slot is no longer
    # awaited (confirmed, not re-asked).
    assert interrupt is not None
    assert interrupt.get("awaiting_slot") == ""

    # PHASE-FLIP: the follow-up is NOT parked as an actionable intent — it is
    # dropped/acknowledged-only today (a later phase enqueues it for draining).
    assert not interrupt.get("intent_queue"), (
        f"F-followup regressed (good!) — follow-up was parked: {interrupt.get('intent_queue')!r}"
    )
    # And control stays in the current agent (the follow-up is not routed to
    # another agent that could actually answer the card question). ``next_node``
    # pointing back at the active agent is just "continue here", not a handoff.
    assert interrupt.get("next_node") in (None, "", "verification_agent"), (
        f"F-followup regressed (good!) — follow-up was routed to answer it: "
        f"{interrupt.get('next_node')!r}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 2. retry turn → LLM-2 (Gemini) recovery.md generation
# ──────────────────────────────────────────────────────────────────────────────


class _RecordingLLM:
    """Records the messages passed to ``ainvoke`` so we can inspect the system
    prompt (proving recovery.md is used) and returns a fixed spoken sentence."""

    def __init__(self) -> None:
        self.calls: list[list] = []

    async def ainvoke(self, messages, **_kw):
        self.calls.append(messages)
        return SimpleNamespace(content="Let's try that again — what's your Member ID?")


async def test_retry_turn_goes_through_gemini_recovery():
    """A genuine non-answer on the awaited slot recovers through LLM-2 (Gemini),
    with guard "RETRY" and the ``generation/recovery.md`` system prompt.

    PHASE-FLIP: the response-unification phase (UNIFIED_VOICE) will move this off
    free-form generation onto the closed-set templates; this assertion flips from
    "generation via recovery.md" to "templated".
    """
    from agent.llm.schema import EventType

    state = {
        "awaiting_slot": "member_id",
        "member_id": "",
        "slot_attempts": {},
        "call_intent": "provider_services",
        "active_agent": "verification_agent",
    }
    agent = _probe_agent(state)
    config = _member_id_config()
    messages = [
        {"role": "assistant", "content": "Can I get your Member ID?"},
        # A plain non-answer: not a stall, not a cannot-provide, no secondary.
        {"role": "user", "content": "banana"},
    ]
    # No extraction this turn (LLM-1 captured nothing), default ANSWERED event.
    decision = build_result({"extracted": {}, "event_type": EventType.ANSWERED.value})

    recording = _RecordingLLM()

    import agent.llm.response_generator as rg

    with patch.object(rg, "get_generation_llm", lambda: recording):
        value, interrupt = await agent._collect_slot(
            dict(state),
            config,
            messages,
            pre_extracted="",
            decision=decision,
        )

    # Not confirmed — this is a retry.
    assert value is None
    assert interrupt is not None
    assert interrupt.get("awaiting_slot") == "member_id"

    # The recovery went through LLM-2 (Gemini getter was invoked exactly once).
    assert len(recording.calls) == 1, (
        f"expected one LLM-2 recovery call, got {len(recording.calls)}"
    )

    # ...using the generation/recovery.md system prompt. Assert against the built
    # prompt itself (robust to wording changes across phases) — it is the recovery
    # prompt concatenated onto the global voice.
    from agent.utils import build_generation_prompt, read_prompt

    system_msg = recording.calls[0][0]
    system_text = getattr(system_msg, "content", "")
    assert system_text == build_generation_prompt(), (
        "expected the generation/recovery.md system prompt to drive LLM-2 recovery"
    )
    assert read_prompt("generation/recovery.md") in system_text


async def test_retry_falls_back_to_static_when_generation_unavailable():
    """Companion characterization: if LLM-2 raises, the retry still produces a
    spoken message (the static fallback) and keeps awaiting the slot — the current
    resilience contract we must not regress."""
    from agent.llm.schema import EventType

    state = {
        "awaiting_slot": "member_id",
        "member_id": "",
        "slot_attempts": {},
        "call_intent": "provider_services",
        "active_agent": "verification_agent",
    }
    agent = _probe_agent(state)
    config = _member_id_config()
    messages = [
        {"role": "assistant", "content": "Can I get your Member ID?"},
        {"role": "user", "content": "banana"},
    ]
    decision = build_result({"extracted": {}, "event_type": EventType.ANSWERED.value})

    class _BoomLLM:
        async def ainvoke(self, *_a, **_k):
            raise RuntimeError("gemini down")

    import agent.llm.response_generator as rg

    with patch.object(rg, "get_generation_llm", lambda: _BoomLLM()):
        value, interrupt = await agent._collect_slot(
            dict(state), config, messages, pre_extracted="", decision=decision
        )

    assert value is None
    assert interrupt is not None
    assert interrupt.get("awaiting_slot") == "member_id"
    # A spoken message was still produced (static RETRY fallback mentions the slot).
    content = interrupt.get("messages")
    text = content.get("content") if isinstance(content, dict) else ""
    assert text.strip(), "expected a static fallback recovery message"
