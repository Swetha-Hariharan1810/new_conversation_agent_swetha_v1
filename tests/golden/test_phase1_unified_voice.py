"""
test_phase1_unified_voice.py — Phase 1: one voice for every turn (UNIFIED_VOICE).

Proves, hermetically:
  * The happy-path ask/transition turns route through the SAME grounded generator
    as retries/clarifies/corrections when UNIFIED_VOICE is on — one voice — and
    stay on the templates when it's off (default). The template is always the
    guaranteed fallback, so a generation failure never drops a turn.
  * The generator is fed the decision as STRUCTURED context (speech_act,
    Collecting, Validated answer this turn, Confirmed, …).
  * Retry latency is unchanged: still exactly one Gemini call.
  * Recovery guardrails hold: a wrong-format retry reply must not open as if the
    value was accepted (no-false-accept), and no generated turn states a value
    outside confirmed_slots ∪ validated_answer (the Phase-4 grounding check,
    added early). The guardrail functions themselves are validated with negatives.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

import tests.golden  # noqa: F401 — ensures src/ is on sys.path
from tests.golden.driver import build_result

pytestmark = pytest.mark.regression


# ── fakes ───────────────────────────────────────────────────────────────────────


class _RecordingLLM:
    """Records every (system, user) message pair and returns a scripted sentence."""

    def __init__(self, content: str = "Whenever you're ready, what's your Member ID?") -> None:
        self.calls: list[list] = []
        self.content = content

    async def ainvoke(self, messages, **_kw):
        self.calls.append(messages)
        return SimpleNamespace(content=self.content)


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


def _first_ask_state():
    # awaiting a DIFFERENT slot than member_id → section 3 (first ask / transition).
    return {
        "awaiting_slot": "first_name",
        "member_id": "",
        "slot_attempts": {},
        "call_intent": "provider_services",
        "active_agent": "verification_agent",
    }


def _last_ai_text(interrupt: dict) -> str:
    msg = interrupt.get("messages")
    return msg.get("content") if isinstance(msg, dict) else ""


# ── happy-path routing: flag off = template, flag on = generator ─────────────────


async def test_first_ask_uses_template_when_flag_off(monkeypatch):
    monkeypatch.delenv("UNIFIED_VOICE", raising=False)
    state = _first_ask_state()
    agent = _probe_agent(state)
    recording = _RecordingLLM()
    import agent.llm.response_generator as rg

    with patch.object(rg, "get_generation_llm", lambda: recording):
        value, interrupt = await agent._collect_slot(
            dict(state),
            _member_id_config(),
            [{"role": "assistant", "content": "Thanks. And your Member ID?"}],
            pre_extracted="",
            decision=build_result({"extracted": {}}),
        )

    assert value is None
    assert interrupt.get("awaiting_slot") == "member_id"
    # No generation call — the template spoke.
    assert recording.calls == []


async def test_first_ask_routed_through_generator_when_on(monkeypatch):
    monkeypatch.setenv("UNIFIED_VOICE", "true")
    state = _first_ask_state()
    agent = _probe_agent(state)
    recording = _RecordingLLM(content="Sure — may I have your Member ID?")
    import agent.llm.response_generator as rg

    with patch.object(rg, "get_generation_llm", lambda: recording):
        value, interrupt = await agent._collect_slot(
            dict(state),
            _member_id_config(),
            [{"role": "assistant", "content": "Thanks. And your Member ID?"}],
            pre_extracted="",
            decision=build_result({"extracted": {}}),
        )

    assert value is None
    assert interrupt.get("awaiting_slot") == "member_id"
    # The generator spoke this turn (one voice), and its text reached the caller.
    assert len(recording.calls) == 1
    assert _last_ai_text(interrupt) == "Sure — may I have your Member ID?"
    # ...with the ask speech act in the STRUCTURED context.
    user_content = recording.calls[0][1].content
    assert "Speech act: ask" in user_content
    assert "Collecting: Member ID" in user_content


async def test_transition_uses_transition_speech_act(monkeypatch):
    monkeypatch.setenv("UNIFIED_VOICE", "true")
    state = _first_ask_state()
    agent = _probe_agent(state)
    recording = _RecordingLLM(content="Thank you — and your Member ID?")
    import agent.llm.response_generator as rg

    with patch.object(rg, "get_generation_llm", lambda: recording):
        # is_transition=True is what SlotPipeline passes after the previous slot
        # was just confirmed.
        await agent._collect_slot(
            dict(state),
            _member_id_config(),
            [{"role": "assistant", "content": "Great, thanks."}],
            pre_extracted="",
            is_transition=True,
            decision=build_result({"extracted": {}}),
        )

    assert len(recording.calls) == 1
    assert "Speech act: transition" in recording.calls[0][1].content


async def test_generator_falls_back_to_template_on_failure(monkeypatch):
    """When generation fails, the exact template string is spoken — no dead turn."""
    monkeypatch.setenv("UNIFIED_VOICE", "true")
    state = _first_ask_state()
    agent = _probe_agent(state)

    class _BoomLLM:
        async def ainvoke(self, *_a, **_k):
            raise RuntimeError("gemini down")

    import agent.llm.response_generator as rg

    captured_fallback = {}
    real = rg.generate_recovery_message

    async def _spy(*args, fallback_text=None, **kwargs):
        captured_fallback["text"] = fallback_text
        return await real(*args, fallback_text=fallback_text, **kwargs)

    with patch.object(rg, "get_generation_llm", lambda: _BoomLLM()):
        with patch.object(rg, "generate_recovery_message", _spy):
            _v, interrupt = await agent._collect_slot(
                dict(state),
                _member_id_config(),
                [{"role": "assistant", "content": "And your Member ID?"}],
                pre_extracted="",
                decision=build_result({"extracted": {}}),
            )

    # A member-id first-ask template was used as the fallback, and it was spoken.
    assert captured_fallback["text"]
    assert _last_ai_text(interrupt) == captured_fallback["text"]
    assert _last_ai_text(interrupt).strip()


# ── retry latency: still exactly one Gemini call ─────────────────────────────────


async def test_retry_is_still_one_gemini_call(monkeypatch):
    monkeypatch.setenv("UNIFIED_VOICE", "true")
    state = {
        "awaiting_slot": "member_id",
        "member_id": "",
        "slot_attempts": {},
        "call_intent": "provider_services",
        "active_agent": "verification_agent",
    }
    agent = _probe_agent(state)
    recording = _RecordingLLM(content="Your Member ID starts with M and six digits — go ahead.")
    import agent.llm.response_generator as rg

    with patch.object(rg, "get_generation_llm", lambda: recording):
        value, interrupt = await agent._collect_slot(
            dict(state),
            _member_id_config(),
            [
                {"role": "assistant", "content": "Can I get your Member ID?"},
                {"role": "user", "content": "banana"},
            ],
            pre_extracted="",
            decision=build_result({"extracted": {}}),
        )

    assert value is None
    assert interrupt.get("awaiting_slot") == "member_id"
    # Exactly one generation call for the retry (latency unchanged).
    assert len(recording.calls) == 1
    assert "Speech act: RETRY" in recording.calls[0][1].content


# ── grounding: the STRUCTURED context only carries grounded values ───────────────


async def test_structured_context_is_grounded(monkeypatch):
    """The context handed to the model contains the validated answer but never a
    prior confirmed *value* (Confirmed lists names only), so the model cannot
    restate an identifier it was never given."""
    from agent.llm.response_generator import build_recovery_context

    ctx_text = build_recovery_context(
        slot_label="date of birth",
        attempt=0,
        speech_act="transition",
        history_text="",
        user_utterance="April third nineteen eighty five",
        confirmed_slots={"member_id": "confirmed", "first_name": "confirmed"},
        validated_answer=None,
        pending_slots=["relationship"],
        parked=None,
        declined=False,
    )
    assert "Confirmed: member_id, first_name" in ctx_text
    # No confirmed VALUE leaks (e.g. a Member ID) — only slot names.
    from agent.responses.grounding import find_ungrounded_values

    # The only concrete-looking token is the spoken date in "Caller just said";
    # that is what the caller said this turn, which is grounded.
    assert find_ungrounded_values(ctx_text, ["April third nineteen eighty five"]) == []


# ── grounding + no-false-accept guardrails (Phase-4 check, added early) ──────────


def test_grounding_check_flags_ungrounded_value():
    from agent.responses.grounding import find_ungrounded_values, is_grounded

    allowed = {"M123456"}  # confirmed_slots ∪ validated_answer for this turn
    grounded = "Thanks — I'll use your Member ID M123456."
    hallucinated = "Thanks — I'll send it to ZIP 94107."

    assert is_grounded(grounded, allowed)
    assert find_ungrounded_values(grounded, allowed) == []
    # A value the model was never given is caught.
    assert "94107" in find_ungrounded_values(hallucinated, allowed)


def test_grounding_ignores_ordinary_text():
    from agent.responses.grounding import find_ungrounded_values

    # No identifier-shaped tokens → nothing to ground.
    assert find_ungrounded_values("Take your time — I'm here when you're ready.", []) == []


def test_no_false_accept_opener_helper():
    from agent.responses.grounding import has_false_accept_opener

    # Forbidden on a wrong-format retry (implies the value was accepted).
    assert has_false_accept_opener("Got it — your Member ID please?")
    assert has_false_accept_opener("Thank you, now your date of birth?")
    assert has_false_accept_opener("Perfect. What's your ZIP?")
    # Legitimate retry phrasings pass.
    assert not has_false_accept_opener("That doesn't look complete — your Member ID starts with M.")
    assert not has_false_accept_opener("Let's try that again — what's your Member ID?")
    # "understanding" must not trip the "understand" opener (word boundary).
    assert not has_false_accept_opener("Understanding this can be tricky — one more time?")


async def test_generated_retry_text_passes_guardrails(monkeypatch):
    """End-to-end: a wrong-format retry's spoken text (as the caller hears it)
    both passes the no-false-accept opener rule and is grounded."""
    monkeypatch.setenv("UNIFIED_VOICE", "true")
    from agent.responses.grounding import find_ungrounded_values, has_false_accept_opener

    state = {
        "awaiting_slot": "member_id",
        "member_id": "",
        "slot_attempts": {},
        "call_intent": "provider_services",
        "active_agent": "verification_agent",
    }
    agent = _probe_agent(state)
    # A compliant retry reply: guides toward the format, opens without a false
    # accept, and states no value that wasn't given this turn.
    reply = "That one looks short — your Member ID should start with M and six digits."
    recording = _RecordingLLM(content=reply)
    import agent.llm.response_generator as rg

    with patch.object(rg, "get_generation_llm", lambda: recording):
        _v, interrupt = await agent._collect_slot(
            dict(state),
            _member_id_config(),
            [
                {"role": "assistant", "content": "Can I get your Member ID?"},
                {"role": "user", "content": "12345"},
            ],
            pre_extracted="",
            decision=build_result({"extracted": {}}),
        )

    spoken = _last_ai_text(interrupt)
    assert spoken == reply  # the guardrail applies to the exact text the caller hears
    assert not has_false_accept_opener(spoken), f"false-accept opener on retry: {spoken!r}"
    # No validated answer this turn (invalid), nothing confirmed → allowed set empty.
    assert find_ungrounded_values(spoken, allowed=[]) == [], f"ungrounded value in retry: {spoken!r}"
