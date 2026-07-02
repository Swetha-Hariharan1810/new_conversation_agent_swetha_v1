"""
test_phase4_hardening.py — Phase 4: latency + naturalness + safety hardening.

Hermetic. Covers:
  * Streaming (STREAM_GENERATION): the generator streams via astream to first
    token; a stream that errors mid-flight falls back to the template — never a
    dead turn; astream (not ainvoke) is used when the flag is on.
  * Grounding guard (belt-and-suspenders): ANY generated turn that states a
    digit-string / email / member-id-shaped value not in
    confirmed_slots ∪ validated_answer is replaced by the template for that act.
  * One-call discipline: the common path makes at most one blocking LLM call.
  * Prioritization order: the structured context orders clauses to match the
    resolver precedence ladder (accept → inline answer → parked → decline → ask).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

import tests.golden  # noqa: F401 — ensures src/ is on sys.path

pytestmark = pytest.mark.regression


# ── fakes ─────────────────────────────────────────────────────────────────────


class _StreamLLM:
    """astream yields chunks; ainvoke is a hard error so we can prove streaming."""

    def __init__(self, chunks, fail_after=None):
        self._chunks = chunks
        self._fail_after = fail_after
        self.astream_calls = 0
        self.ainvoke_calls = 0

    async def ainvoke(self, *_a, **_k):
        self.ainvoke_calls += 1
        return SimpleNamespace(content="".join(self._chunks))

    async def astream(self, *_a, **_k):
        self.astream_calls += 1
        for i, piece in enumerate(self._chunks):
            if self._fail_after is not None and i >= self._fail_after:
                raise RuntimeError("stream broke mid-flight")
            yield SimpleNamespace(content=piece)


class _InvokeLLM:
    def __init__(self, content):
        self.content = content
        self.ainvoke_calls = 0

    async def ainvoke(self, *_a, **_k):
        self.ainvoke_calls += 1
        return SimpleNamespace(content=self.content)


async def _gen(**kwargs):
    from agent.llm.response_generator import generate_recovery_message

    base = {
        "slot_name": "member_id",
        "attempt": 0,
        "guard": "RETRY",
        "last_messages": [{"role": "assistant", "content": "Member ID?"}],
    }
    base.update(kwargs)
    return await generate_recovery_message(**base)


# ── streaming ─────────────────────────────────────────────────────────────────


async def test_streaming_used_when_flag_on(monkeypatch):
    monkeypatch.setenv("STREAM_GENERATION", "true")
    llm = _StreamLLM(["Let's ", "try that ", "again — your Member ID?"])
    with patch("agent.llm.response_generator.get_generation_llm", lambda: llm):
        text = await _gen()
    assert text == "Let's try that again — your Member ID?"
    assert llm.astream_calls == 1
    assert llm.ainvoke_calls == 0  # streamed, not blocking-invoked


async def test_no_streaming_when_flag_off(monkeypatch):
    monkeypatch.delenv("STREAM_GENERATION", raising=False)
    llm = _StreamLLM(["should not be streamed"])
    with patch("agent.llm.response_generator.get_generation_llm", lambda: llm):
        text = await _gen()
    assert llm.ainvoke_calls == 1
    assert llm.astream_calls == 0
    assert text == "should not be streamed"


async def test_stream_error_midflight_falls_back_to_template(monkeypatch):
    monkeypatch.setenv("STREAM_GENERATION", "true")
    llm = _StreamLLM(["Let's ", "try ", "again"], fail_after=2)  # errors after 2 chunks
    with patch("agent.llm.response_generator.get_generation_llm", lambda: llm):
        text = await _gen(slot_name="member_id", fallback_text="Could you share your Member ID?")
    # The mid-flight failure fell back to the exact template — no dead turn.
    assert text == "Could you share your Member ID?"


# ── grounding guard (belt-and-suspenders) on every generated turn ─────────────


async def test_grounding_guard_replaces_leaked_value_with_template(monkeypatch):
    monkeypatch.delenv("STREAM_GENERATION", raising=False)
    # The model invents a ZIP that was never grounded this turn.
    llm = _InvokeLLM("Sure — I'll use ZIP 94107 for that; your Member ID?")
    with patch("agent.llm.response_generator.get_generation_llm", lambda: llm):
        text = await _gen(
            slot_name="member_id",
            fallback_text="Could you share your Member ID?",
            grounded_values=[],  # nothing grounded this turn
        )
    assert "94107" not in text
    assert text == "Could you share your Member ID?"


async def test_grounding_guard_allows_grounded_value(monkeypatch):
    monkeypatch.delenv("STREAM_GENERATION", raising=False)
    # A member id that WAS validated this turn is allowed to be acknowledged.
    llm = _InvokeLLM("Thanks — I've got M123456; and your date of birth?")
    with patch("agent.llm.response_generator.get_generation_llm", lambda: llm):
        text = await _gen(
            slot_name="dob",
            guard="ANSWERED_WITH_FOLLOWUP",
            extracted_value="M123456",
            fallback_text="And your date of birth?",
        )
    assert text == "Thanks — I've got M123456; and your date of birth?"


async def test_grounding_guard_ignores_ordinary_sentences(monkeypatch):
    monkeypatch.delenv("STREAM_GENERATION", raising=False)
    llm = _InvokeLLM("Take your time — what's your Member ID when you're ready?")
    with patch("agent.llm.response_generator.get_generation_llm", lambda: llm):
        text = await _gen(fallback_text="fallback")
    assert text.startswith("Take your time")


# ── one-call discipline ───────────────────────────────────────────────────────


async def test_common_path_one_blocking_call(monkeypatch):
    monkeypatch.delenv("STREAM_GENERATION", raising=False)
    llm = _InvokeLLM("Let's try that again — your Member ID?")
    with patch("agent.llm.response_generator.get_generation_llm", lambda: llm):
        await _gen()
    assert llm.ainvoke_calls == 1  # exactly one blocking LLM call


# ── prioritization order matches the resolver precedence ladder ───────────────


def test_context_orders_clauses_by_precedence():
    from agent.llm.response_generator import build_recovery_context

    ctx = build_recovery_context(
        slot_label="date of birth",
        attempt=0,
        speech_act="multi_intent",
        history_text="",
        user_utterance="…",
        confirmed_slots={"member_id": "confirmed"},
        validated_answer="01/01/1990",
        pending_slots=None,
        parked=["benefits_agent"],
        declined=True,
        answered_inline=["Your card is active."],
        next_ask="relationship",
        correction_field=None,
    )
    # accept → inline answer → parked → decline → next ask
    i_accept = ctx.index("Validated answer this turn")
    i_inline = ctx.index("Answer to include")
    i_parked = ctx.index("Parked")
    i_declined = ctx.index("Declined")
    i_next = ctx.index("Next, ask for")
    assert i_accept < i_inline < i_parked < i_declined < i_next
