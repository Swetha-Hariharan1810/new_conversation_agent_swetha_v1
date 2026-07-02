"""
test_phase1_voice_safety.py — Phase 1 (Bug 1, generalized): the voice-safety layer.

Hermetic. Covers the two invariants of the fix:
  * An internal instruction can never reach the caller — instruction-style
    guidance travels on ``generator_directive`` (its own ``Guidance:`` context
    line) and is NEVER interpolated into a spoken fallback template; a
    directive-style label from an un-migrated call site is detected and blocked
    (metric="directive_label_blocked").
  * The grounding guard can never contradict what the prompt legitimately asked
    the model to say — ``turn_grounding_allowlist`` grounds every value the
    agent deliberately reads back (ZIP/phone/email on file, spaced and
    contiguous digit variants), and a debug assertion fails fast when a
    directive asks the model to speak a value that isn't grounded.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

import tests.golden  # noqa: F401 — ensures src/ is on sys.path

pytestmark = pytest.mark.regression


class _InvokeLLM:
    def __init__(self, content):
        self.content = content
        self.calls = []

    async def ainvoke(self, messages, *_a, **_k):
        self.calls.append(messages)
        return SimpleNamespace(content=self.content)


class _BoomLLM:
    async def ainvoke(self, *_a, **_k):
        raise RuntimeError("gemini down")


async def _gen(**kwargs):
    from agent.llm.response_generator import generate_recovery_message

    base = {
        "slot_name": "member_id",
        "attempt": 1,
        "guard": "RETRY",
        "last_messages": [{"role": "assistant", "content": "Member ID?"}],
    }
    base.update(kwargs)
    return await generate_recovery_message(**base)


# ── the directive never reaches the caller ────────────────────────────────────


async def test_directive_never_interpolated_into_fallback(monkeypatch):
    """Generation fails → the spoken fallback carries the noun-phrase label,
    never one word of the instruction-style directive."""
    monkeypatch.delenv("STREAM_GENERATION", raising=False)
    directive = (
        "Ask whether the ZIP code 1 2 1 3 9 on file is correct (yes or no) — "
        "if they say their address changed, ask for their current ZIP."
    )
    with patch("agent.llm.response_generator.get_generation_llm", lambda: _BoomLLM()):
        text = await _gen(
            slot_name="zip_confirmed",
            slot_label_override="ZIP code confirmation",
            generator_directive=directive,
            grounded_values=["12139"],
        )
    assert "ZIP code confirmation" in text
    assert "(yes or no)" not in text
    assert "if they" not in text
    assert "ask for their current" not in text


async def test_directive_style_label_blocked_from_fallback(monkeypatch):
    """Belt-and-suspenders: an un-migrated call site still passing an
    instruction-style label cannot leak it through a fallback template."""
    monkeypatch.delenv("STREAM_GENERATION", raising=False)
    bad_label = (
        "whether the ZIP code on file is correct (yes or no) — if they say it changed, ask for their ZIP"
    )
    import agent.llm.response_generator as rg

    with patch("agent.llm.response_generator.get_generation_llm", lambda: _BoomLLM()):
        with patch.object(rg.logger, "warning", wraps=rg.logger.warning) as warn:
            text = await _gen(slot_name="zip_confirmed", slot_label_override=bad_label)
    assert bad_label not in text
    assert "zip confirmed" in text  # plain slot_name.replace("_", " ") label
    assert any(
        call.kwargs.get("extra", {}).get("metric") == "directive_label_blocked"
        for call in warn.call_args_list
    ), "expected metric=directive_label_blocked warning"


def test_default_slot_labels_are_noun_phrases():
    """Every static label must survive the defensive fallback check unchanged —
    short, and free of instruction-style markers."""
    from agent.llm.response_generator import _DIRECTIVE_MARKERS, _SLOT_LABELS

    for slot, label in _SLOT_LABELS.items():
        assert len(label) <= 60, f"{slot}: label too long: {label!r}"
        lowered = label.lower()
        for marker in _DIRECTIVE_MARKERS:
            assert marker not in lowered, f"{slot}: directive marker {marker!r} in label {label!r}"


async def test_guidance_rendered_as_own_context_line(monkeypatch):
    monkeypatch.delenv("STREAM_GENERATION", raising=False)
    llm = _InvokeLLM("Is the ZIP code 1 2 1 3 9 we have on file still correct?")
    with patch("agent.llm.response_generator.get_generation_llm", lambda: llm):
        await _gen(
            slot_name="zip_confirmed",
            slot_label_override="ZIP code confirmation",
            generator_directive="Ask whether the ZIP code 1 2 1 3 9 on file is correct (yes or no).",
            grounded_values=["12139"],
        )
    context = llm.calls[0][1].content
    assert "Collecting: ZIP code confirmation" in context
    assert "Guidance: Ask whether the ZIP code 1 2 1 3 9 on file is correct" in context


# ── the guard never contradicts a legitimate read-back ────────────────────────


async def test_spoken_readback_passes_guard_when_grounded(monkeypatch):
    """The directive asks the model to read the ZIP back spaced; the allow-list
    grounds it, so the generated read-back is spoken, not vetoed."""
    monkeypatch.delenv("STREAM_GENERATION", raising=False)
    from agent.responses.grounding import turn_grounding_allowlist

    state = {"awaiting_slot": "zip_confirmed", "zip_code": "12139"}
    reply = "Just checking — is 1 2 1 3 9 still the right ZIP code for you, yes or no?"
    llm = _InvokeLLM(reply)
    with patch("agent.llm.response_generator.get_generation_llm", lambda: llm):
        text = await _gen(
            slot_name="zip_confirmed",
            slot_label_override="ZIP code confirmation",
            generator_directive="Ask whether the ZIP code 1 2 1 3 9 on file is correct (yes or no).",
            grounded_values=turn_grounding_allowlist(state, None, extracted_value=None, answered_inline=None),
            fallback_text="Is the ZIP we have on file still correct?",
        )
    assert text == reply  # the read-back was NOT replaced by the template


async def test_directive_with_ungrounded_value_fails_fast(monkeypatch):
    """Debug assertion: a directive asking the model to speak a value that is
    not in grounded_values is a call-site bug and must fail loudly in tests."""
    monkeypatch.delenv("STREAM_GENERATION", raising=False)
    with patch("agent.llm.response_generator.get_generation_llm", lambda: _InvokeLLM("ok")):
        with pytest.raises(AssertionError, match="ungrounded value"):
            await _gen(
                slot_name="zip_confirmed",
                generator_directive="Ask whether the ZIP code 94107 on file is correct.",
                grounded_values=[],  # bug: the ZIP the directive reads back is missing
            )


def test_spaced_digit_readback_normalizes_against_allowed():
    """ "1 2 1 3 9" ≡ "12139": a spaced spoken read-back matches the contiguous
    value on file, and is still caught when NOT grounded."""
    from agent.responses.grounding import find_ungrounded_values

    assert find_ungrounded_values("Is 1 2 1 3 9 still your ZIP?", ["12139"]) == []
    assert find_ungrounded_values("Is 1 2 1 3 9 still your ZIP?", []) == ["1 2 1 3 9"]
    # Spaced phone read-back too.
    assert find_ungrounded_values("Is 5 5 5 1 2 3 4 5 6 7 right?", ["555-123-4567"]) == []


def test_turn_grounding_allowlist_completeness():
    from agent.responses.grounding import turn_grounding_allowlist

    ctx = SimpleNamespace(caller_first_name="Emily")

    # ZIP on file during a zip-confirmation act — spaced and contiguous variants.
    state = {"awaiting_slot": "zip_confirmed", "zip_code": "12139"}
    allowed = turn_grounding_allowlist(state, ctx, extracted_value="fax", answered_inline=["by email"])
    assert "fax" in allowed
    assert "by email" in allowed
    assert "Emily" in allowed
    assert "12139" in allowed
    assert "1 2 1 3 9" in allowed

    # Phone on file during phone_confirmed — formatted, contiguous, and spaced.
    state = {"awaiting_slot": "phone_confirmed", "phone_number": "555-123-4567"}
    allowed = turn_grounding_allowlist(state, ctx, extracted_value=None, answered_inline=None)
    assert "555-123-4567" in allowed
    assert "5551234567" in allowed
    assert " ".join("5551234567") in allowed

    # Email being confirmed — pending value wins alongside the one on file.
    state = {"awaiting_slot": "email_confirmed", "pending_email": "a@b.com", "email": "old@b.com"}
    allowed = turn_grounding_allowlist(state, ctx, extracted_value=None, answered_inline=None)
    assert "a@b.com" in allowed
    assert "old@b.com" in allowed

    # Explicit slot_name overrides state's awaiting_slot; explicit readbacks add in.
    allowed = turn_grounding_allowlist(
        {"zip_code": "94107"},
        None,
        extracted_value=None,
        answered_inline=None,
        slot_name="zip_confirmed",
        readback_values=["9 4 1 0 7"],
    )
    assert "94107" in allowed
    assert "9 4 1 0 7" in allowed


# ── migrated call sites: correction ack grounds what it reads back ────────────


async def test_correction_ack_name_readback_is_grounded_and_label_safe(monkeypatch):
    """The name-correction ack passes the instruction on the directive (never the
    label) and grounds the corrected value it asks the model to read back."""
    monkeypatch.delenv("STREAM_GENERATION", raising=False)
    from agent.conversation.context import ConversationContext
    from agent.core.agent import BaseAgent

    class _Probe(BaseAgent):
        AGENT_NAME = "verification_agent"

        async def run(self, state):  # pragma: no cover - not exercised
            return {}

    agent = _Probe.from_state({"awaiting_slot": "dob"})
    ctx = ConversationContext()
    decision = SimpleNamespace(corrections={"first_name": "Emilia"})
    llm = _InvokeLLM("Of course — Emilia it is; now, what's your date of birth?")
    with patch("agent.llm.response_generator.get_generation_llm", lambda: llm):
        text = await agent._generate_correction_ack(
            {"awaiting_slot": "dob"},
            ["first_name"],
            "dob",
            ctx,
            [{"role": "user", "content": "it's Emilia, not Emily"}],
            decision=decision,
        )
    # The read-back was grounded, so the generated ack survived the guard.
    assert text == "Of course — Emilia it is; now, what's your date of birth?"
    context = llm.calls[0][1].content
    # Instruction went to Guidance; Collecting stayed a noun phrase.
    assert "Guidance: The caller corrected their first name to 'Emilia'" in context
    assert "Collecting: date of birth" in context
