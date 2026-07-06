"""
Single-ask invariant tests (Phase 2, fixes Bug A / the Emily-Carter double-ask).

Covers:
  - sanitize_generated: confirmed-slot re-asks stripped, trailing/next-slot
    questions dropped when Python appends the ask, fallback on emptied text
  - CORRECTION_ACK: pure corrections no longer routed to FOLLOWUP_DECLINE,
    prompt file registered, fallback entry present
  - one-ask invariant on the combined message out of _handle_answered_followup
"""

from types import SimpleNamespace

import pytest

from agent.conversation.context import ConversationContext
from agent.core.slot_manager import SlotManagerMixin, _InternalSlotConfig
from agent.llm.response_generator import _FALLBACKS, sanitize_generated
from agent.utils import build_generation_prompt

# ── sanitize_generated ───────────────────────────────────────────────────────


def test_confirmed_slot_reask_stripped():
    text = "Thanks Emily, I've updated your name. Could you confirm your Member ID number again?"
    out = sanitize_generated(
        text,
        guard="CORRECTION_ACK",
        confirmed_labels=("member_id", "first_name"),
        will_append_ask=False,
        fallback_slot_label="first name",
    )
    assert out == "Thanks Emily, I've updated your name."


def test_confirmed_slot_synonym_match_is_case_insensitive():
    text = "All set. What was your DATE OF BIRTH again?"
    out = sanitize_generated(text, guard="FOLLOWUP_PARK", confirmed_labels=("dob",))
    assert out == "All set."


def test_confirmed_slot_mention_without_question_survives():
    text = "I've noted your date of birth, thank you."
    out = sanitize_generated(text, guard="FOLLOWUP_PARK", confirmed_labels=("dob",))
    assert out == text


def test_trailing_question_dropped_when_ask_will_be_appended():
    text = "Got it, thanks. Shall we keep going?"
    out = sanitize_generated(text, guard="FOLLOWUP_PARK", will_append_ask=True, next_slot_label="zip_code")
    assert out == "Got it, thanks."


def test_next_slot_mention_dropped_when_ask_will_be_appended():
    text = "Got it, thanks. Next I'll need your five-digit ZIP code."
    out = sanitize_generated(text, guard="FOLLOWUP_PARK", will_append_ask=True, next_slot_label="zip_code")
    assert out == "Got it, thanks."


def test_trailing_question_kept_when_no_ask_appended():
    # RETRY re-asks the awaiting (unconfirmed) slot itself — must survive.
    text = "No problem — could you read me your Member ID once more?"
    out = sanitize_generated(text, guard="RETRY", confirmed_labels=("first_name", "dob"))
    assert out == text


def test_fallback_on_emptied_text():
    text = "Could you confirm your Member ID again?"
    out = sanitize_generated(text, guard="FOLLOWUP_DECLINE", confirmed_labels=("member_id",))
    assert out == _FALLBACKS["FOLLOWUP_DECLINE"]


def test_fallback_formats_slot_label():
    out = sanitize_generated(
        "What's your date of birth again?",
        guard="CORRECTION_ACK",
        confirmed_labels=("dob",),
        fallback_slot_label="first name",
    )
    assert out == "Got it — I've updated your first name."


def test_one_ask_invariant_on_combined_message():
    generated = "Perfect, I've noted that — should I also update your email address? What else can I do?"
    ask = "Could I have your five-digit ZIP code?"
    sanitized = sanitize_generated(
        generated,
        guard="FOLLOWUP_ANSWER",
        confirmed_labels=("email",),
        will_append_ask=True,
        next_slot_label="zip_code",
        fallback_slot_label="email",
    )
    combined = sanitized.rstrip() + " " + ask
    assert combined.count("?") == 1
    assert combined.rstrip().endswith(ask)


# ── CORRECTION_ACK registration ──────────────────────────────────────────────


def test_correction_ack_fallback_registered():
    assert _FALLBACKS["CORRECTION_ACK"] == "Got it — I've updated your {slot_label}."


def test_correction_ack_prompt_file_registered():
    prompt = build_generation_prompt("CORRECTION_ACK")
    assert "Event: CORRECTION_ACK" in prompt
    assert "Never ask any question" in prompt


# ── CORRECTION_ACK routing + one-ask invariant end to end (no live LLM) ─────


class _FakeAgent(SlotManagerMixin):
    AGENT_NAME = "test_agent"

    def __init__(self):
        self._slots = {}
        self._newly_confirmed = set()
        self._pending_ambiguous_resets = set()

    def ask_member(self, state, message):
        return {"response": message}


def _cfg(slot_name, prompt=""):
    return SimpleNamespace(
        slot_name=slot_name,
        prompt=prompt,
        normalizer=lambda v: str(v).strip().title(),
        validator=lambda v: SimpleNamespace(valid=True),
        slot_type=None,
    )


@pytest.fixture
def fake_generation(monkeypatch):
    """Patch LLM 2 with a canned double-ask response; capture the guard used."""
    captured = {}

    async def fake_generate(**kwargs):
        captured.update(kwargs)
        return (
            "Thanks Emily — I've updated your first name. "
            "Could you confirm your Member ID number again? And what's your ZIP code?"
        )

    import agent.llm.response_generator as rg

    monkeypatch.setattr(rg, "generate_recovery_message", fake_generate)
    return captured


async def test_pure_correction_routes_to_correction_ack_with_single_ask(fake_generation):
    agent = _FakeAgent()
    ctx = ConversationContext(
        confirmed_slots=["first_name", "last_name", "member_id"],
        caller_first_name="Emma",
    )
    decision = SimpleNamespace(
        extracted={},
        corrections={"first_name": "emily"},
        update_target="",
        followup_query="",
        followup_disposition=None,
    )
    slot_configs = {
        "first_name": _cfg("first_name"),
        "zip_code": _cfg("zip_code", prompt="Could I have your five-digit ZIP code?"),
    }
    collected = {}

    normalized, interrupt = await agent._handle_answered_followup(
        {},
        _InternalSlotConfig(
            slot_name="dob",
            prompt="What is your date of birth?",
            normalizer=str,
            validator=lambda v: SimpleNamespace(valid=True),
        ),
        [{"role": "user", "content": "March first 1990 — actually my name is Emily"}],
        "03/01/1990",
        ctx,
        decision=decision,
        pending_slots=["dob", "zip_code"],
        slot_configs=slot_configs,
        collected=collected,
    )

    assert normalized == "03/01/1990"
    # Pure correction must NOT be treated as a declined follow-up.
    assert fake_generation["guard"] == "CORRECTION_ACK"

    message = interrupt["response"]
    # Confirmed-slot re-ask and competing next-slot question stripped; the
    # appended static ask is the one and only ask in the combined utterance.
    assert "Member ID" not in message
    assert message.count("?") == 1
    assert message.rstrip().endswith("Could I have your five-digit ZIP code?")
    assert interrupt["awaiting_slot"] == "zip_code"
