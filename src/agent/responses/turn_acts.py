"""
turn_acts.py — Closed-set response templates for resolver speech-acts (Phase 3B).

The resolver selects exactly one speech-act label from a closed set; this module
renders it. There is NO generative surface here: every sentence is a fixed
template filled ONLY with resolver-validated values (a confirmed slot value, a
known field label). Phrasings are rotated deterministically by attempt count
(``pool[attempt % len(pool)]``) — the same rotation idea as ``responses/`` pools,
but deterministic for testability rather than random.

Covered acts: re_ask, clarify, correction_ack, unsupported_decline.
(multi_intent_ack / open_redirect arrive with the rest of Phase 3C.)
"""

from __future__ import annotations

# Human-readable labels for fields the resolver may route a correction to.
FIELD_LABELS: dict[str, str] = {
    "zip_code": "ZIP code",
    "fax": "fax number",
    "email": "email address",
    "phone_number": "phone number",
}

# What to ask for when re-collecting a corrected field at its owner.
FIELD_ASK_LABELS: dict[str, str] = {
    "zip_code": "current 5-digit ZIP code",
    "fax": "correct fax number",
    "email": "correct email address",
    "phone_number": "correct phone number",
}

# ── Template pools (no free text — placeholders are resolver-validated) ─────────

_CORRECTION_ACK_WITH_SLOT = [
    "Got it — I'll send it by {slot_value}. First, let's update your {field_label} "
    "so the list is for the right area. What's your {ask_label}?",
    "Sure, {slot_value} it is. Before I send it, let's correct your {field_label} "
    "so the list matches your area — what's your {ask_label}?",
    "Okay, I'll use {slot_value}. Let me get your {field_label} updated first so the "
    "list is accurate. What's your {ask_label}?",
]

_CORRECTION_ACK_FIX_ONLY = [
    "Got it — let's update your {field_label} first. What's your {ask_label}?",
    "Sure, I'll get your {field_label} corrected. What's your {ask_label}?",
]

_RE_ASK = [
    "Sorry, I didn't catch that — could you share your {slot_label}?",
    "Let's try that again — what's your {slot_label}?",
    "Could you repeat your {slot_label} for me?",
]

_CLARIFY = [
    "I want to make sure I get this right — could you say your {slot_label} once more?",
    "Just to confirm, could you repeat your {slot_label}?",
]

_UNSUPPORTED_DECLINE = [
    "I'm not able to help with that one here, but I can keep going with what we were doing.",
    "That's outside what I can take care of on this call, but let's continue.",
]


def _rotate(pool: list[str], attempt: int) -> str:
    """Deterministic rotation by attempt count."""
    return pool[max(0, attempt) % len(pool)]


def field_label(field: str) -> str:
    return FIELD_LABELS.get(field, field.replace("_", " "))


def field_ask_label(field: str) -> str:
    return FIELD_ASK_LABELS.get(field, f"correct {field.replace('_', ' ')}")


def render_correction_ack(
    *,
    field: str,
    attempt: int = 0,
    slot_value: str | None = None,
) -> str:
    """Acknowledge a correction (and, when present, the slot answer given in the
    same turn) and ask for the corrected value. Values must be resolver-validated."""
    flabel = field_label(field)
    ask = field_ask_label(field)
    if slot_value:
        return _rotate(_CORRECTION_ACK_WITH_SLOT, attempt).format(
            slot_value=slot_value, field_label=flabel, ask_label=ask
        )
    return _rotate(_CORRECTION_ACK_FIX_ONLY, attempt).format(field_label=flabel, ask_label=ask)


def render_re_ask(*, slot_label: str, attempt: int = 0) -> str:
    return _rotate(_RE_ASK, attempt).format(slot_label=slot_label)


def render_clarify(*, slot_label: str, attempt: int = 0) -> str:
    return _rotate(_CLARIFY, attempt).format(slot_label=slot_label)


def render_unsupported_decline(*, attempt: int = 0) -> str:
    return _rotate(_UNSUPPORTED_DECLINE, attempt)
