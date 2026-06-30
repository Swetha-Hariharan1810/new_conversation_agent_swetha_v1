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
    "phone": "phone number",
    "member_id": "Member ID",
    "dob": "date of birth",
    "first_name": "first name",
    "last_name": "last name",
    "relationship": "plan relationship",
    "provider_type": "provider type",
    "delivery_method": "delivery method",
    "reference_number": "reference number",
}

# What to ask for when re-collecting a corrected field at its owner.
FIELD_ASK_LABELS: dict[str, str] = {
    "zip_code": "current 5-digit ZIP code",
    "fax": "correct fax number",
    "email": "correct email address",
    "phone_number": "correct phone number",
    "phone": "correct phone number",
    "member_id": "correct Member ID",
    "dob": "correct date of birth",
    "first_name": "correct first name",
    "last_name": "correct last name",
    "relationship": "relationship to the plan",
    "provider_type": "type of provider you're looking for",
    "reference_number": "correct reference number",
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
    "Okay — let's fix your {field_label}. What's your {ask_label}?",
]

_RE_ASK = [
    "Sorry, I didn't catch that — could you share your {slot_label}?",
    "Let's try that again — what's your {slot_label}?",
    "Could you repeat your {slot_label} for me?",
]

_CLARIFY = [
    "I want to make sure I get this right — could you say your {slot_label} once more?",
    "Just to confirm, could you repeat your {slot_label}?",
    "Sorry, one more time — what's your {slot_label}?",
]

_UNSUPPORTED_DECLINE = [
    "I'm not able to help with that one here, but I can keep going with what we were doing.",
    "That's outside what I can take care of on this call, but let's continue.",
    "I can't assist with that part here, but we can carry on with what we were doing.",
]

_OPEN_REDIRECT = [
    "I'm not able to help with that one here. Is there anything else I can help you with?",
    "That's not something I can do on this call — what else can I help you with?",
    "I can't take care of that one here. Is there something else I can help with?",
]

# Multi-intent acknowledgement: confirm we heard the parked request(s) and that
# they will be handled, optionally noting what is being rebuilt/finished first.
_MULTI_ACK = [
    "Got it — I'll help with {parked} as well. Let's keep going and I'll come right back to it.",
    "Sure — I've noted {parked} too, and I'll take care of it in a moment.",
    "Absolutely — I'll get to {parked} as well; let me finish this first.",
]

_MULTI_ACK_WITH_REBUILD = [
    "Got it — I'll take care of {parked} too. First, let me {rebuilding}.",
    "Sure — I've noted {parked}. Before that, let me {rebuilding}.",
    "Absolutely — I'll handle {parked} as well. First, let me {rebuilding}.",
]

# Human-readable labels per owning agent, for the multi-intent acknowledgement.
OWNER_LABELS: dict[str, str] = {
    "verification_agent": "verifying your account",
    "provider_search_agent": "your provider search",
    "delivery_management_agent": "your delivery details",
    "benefits_agent": "your benefits question",
    "care_wellness_agent": "the wellness program",
    "claim_adjustment_agent": "your claim",
    "records_coordination_agent": "your records",
    "notification_setup_agent": "your notification preferences",
    "follow_up_agent": "your other question",
}


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


def render_open_redirect(*, attempt: int = 0) -> str:
    return _rotate(_OPEN_REDIRECT, attempt)


def owner_label(owner: str) -> str:
    return OWNER_LABELS.get(owner, "that")


def _join_labels(labels: list[str]) -> str:
    """Order-preserving dedup + natural-language join."""
    seen: list[str] = []
    for label in labels:
        if label not in seen:
            seen.append(label)
    if not seen:
        return "that"
    if len(seen) == 1:
        return seen[0]
    if len(seen) == 2:
        return f"{seen[0]} and {seen[1]}"
    return ", ".join(seen[:-1]) + f", and {seen[-1]}"


def render_multi_intent_ack(
    parked_owners: list[str],
    *,
    rebuilding: str | None = None,
    attempt: int = 0,
) -> str:
    """Acknowledge parked secondary intent(s), keyed on the resolver's structured
    outcome. ``parked_owners`` are resolved agent names; ``rebuilding`` is an
    optional human phrase for the dependent action being completed first."""
    parked = _join_labels([owner_label(o) for o in parked_owners])
    if rebuilding:
        return _rotate(_MULTI_ACK_WITH_REBUILD, attempt).format(parked=parked, rebuilding=rebuilding)
    return _rotate(_MULTI_ACK, attempt).format(parked=parked)
