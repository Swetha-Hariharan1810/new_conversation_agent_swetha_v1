"""
response_generator.py — LLM 2: natural recovery response generation.

Called only on recovery turns. Uses Gemini (get_routing_llm).
Input is intentionally minimal. Output is one spoken sentence.
"""

from __future__ import annotations

from agent.llm.config import get_generation_llm
from agent.logger import get_logger
from agent.utils import build_generation_prompt, build_history

logger = get_logger(__name__)

_SLOT_LABELS: dict[str, str] = {
    "first_name": "first name",
    "last_name": "last name",
    "member_id": "Member ID — Must begin with m followed by 6 digit",
    "dob": "date of birth — Must include year, month, and day",
    "relationship": "whether they are the plan holder or dependent",
    "phone_confirmed": "phone number on file — yes or no",
    "phone_confirmation": "phone number on file — yes or no",
    "caller_role": "relationship to the plan",
    "provider_type": "type of provider they are looking for",
    "zip_code": "five-digit ZIP code",
    "delivery_method": "fax or email",
    "intent": "what they need help with today — ask openly, never list options",
    "topic": "what they need help with",
    "reference_number": "reference number — should be 8 digits",
    "upload_method": "how they want to provide the medical records — upload "
    "themselves, have their doctor send them, or have us contact their provider",
    "upload_consent": "whether they want to receive a secure upload link via email (yes or no)",
    "personal_guide_consent": "yes or no — whether they want a Personal Guide to "
    "contact their provider and request the medical records on their behalf",
    "email": "correct email address for the upload link",
    "notification_method": "preferred notification channel — SMS or email",
    "phone": "correct phone number",
    "n2_notification_method": "preferred channel for claim progress updates — SMS or email",
    "timeline_question": (
        "whether they have questions about the timeline — "
        "say yes to hear it, no to skip, or ask their question directly"
    ),
}

# ── Recovery guard labels ────────────────────────────────────────────────────
# These are Python-internal routing labels passed to generate_recovery_message().
# They are NOT LLM extraction outputs (see llm/schema.py EventType for those).
#
# Label          | Produced by              | Meaning
# ---------------|--------------------------|-----------------------------------
# "RETRY"        | _collect_slot            | Genuine failed answer — attempt counted
# "CLARIFY"      | _collect_slot            | First AMBIGUOUS turn — no attempt cost;
#                |                          | ask caller to repeat more clearly
# "CORRECTION"   | _generate_correction_ack | Caller corrected a confirmed slot
# "INTERRUPTION" | guards.py               | Caller switched topic mid-collection
# "OFFTOPIC_AGENT"| guards.py              | Wrong-agent topic — steer back
# "OFFTOPIC"     | guards.py (fallback)    | Legacy alias for OFFTOPIC_AGENT
#
# The distinction between RETRY and CLARIFY matters:
#   RETRY   → attempt_count was just incremented; LLM should re-ask firmly
#   CLARIFY → attempt_count unchanged; LLM should re-ask gently, no implication
#             that the caller did anything wrong
# Speech acts the unified voice speaks (Phase 1). "ask" and "transition" are the
# happy-path acts routed through the generator when UNIFIED_VOICE is on; the rest
# are the recovery acts the generator already spoke.
SPEECH_ACT_ASK = "ask"
SPEECH_ACT_TRANSITION = "transition"
SPEECH_ACT_MULTI_INTENT = "multi_intent"

_FALLBACKS: dict[str, str] = {
    "ask": "Could you provide your {slot_label}?",
    "transition": "Thank you. And your {slot_label}?",
    "INTERRUPTION": "Of course — and I still need your {slot_label}.",
    "OFFTOPIC": "Let me finish this first — your {slot_label}?",
    "RETRY": "Could you try your {slot_label} once more?",
    "OFFTOPIC_AGENT": "Let's stay focused — could I get your {slot_label}?",
    "CORRECTION": "Got it — I've updated that. Now could I get your {slot_label}?",
    "CLARIFY": (
        "I just want to make sure I have that right — \ncould you say your {slot_label} one more time for me?"
    ),
    "ANSWERED_WITH_FOLLOWUP": "Got that — and to confirm, you said {slot_label}.",
}


def _owner_label(owner: str) -> str:
    """Human phrase for a parked owner agent (lazy import avoids a cycle)."""
    try:
        from agent.responses.turn_acts import owner_label

        return owner_label(owner)
    except Exception:  # pragma: no cover - defensive
        return "that"


def build_recovery_context(
    *,
    slot_label: str,
    attempt: int,
    speech_act: str,
    history_text: str,
    user_utterance: str | None,
    confirmed_slots: dict | None,
    validated_answer: str | None,
    pending_slots: list[str] | None,
    parked: list[str] | None,
    declined: bool,
    answered_inline: list[str] | None = None,
    next_ask: str | None = None,
    correction_field: str | None = None,
) -> str:
    """Build the STRUCTURED decision context handed to the generator.

    The model does not infer the decision — it reads these labelled lines and
    phrases them into ONE sentence (composing across clauses in precedence order
    for a multi-intent turn). Only grounded, this-turn values ever appear as
    concrete values (``Validated answer this turn`` and ``Answer to include``);
    ``Confirmed`` lists slot *names*, never their values, so the generator cannot
    restate a prior identifier it was never given.
    """
    lines = [
        f"Speech act: {speech_act}",
        f"Collecting: {slot_label}",
        f"Attempt: {attempt}",
    ]
    if validated_answer is not None:
        lines.append(f"Validated answer this turn: {validated_answer}")
    if correction_field:
        lines.append(f"Correction acknowledged: {correction_field}")
    if answered_inline:
        for ans in answered_inline:
            lines.append(f"Answer to include (grounded, say verbatim): {ans}")
    if parked:
        parked_str = ", ".join(_owner_label(o) for o in parked)
        lines.append(f"Parked (say you'll get to it in a moment): {parked_str}")
    if declined:
        lines.append("Declined (briefly say you can't help with that one here): yes")
    if next_ask:
        lines.append(f"Next, ask for: {next_ask}")
    if confirmed_slots:
        lines.append("Confirmed: " + ", ".join(str(k) for k in confirmed_slots))
    elif confirmed_slots is not None:
        lines.append("Confirmed: nothing yet")
    if pending_slots:
        lines.append("Pending: " + ", ".join(pending_slots))
    if user_utterance:
        lines.append(f'Caller just said: "{user_utterance}"')
    lines += ["", "Conversation:", history_text]
    return "\n".join(lines)


async def generate_recovery_message(
    *,
    slot_name: str,
    attempt: int,
    guard: str,
    last_messages: list[dict],
    slot_label_override: str | None = None,
    caller_name: str | None = None,
    confirmed_slots: dict | None = None,
    user_utterance: str | None = None,
    extracted_value: str | None = None,
    pending_slots: list[str] | None = None,
    allow_followup_event: bool = False,  # accepted for back-compat; unused
    speech_act: str | None = None,
    parked: list[str] | None = None,
    declined: bool = False,
    answered_inline: list[str] | None = None,
    next_ask: str | None = None,
    correction_field: str | None = None,
    fallback_text: str | None = None,
) -> str:
    """
    Generate one natural spoken sentence via LLM 2 (Gemini) — the single voice
    for every turn (Phase 1).

    guard/speech_act: the speech act to phrase — "ask" | "transition" | "RETRY" |
      "CLARIFY" | "CORRECTION" | "ANSWERED_WITH_FOLLOWUP" | "INTERRUPTION" |
      "OFFTOPIC_AGENT". ``guard`` is the legacy name; when ``speech_act`` is given
      it is what's shown to the model (``guard`` still selects the static fallback).
    last_messages: recent conversation history (role/content dicts).
    caller_name: caller's first name if known, for personalisation.
    confirmed_slots: dict of already-confirmed slot names (values are not sent).
    user_utterance: the caller's most recent utterance.
    extracted_value: the value VALIDATED this turn, if any (safe to acknowledge).
    pending_slots: slots still to be collected, if known.
    parked: owner agents of side requests parked for later, if any.
    declined: whether a side request was declined this turn.
    fallback_text: exact string to return if generation fails/empties (the
      caller's template), guaranteeing no dead turn. Falls back to the static
      per-act string when not provided.

    Falls back to a static string on any exception, so a turn is never dropped.
    """
    history_text = "\n".join(build_history(last_messages, n=4))

    # Use the live prompt text when provided (dynamic slots such as relationship
    # and phone_confirmed whose options are only known at runtime from SF).
    # Fall back to the static label dict for fixed slots.
    slot_label = slot_label_override or _SLOT_LABELS.get(
        slot_name,
        (slot_name or _SLOT_LABELS["intent"]).replace("_", " "),
    )

    user_content = build_recovery_context(
        slot_label=slot_label,
        attempt=attempt,
        speech_act=speech_act or guard,
        history_text=history_text,
        user_utterance=user_utterance,
        confirmed_slots=confirmed_slots,
        validated_answer=extracted_value,
        pending_slots=pending_slots,
        parked=parked,
        declined=declined,
        answered_inline=answered_inline,
        next_ask=next_ask,
        correction_field=correction_field,
    )

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = get_generation_llm()
        response = await llm.ainvoke(
            [
                SystemMessage(content=build_generation_prompt()),
                HumanMessage(content=user_content),
            ]
        )
        text = (response.content or "").strip()
        if text:
            return text
    except Exception:
        logger.exception("generate_recovery_message: LLM 2 failed — using fallback")

    if fallback_text:
        return fallback_text
    return _FALLBACKS.get(guard, "Could you try again?").format(slot_label=slot_label)
