"""
response_generator.py — LLM 2: natural recovery response generation.

Called only on recovery turns. Uses Gemini (get_routing_llm).
Input is intentionally minimal. Output is one spoken sentence.
"""

from __future__ import annotations

from agent.core import flags
from agent.llm.config import get_generation_llm
from agent.logger import get_logger
from agent.responses.grounding import find_ungrounded_values
from agent.utils import build_generation_prompt, build_history

logger = get_logger(__name__)

# SHORT NOUN PHRASES only. A label appears both in the prompt context line
# ``Collecting:`` and interpolated into the spoken _FALLBACKS templates, so it
# must never carry instruction-style text — that belongs in _SLOT_DIRECTIVES /
# the ``generator_directive`` parameter, which is never interpolated anywhere.
_SLOT_LABELS: dict[str, str] = {
    "first_name": "first name",
    "last_name": "last name",
    "member_id": "Member ID",
    "dob": "date of birth",
    "relationship": "relationship to the plan holder",
    "phone_confirmed": "phone number confirmation",
    "phone_confirmation": "phone number confirmation",
    "caller_role": "relationship to the plan",
    "provider_type": "type of provider they are looking for",
    "zip_code": "five-digit ZIP code",
    "zip_confirmed": "ZIP code confirmation",
    "delivery_method": "delivery method (fax or email)",
    "fax": "fax number",
    "fax_confirmed": "fax number confirmation",
    "benefits_response": "interest in related benefits",
    "intent": "reason for calling today",
    "topic": "what they need help with",
    "reference_number": "reference number",
    "upload_method": "way to provide the medical records",
    "upload_consent": "secure upload link consent",
    "personal_guide_consent": "Personal Guide consent",
    "email": "email address",
    "email_confirmed": "email address confirmation",
    "notification_method": "notification channel (SMS or email)",
    "phone": "phone number",
    "n2_notification_method": "channel for claim progress updates (SMS or email)",
    "timeline_question": "timeline question",
}

# Default instruction-style guidance per slot, rendered as its own ``Guidance:``
# context line for the model. NEVER interpolated into a _FALLBACKS template —
# the caller only ever hears the noun-phrase label from _SLOT_LABELS.
_SLOT_DIRECTIVES: dict[str, str] = {
    "member_id": "The Member ID must begin with M followed by 6 digits.",
    "dob": "The date of birth must include year, month, and day.",
    "relationship": "Ask whether they are the plan holder or a dependent.",
    "phone_confirmed": "Ask whether the phone number on file is correct — yes or no.",
    "phone_confirmation": "Ask whether the phone number on file is correct — yes or no.",
    "zip_confirmed": "Ask whether the ZIP code on file is correct — yes or no.",
    "intent": "Ask openly what they need help with today — never list options.",
    "reference_number": "The reference number should be 8 digits.",
    "upload_method": (
        "Ask how they want to provide the medical records — upload themselves, "
        "have their doctor send them, or have us contact their provider."
    ),
    "upload_consent": "Ask whether they want to receive a secure upload link via email — yes or no.",
    "personal_guide_consent": (
        "Ask yes or no — whether they want a Personal Guide to contact their "
        "provider and request the medical records on their behalf."
    ),
    "notification_method": "Ask for their preferred notification channel — SMS or email.",
    "n2_notification_method": "Ask for their preferred channel for claim progress updates — SMS or email.",
    "timeline_question": (
        "Ask whether they have questions about the timeline — they can say yes "
        "to hear it, no to skip, or ask their question directly."
    ),
    "delivery_method": "Ask whether they'd like it by fax or email.",
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


# Instruction-style tells: a label carrying any of these was written for the
# MODEL, not the caller, and must never be spoken via a fallback template.
_DIRECTIVE_MARKERS: tuple[str, ...] = ("—", "(yes or no)", "ask for", "if they")


def _fallback_safe_label(slot_name: str, label: str) -> str:
    """Belt-and-suspenders for the static fallback path: a call site that hasn't
    migrated to ``generator_directive`` may still pass instruction-style text as
    the label. Detect it and interpolate the plain slot name instead, so an
    internal instruction can never reach the caller."""
    lowered = (label or "").lower()
    if len(label) > 60 or any(marker in lowered for marker in _DIRECTIVE_MARKERS):
        logger.warning(
            "generate_recovery_message: directive-style label blocked from fallback template",
            extra={"metric": "directive_label_blocked", "slot": slot_name},
        )
        return (slot_name or "that").replace("_", " ")
    return label


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
    directive: str | None = None,
) -> str:
    """Build the STRUCTURED decision context handed to the generator.

    The model does not infer the decision — it reads these labelled lines and
    phrases them into ONE sentence (composing across clauses in precedence order
    for a multi-intent turn). ``Collecting`` carries a short noun-phrase label;
    instruction-style guidance travels on its own ``Guidance:`` line (never in
    the label, so it can never leak into a spoken fallback template). Only
    grounded, this-turn values ever appear as concrete values (``Validated
    answer this turn`` and ``Answer to include``); ``Confirmed`` lists slot
    *names*, never their values, so the generator cannot restate a prior
    identifier it was never given.
    """
    lines = [
        f"Speech act: {speech_act}",
        f"Collecting: {slot_label}",
        f"Attempt: {attempt}",
    ]
    if directive:
        lines.append(f"Guidance: {directive}")
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
    generator_directive: str | None = None,
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
    grounded_values: list[str] | None = None,
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
    slot_label_override: SHORT NOUN PHRASE only (e.g. "ZIP code confirmation") —
      it is spoken via the fallback templates. Instruction-style text belongs in
      ``generator_directive``.
    generator_directive: instruction-style guidance for the model (format hints,
      what to confirm, a grounded value to read back). Rendered as its own
      ``Guidance:`` context line; NEVER interpolated into a fallback template.
      Any concrete value it asks the model to speak MUST be in ``grounded_values``
      (enforced by a debug assertion), or the grounding guard would veto the very
      read-back the directive asked for.
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

    # Use the live label when provided (dynamic slots such as relationship and
    # phone_confirmed whose phrasing is only known at runtime from SF). Fall
    # back to the static noun-phrase dict for fixed slots. Instruction-style
    # guidance travels separately on the directive, never in the label.
    slot_label = slot_label_override or _SLOT_LABELS.get(
        slot_name,
        (slot_name or _SLOT_LABELS["intent"]).replace("_", " "),
    )
    directive = generator_directive or _SLOT_DIRECTIVES.get(slot_name)

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
        directive=directive,
    )

    # Concrete values the generated text may state this turn. Computed up front
    # so the directive invariant below and the post-generation guard agree.
    allowed = _grounded_allowed(
        grounded_values=grounded_values,
        extracted_value=extracted_value,
        caller_name=caller_name,
        answered_inline=answered_inline,
    )
    # Invariant: if the prompt/directive asks the model to SPEAK a value, that
    # value must be grounded — otherwise the guard would veto the legitimate
    # read-back and the fallback would contradict what was asked (Bug 1).
    if __debug__ and directive:
        _directive_leaks = find_ungrounded_values(directive, allowed)
        assert not _directive_leaks, (
            f"generator_directive asks the model to speak ungrounded value(s) "
            f"{_directive_leaks!r} — pass them via grounded_values= "
            f"(see agent.responses.grounding.turn_grounding_allowlist)"
        )

    def _fallback() -> str:
        if fallback_text:
            return fallback_text
        safe_label = _fallback_safe_label(slot_name, slot_label)
        return _FALLBACKS.get(guard, "Could you try again?").format(slot_label=safe_label)

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = get_generation_llm()
        messages = [
            SystemMessage(content=build_generation_prompt()),
            HumanMessage(content=user_content),
        ]
        # Phase 4: stream to first token to cut perceived latency (voice). A stream
        # that errors mid-flight falls back to the template — never a dead turn.
        if flags.stream_generation():
            text = await _stream_generation(llm, messages)
        else:
            response = await llm.ainvoke(messages)
            text = (response.content or "").strip()
        if text:
            # Phase 4 grounding guard (belt-and-suspenders): free-flowing generation
            # may never state a concrete value (member id / ZIP / phone / date /
            # email) that wasn't grounded this turn. On violation, use the
            # deterministic template for this act — an invented sensitive value can
            # never reach the caller even if the prompt is imperfect.
            leaked = find_ungrounded_values(text, allowed)
            if leaked:
                logger.warning(
                    "generate_recovery_message: ungrounded value(s) in output — using template",
                    extra={"speech_act": speech_act or guard, "n_leaked": len(leaked)},
                )
                return _fallback()
            return text
    except Exception:
        logger.exception("generate_recovery_message: LLM 2 failed — using fallback")

    return _fallback()


def _grounded_allowed(
    *,
    grounded_values: list[str] | None,
    extracted_value: str | None,
    caller_name: str | None,
    answered_inline: list[str] | None,
) -> list[str]:
    """Concrete values the generated text is allowed to state this turn:
    confirmed_slots ∪ validated_answer (∪ a known first name ∪ inline answers)."""
    if grounded_values is not None:
        allowed = list(grounded_values)
    else:
        allowed = []
        if extracted_value:
            allowed.append(extracted_value)
    if caller_name:
        allowed.append(caller_name)
    if answered_inline:
        allowed.extend(answered_inline)
    return allowed


async def _stream_generation(llm, messages) -> str:
    """Accumulate a streamed generation to first token onward. Raises on a
    mid-flight stream error so the caller falls back to the template."""
    chunks: list[str] = []
    async for chunk in llm.astream(messages):
        piece = getattr(chunk, "content", "") or ""
        if piece:
            chunks.append(piece)
    return "".join(chunks).strip()
