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
# "ANSWERED_WITH_FOLLOWUP" | _collect_slot / intake | Slot value accepted this turn;
#                |                          | acknowledge it and address the caller's
#                |                          | side question — no attempt cost, no re-ask
# "CORRECTION"   | _generate_correction_ack | Caller corrected a confirmed slot
# "INTERRUPTION" | guards.py               | Caller switched topic mid-collection
# "OFFTOPIC_AGENT"| guards.py              | Wrong-agent topic — steer back
# "OFFTOPIC"     | guards.py (fallback)    | Legacy alias for OFFTOPIC_AGENT
#
# The distinction between RETRY and CLARIFY matters:
#   RETRY   → attempt_count was just incremented; LLM should re-ask firmly
#   CLARIFY → attempt_count unchanged; LLM should re-ask gently, no implication
#             that the caller did anything wrong
_FALLBACKS: dict[str, str] = {
    "INTERRUPTION": "Of course — and I still need your {slot_label}.",
    "OFFTOPIC": "Let me finish this first — your {slot_label}?",
    "RETRY": "Could you try your {slot_label} once more?",
    "OFFTOPIC_AGENT": "Let's stay focused — could I get your {slot_label}?",
    "CORRECTION": "Got it — I've updated that. Now could I get your {slot_label}?",
    "CLARIFY": (
        "I just want to make sure I have that right — \ncould you say your {slot_label} one more time for me?"
    ),
    "ANSWERED_WITH_FOLLOWUP": "Got that — and to confirm, that's your {slot_label}.",
}


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
) -> str:
    """
    Generate a natural recovery response via LLM 2 (Gemini).

    guard: "CORRECTION" | "CLARIFY" | "INTERRUPTION" | "OFFTOPIC" | "RETRY" | "OFFTOPIC_AGENT"
           | "ANSWERED_WITH_FOLLOWUP"
    last_messages: recent conversation history (role/content dicts); up to 8 messages used.
    caller_name: caller's first name if already confirmed, for personalisation.
    confirmed_slots: dict of slot names → confirmed values for this session.
    user_utterance: the caller's most recent utterance, so the LLM knows what was said.
    extracted_value: value successfully extracted this turn, if any.
    pending_slots: ordered list of slot names still to be collected, if known.

    Falls back to static string on any exception.
    """
    history_text = "\n".join(build_history(last_messages, n=4))

    # Use the live prompt text when provided (dynamic slots such as relationship
    # and phone_confirmed whose options are only known at runtime from SF).
    # Fall back to the static label dict for fixed slots.
    slot_label = slot_label_override or _SLOT_LABELS.get(
        slot_name,
        (slot_name or _SLOT_LABELS["intent"]).replace("_", " "),
    )

    content_lines = [
        "Conversation:",
        history_text,
        "",
    ]
    if user_utterance:
        content_lines.append(f"Caller just said: {user_utterance}")
    content_lines += [
        f"Collecting: {slot_label}",
        f"Attempt:    {attempt}",
    ]
    if confirmed_slots is not None and len(confirmed_slots) > 0:
        filled = ", ".join(f"{k}={v}" for k, v in confirmed_slots.items())
        content_lines.append(f"Confirmed:  {filled}")
    elif confirmed_slots is not None:
        content_lines.append("Confirmed:  nothing yet")
    if pending_slots:
        content_lines.append(f"Pending:    {', '.join(pending_slots)}")
    if extracted_value is not None:
        content_lines.append(f"Extracted this turn: {extracted_value}")
    if guard in ("CORRECTION", "CLARIFY", "OFFTOPIC_AGENT", "ANSWERED_WITH_FOLLOWUP"):
        content_lines.append(f"Event:      {guard}")

    user_content = "\n".join(content_lines)

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

    return _FALLBACKS.get(guard, "Could you try again?").format(slot_label=slot_label)
