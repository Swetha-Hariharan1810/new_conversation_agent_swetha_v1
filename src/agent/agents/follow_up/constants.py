"""constants.py — Configuration constants for FollowUpAgent."""

from __future__ import annotations

AGENT_NAME = "follow_up_agent"

# ── Escalation thresholds ─────────────────────────────────────────────────────
# UPDATE_REQUEST: immediate escalation every time — no threshold, no counting.
# MSG_CANNOT_ANSWER: escalate after this many consecutive cannot-answer turns.
MAX_CANNOT_ANSWER_BEFORE_ESCALATE: int = 3

# ── Log labels ────────────────────────────────────────────────────────────────
LOG_ENTERED = "follow_up_agent: entered"
LOG_ANSWERED = "follow_up_agent: answered from session context"
LOG_CLOSURE = "follow_up_agent: closure signal detected"
LOG_CANNOT_ANSWER = "follow_up_agent: LLM could not answer from context"

# ── Closure keywords ──────────────────────────────────────────────────────────
CLOSURE_KEYWORDS: frozenset[str] = frozenset(
    {
        "no",
        "nope",
        "nah",
        "no thank",
        "no thanks",
        "that's all",
        "thats all",
        "that's it",
        "thats it",
        "nothing else",
        "nothing more",
        "all good",
        "all set",
        "i'm good",
        "im good",
        "i'm done",
        "im done",
        "done",
        "good",
        "ok",
        "okay",
        "great",
        "perfect",
        "thanks",
        "thank you",
        "bye",
        "goodbye",
        "that was helpful",
        "that helped",
        "very helpful",
        "have a good",
        "have a great",
    }
)

# Bare affirmations — intercepted before the LLM call.
BARE_AFFIRMATIONS: frozenset[str] = frozenset(
    {
        "yes",
        "yeah",
        "yep",
        "yup",
        "sure",
        "okay",
        "ok",
        "please",
        "yes please",
        "yes thank you",
        "sure please",
    }
)

# ── Message pools ─────────────────────────────────────────────────────────────

MSG_NUDGE: list[str] = [
    "Is there a specific question I can help you with, or are you all set for today?",
    "Did you have a specific question, or is there anything else I can help you with?",
    "Is there something specific from our call I can clarify, or are you good to go?",
]

MSG_CONTINUATION: list[str] = [
    "Is there anything else from our call today I can help with?",
    "Do you have any other questions about what we covered?",
    "Is there anything else I can clarify for you?",
]

MSG_CANNOT_ANSWER: list[str] = [
    "I'm sorry, I don't have that information from our call today.",
    "That isn't something we covered during this call, so I don't have those details available.",
    "I don't have that information available from what we discussed today.",
]

# Delivered on every UPDATE_REQUEST turn, immediately before signal_escalate().
# A single fixed string — no variation, no pool.
MSG_UPDATE_REQUEST_ESCALATE: str = (
    "I'm sorry, I'm not able to make changes during this part of the call. "
    "However, I will transfer you to a representative for further assistance."
)

# Max total follow-up turns (safety net).
MAX_FOLLOW_UP_TURNS: int = 10
