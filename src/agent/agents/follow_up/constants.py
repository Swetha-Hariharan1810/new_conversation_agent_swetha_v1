"""constants.py — Configuration constants for FollowUpAgent."""
from __future__ import annotations

AGENT_NAME = "follow_up_agent"

# ── Log labels ────────────────────────────────────────────────────────────────
LOG_ENTERED       = "follow_up_agent: entered"
LOG_ANSWERED      = "follow_up_agent: answered from session context"
LOG_CLOSURE       = "follow_up_agent: closure signal detected"
LOG_CANNOT_ANSWER = "follow_up_agent: LLM could not answer from context"

# ── Closure keywords ──────────────────────────────────────────────────────────
# When the member's utterance matches one of these, the agent signals closure
# instead of attempting to answer.
CLOSURE_KEYWORDS: frozenset[str] = frozenset({
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
})

# ── Continuation question pool ─────────────────────────────────────────────────
# Appended after every answered follow-up to invite the next question.
MSG_CONTINUATION: list[str] = [
    "Would you like help with anything else?",
    "Is there anything else I can help you with today?",
    "Is there anything else I can assist you with?",
    "Do you have any other questions?",
]

# ── Cannot-answer fallback ─────────────────────────────────────────────────────
# Returned when the LLM cannot find the answer in the session snapshot.
MSG_CANNOT_ANSWER: list[str] = [
    "I'm sorry, I don't have that specific information available from our call today.",
    "I wasn't able to find that detail in what we covered today.",
    "I don't have enough information on hand to answer that one.",
]

# Max consecutive follow-up turns before routing silently to closure.
MAX_FOLLOW_UP_TURNS: int = 5
