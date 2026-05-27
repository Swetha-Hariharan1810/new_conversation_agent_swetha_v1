"""constants.py — Configuration for IntakeAgent."""

from __future__ import annotations

MAX_CLARIFICATION_ATTEMPTS = 2

# Regulatory: contractual opening statement — must not be LLM-generated
GREETING = (
    "Thank you for calling Sagility Health. "
    "For quality assurance purposes, your call may be monitored or recorded. "
    "At the end of your call today, there will be a short survey to provide "
    "feedback regarding the representative you spoke with. "
    "Your participation is greatly appreciated. "
    "Please tell me how I can assist you today."
)

INTENT_SLOT = "intent"

UNCLEAR_INTENT_REASON = "Intent could not be classified after maximum clarification attempts"
OFFTOPIC_REASON = "Repeated unsupported or off-topic requests"
EXPLICIT_ESCALATION_REASON = "Caller requested human assistance"

LOG_INTAKE_GREETING = "IntakeAgent: delivering greeting"
LOG_INTENT_CLASSIFIED = "IntakeAgent: intent classified"
LOG_INTENT_UNCLEAR = "IntakeAgent: intent unclear"
LOG_OFFTOPIC = "IntakeAgent: off-topic request"
LOG_MAX_RETRY = "IntakeAgent: max clarification reached"
LOG_CLASSIFICATION_FAILURE = "IntakeAgent: classification failure"

INTENT_BRIDGE_MSGS = [
    "I can definitely help with that. To get started, could I get your first name?",
    "Of course — happy to help. Can I start with your first name?",
    "Absolutely. Let me pull that up for you — first, what's your first name?",
    "Sure thing. Could I start with your first name?",
]
# Backward-compatible sentinel: common substring present in every pool message.
INTENT_BRIDGE_MSG = "your first name?"

# Escalation handoff messages — delivered at moment of escalation
UNCLEAR_ESCALATION = "Not a problem — let me connect you with a representative who can assist you further."
OFFTOPIC_ESCALATION = "Let me connect you with a representative who may better assist you."

UNCLEAR_FIRST_ATTEMPT_MSGS = [
    "I'd be happy to help. Could you tell me a little more about what you need today?",
    "Of course. What can I help you with?",
    "Sure thing. What brings you in today?",
    "Happy to help — what would you like assistance with?",
]
