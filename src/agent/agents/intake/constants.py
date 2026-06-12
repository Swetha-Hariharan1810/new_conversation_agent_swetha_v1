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

# Out-of-scope handling
LOG_OUT_OF_SCOPE = "IntakeAgent: out-of-scope intent detected"

OUT_OF_SCOPE_REASON = "Caller intent is outside covered workflows"

# Keyword → (human-readable team label, phone number)
# Checked against the caller's last utterance using substring match.
# Order matters — more specific phrases must come before shorter ones.
# e.g. "insurance card" before "card", "prior approval" before "prior"
OUT_OF_SCOPE_KEYWORD_ROUTING: list[tuple[str, str, str]] = [
    # billing
    ("insurance card", "our member services team", "1-800-555-0102"),
    ("member card", "our member services team", "1-800-555-0102"),
    ("id card", "our member services team", "1-800-555-0102"),
    ("prior approval", "our authorizations team", "1-800-555-0103"),
    ("prior auth", "our authorizations team", "1-800-555-0103"),
    ("billing", "our billing team", "1-800-555-0101"),
    ("invoice", "our billing team", "1-423-872-2404"),
    ("payment", "our billing team", "1-423-872-24041"),
    ("pay my", "our billing team", "1-800-555-0101"),
    ("my bill", "our billing team", "1-800-555-0101"),
    ("referral", "our authorizations team", "1-800-555-0103"),
    ("authorization", "our authorizations team", "1-800-555-0103"),
    ("prescription", "our pharmacy benefits team", "1-800-555-0104"),
    ("pharmacy", "our pharmacy benefits team", "1-800-555-0104"),
    ("medication", "our pharmacy benefits team", "1-800-555-0104"),
    ("drug", "our pharmacy benefits team", "1-800-555-0104"),
    ("enrol", "our member services team", "1-800-555-0102"),
    ("enroll", "our member services team", "1-800-555-0102"),
    ("cancel", "our member services team", "1-800-555-0102"),
    ("coverage", "our member services team", "1-800-555-0102"),
]

# Fallback when no keyword matches
OUT_OF_SCOPE_FALLBACK_TEAM = "our member services team"
OUT_OF_SCOPE_FALLBACK_NUMBER = "1-800-555-0102"

# Message templates. Placeholders:
#   {topic_description} — what the caller said, in their words
#   {team}              — human-readable team name from routing table
#   {number}            — phone number from routing table
OUT_OF_SCOPE_MSG_TEMPLATES = [
    (
        "I understand you're calling about {topic_description}. "
        "That's handled by {team} — I'll connect you now. "
        "Their direct number is {number} if you need to call back."
    ),
    (
        "Got it — {topic_description} is something {team} can help you with. "
        "Let me transfer you to them now. "
        "You can also reach them directly at {number}."
    ),
    (
        "Thanks for letting me know. For {topic_description}, "
        "you'll want to speak with {team}. "
        "I'm connecting you now — their number is {number}."
    ),
]
