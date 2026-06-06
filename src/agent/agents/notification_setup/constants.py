"""constants.py — Configuration constants for NotificationSetupAgent."""

from __future__ import annotations

AGENT_NAME = "notification_setup_agent"

LOG_ENTERED = "notification_setup_agent: entered"
LOG_PREFERENCE_SAVED = "notification_setup_agent: notification preference saved to Salesforce"
LOG_N2_PREFERENCE_SAVED = "notification_setup_agent: timeline notification preference saved"
LOG_METHOD_COLLECTED = "notification_setup_agent: notification_method collected"

# ── Initial ask ────────────────────────────────────────────────────────────────
# Note: In Scenario B the bridge is already delivered by records_coordination
# (MSG_NOTIFICATION_BRIDGE). The initial ask here fires only when entering
# notification_setup directly (no records sub-agent, or on re-entry).
NOTIFICATION_METHOD_ASK = [
    "To keep you posted, I can send you notifications whenever there are updates "
    "in this request. Do you want to receive them via SMS or email?",
    "How would you like to receive status updates on this request — SMS or email?",
    "I can send you case status updates by SMS or email. Which do you prefer?",
]

# ── Phone readback ─────────────────────────────────────────────────────────────
# Placeholder: {phone}
PHONE_READBACK_TEMPLATES = [
    "The phone number we have on file is {phone}. Is this correct or has it been changed?",
    "I'll send SMS updates to {phone}. Is that still correct?",
    "I have {phone} on file for SMS. Is that right?",
]

# ── Email readback ─────────────────────────────────────────────────────────────
# Placeholder: {email}
EMAIL_READBACK_TEMPLATES = [
    "The email address we have on file is {email}. Is this correct or has it been changed?",
    "I'll send updates to {email}. Is that still the right address?",
    "I have {email} on file. Is that the correct email for notifications?",
]

# ── Contact update prompts ─────────────────────────────────────────────────────
PHONE_UPDATE_PROMPTS = [
    "No problem — what is the correct phone number?",
    "Got it — could I get the updated phone number?",
    "Sure — what phone number should we use?",
]
EMAIL_UPDATE_PROMPTS = [
    "No problem — what is the correct email address?",
    "Got it — could I get the updated email address?",
    "Sure — what email address should we use?",
]

# ── Confirmation messages ──────────────────────────────────────────────────────
# Placeholders: {method}, {contact}
PREFERENCE_SAVED_TEMPLATES = [
    "Thank you. We will send the notification once we have reached your doctor.",
    "Got it. You'll receive updates via {method} at {contact}.",
    "Perfect. I've set up {method} notifications to {contact}.",
]

# ── Timeline bridge (connects notification confirm to timeline explanation) ────
TIMELINE_BRIDGE_TEMPLATES = [
    "I can walk you through the expected timeline for this request.",
    "Let me share the expected timeline with you.",
    "Here's what to expect in terms of timeline.",
]

# ── Same email confirmation sentence (used when re-confirming) ─────────────────
SAME_EMAIL_CONFIRM_TEMPLATES = [
    "Sure, I will use the same email address on record, {email}.",
    "I'll send it to {email}. Is there anything else I can help you with today?",
]

# ── Timeline fixed answer ──────────────────────────────────────────────────────
MSG_TIMELINE_ANSWER = [
    "It will take 5 to 10 business days from the date we receive the required additional information.",
    "The request is typically finalized within 5 to 10 business days of receiving the required information.",
]

# ── Second notification preference (claim timeline / progress updates) ─────────
N2_METHOD_ASK = [
    "To keep you posted, I can send you notifications whenever there are updates "
    "in this request. Do you want to receive them via SMS or email?",
    "I can also send you progress updates on this request by SMS or email. Which do you prefer?",
]

N2_EMAIL_CONFIRM = [
    "Sure, I will use the same email address on record, {email}.",
    "Perfect, I'll send progress updates to {email}.",
]

N2_PHONE_CONFIRM = [
    "Sure, I will send SMS notifications to {phone}.",
    "Got it, I'll send progress updates by SMS to {phone}.",
]

# ── Escalation messages ────────────────────────────────────────────────────────
MSG_METHOD_EXHAUST = [
    "I wasn't able to confirm your notification preference. Let me connect you with a representative.",
    "I wasn't able to set up your notification preference. Connecting you with a specialist.",
]
MSG_CONTACT_EXHAUST = [
    "I wasn't able to confirm your contact details for notifications. "
    "Let me connect you with a representative.",
]
