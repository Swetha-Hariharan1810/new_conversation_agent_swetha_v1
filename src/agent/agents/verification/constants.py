"""
Configuration constants for VerificationAgent.

IMPORTANT:
This module centralizes all fixed values:
- retry limits
- slot ordering
- error messages
- logging event names

Do NOT place:
- orchestration logic
- prompt strings
- LLM schemas
inside this file.
"""

from __future__ import annotations

# =========================================================
# Retry Limits
# =========================================================

MAX_LOOKUP_ATTEMPTS = 2

# Maximum rejection cycles before escalating the name confirmation loop.
# One cycle = one readback delivered → member rejects → (correction collected or not).
# Successful confirmation never increments this counter.
MAX_NAME_CONFIRM_ATTEMPTS = 3

# =========================================================
# Agent Identity
# =========================================================

VERIFICATION_AGENT_NAME = "verification_agent"

# =========================================================
# Slot Ordering
# =========================================================

IDENTITY_SLOT_ORDER = [
    "first_name",
    "last_name",
    "member_id",
    "dob",
]

# =========================================================
# Completion Message Template
# =========================================================

VERIFIED_MSG_TEMPLATES = [
    "Thank you, {first_name}. I've verified your account.",
    "Got it, {first_name} — your account is verified.",
    "Perfect, {first_name}. I've confirmed your identity.",
    "All set, {first_name}. Your account is verified.",
]

# =========================================================
# Logging Events
# =========================================================

LOG_ENTERED = "verification_agent: entered"
LOG_VERIFIED = "VerificationAgent: fully verified — signalling complete"
LOG_LOOKUP_FAIL = "VerificationAgent: SF lookup failed"
LOG_SLOT_CORRECTED = "VerificationAgent: slot corrected"
LOG_LLM_EXTRACT_FAIL = "VerificationAgent: LLM extraction failed — using empty decision"
LOG_INVALID_MEMBER_ID = "VerificationAgent: invalid member ID blocked"
LOG_INVALID_DOB = "VerificationAgent: invalid DOB blocked"

# =========================================================
# Name readback / confirmation messages
# =========================================================
# Placeholder: {spelled} — full name spelled with hyphens, e.g. "E-M-I-L-Y  C-A-R-T-E-R"

NAME_READBACK_TEMPLATES = [
    "Thank you. Just to confirm — is your name {spelled}, correct?",
    "Got it. So that's {spelled} — is that right?",
    "Let me read that back: {spelled}. Is that correct?",
]

# Used after bare "no" — agent asks for the correct name before re-reading back.
NAME_CORRECTION_PROMPTS = [
    "No problem — could you give me the correct first and last name?",
    "Got it — what is the correct name on the account?",
    "Sure, what is the correct name?",
]

# Escalation message when MAX_NAME_CONFIRM_ATTEMPTS is reached.
MSG_NAME_CONFIRM_EXHAUST = [
    "I wasn't able to confirm the name on the account after a few tries. "
    "Let me connect you with a representative who can assist.",
    "I wasn't able to verify the name after several attempts. "
    "Connecting you with a specialist now.",
]

# ── New log labels ────────────────────────────────────────────────────────────
LOG_NAME_READBACK         = "VerificationAgent: name readback delivered"
LOG_NAME_CONFIRMED        = "VerificationAgent: name confirmed by member"
LOG_NAME_CORRECTED        = "VerificationAgent: name corrected by member"
LOG_NAME_CONFIRM_EXHAUST  = "VerificationAgent: name confirmation exhausted — escalating"
