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
