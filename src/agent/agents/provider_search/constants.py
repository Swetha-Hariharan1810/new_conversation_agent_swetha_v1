"""
constants.py — Configuration constants for ProviderSearchAgent.

Contains ONLY:
  - Agent name string
  - Retry limits
  - Log labels
  - Static message pools

No logic, no imports from other agent modules.
"""

from __future__ import annotations

# ── Agent identity ────────────────────────────────────────────────────────────

AGENT_NAME = "provider_search_agent"

# ── Retry limits ──────────────────────────────────────────────────────────────

MAX_SLOT_ATTEMPTS = 2

# ── Log labels ────────────────────────────────────────────────────────────────

LOG_ENTERED = "provider_search_agent: entered"
LOG_PROVIDER_TYPE = "provider_search_agent: provider_type collected"
LOG_ZIP_CONFIRMED = "provider_search_agent: zip_code confirmed"
LOG_ZIP_UPDATED = "provider_search_agent: zip_code updated"

# ── ZIP confirmation templates ────────────────────────────────────────────────

ZIP_CONFIRM_TEMPLATES = [
    "Thank you. Your ZIP code is {zip_code}, correct?",
    "I have your ZIP code as {zip_code}. Is that right?",
    "Just to confirm — your ZIP code is {zip_code}?",
]

ZIP_UPDATE_PROMPT = "No problem — what is your current 5-digit ZIP code?"

# ── Delivery bridge templates (spoken after ZIP confirmed, before delivery sub-agent) ──

DELIVERY_BRIDGE_TEMPLATES = [
    "Great, I have a list of in-network {provider_type}s ready for you. "
    "Would you like that sent by fax or email?",
    "I have your in-network provider list ready — "
    "would you prefer to receive it by fax or email?",
    "I can send you a list of in-network {provider_type}s right away. "
    "Shall I send it to your fax or email?",
    "Your in-network provider list is ready. "
    "Would you like it sent by fax or email?",
]

# ── Escalation messages ───────────────────────────────────────────────────────

MSG_NOT_VERIFIED = [
    "I'm sorry, I need to verify your account first. Let me connect you with a representative.",
    "I wasn't able to confirm your identity. Connecting you with a specialist now.",
]

MSG_ZIP_EXHAUST = [
    "I'm having trouble confirming your ZIP code. Let me connect you with a representative.",
    "I wasn't able to confirm your ZIP code. Connecting you with a specialist now.",
]
