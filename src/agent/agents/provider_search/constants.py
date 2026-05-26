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

ZIP_UPDATE_PROMPT = "Please provide the 5-digit ZIP code you'd like to update to."

# ── Delivery bridge templates (spoken after ZIP confirmed, before delivery sub-agent) ──

DELIVERY_BRIDGE_TEMPLATES = [
    "Alright. I have a list of in-network providers.",
    "Great — I've got your in-network provider list ready.",
    "Perfect. I have a list of in-network {provider_type}s for you.",
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
