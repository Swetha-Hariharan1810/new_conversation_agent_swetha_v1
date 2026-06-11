"""
constants.py — Configuration constants for DeliveryManagementAgent.
"""

from __future__ import annotations

# ── Agent identity ────────────────────────────────────────────────────────────

AGENT_NAME = "delivery_management_agent"

# ── Retry limits ──────────────────────────────────────────────────────────────

# Bound on contact readback rejections (explicit "no" or a replacement value)
# before escalating — prevents an unbounded confirm/update cycle.
MAX_CONTACT_CHANGE_CYCLES = 3

# ── Log labels ────────────────────────────────────────────────────────────────

LOG_ENTERED = "delivery_management_agent: entered"
LOG_METHOD_COLLECTED = "delivery_management_agent: delivery_method collected"
LOG_CONTACT_CONFIRMED = "delivery_management_agent: contact confirmed"
LOG_CONTACT_UPDATED = "delivery_management_agent: contact updated"
LOG_LIST_DISPATCHED = "delivery_management_agent: provider list dispatched"

# ── Delivery window ───────────────────────────────────────────────────────────

DELIVERY_WINDOW_MSG = [
    "Great! Expect to receive the list within 30 minutes.",
    "Perfect — you should receive it within 30 minutes.",
    "The list will be sent within 30 minutes.",
]

# ── Benefits offer ────────────────────────────────────────────────────────────

BENEFITS_OFFER_TEMPLATES = [
    (
        "Since you're calling about a list of in-network providers, "
        "would you like to also get the benefits for office visits with your {provider_type}?"
    ),
    ("Would you also like me to go over the benefits for office visits with your {provider_type}?"),
    ("I can also provide your benefits information for {provider_type} visits — would that be helpful?"),
]

# ── Delivery method prompt ────────────────────────────────────────────────────

FAX_CONFIRM_TEMPLATES = [
    "How do you want us to send your request — via fax or email?",
    "Would you prefer to receive the provider list by fax or email?",
]

# ── Contact readback templates ────────────────────────────────────────────────

FAX_READBACK_TEMPLATES = [
    "Definitely. The fax number we have on file is {fax}. Is this correct?",
    "I'll send it to {fax} — is that the right fax number?",
]

EMAIL_READBACK_TEMPLATES = [
    "I'll send it to {email}. Is that the right email address?",
    "The email we have on file is {email}. Is that correct?",
]

# ── Contact update prompts ────────────────────────────────────────────────────

FAX_UPDATE_PROMPTS = [
    "No problem — what is the correct fax number?",
    "Got it — could I get the updated fax number?",
    "Sure — what fax number should we use?",
]

EMAIL_UPDATE_PROMPTS = [
    "No problem — what is the correct email address?",
    "Got it — could I get the updated email address?",
    "Sure — what email address should we use?",
]

# ── Escalation messages ───────────────────────────────────────────────────────

MSG_CONTACT_EXHAUST = [
    "I'm having trouble confirming your contact details. Let me connect you with a representative.",
    "I wasn't able to confirm your delivery details. Connecting you with a specialist now.",
]

MSG_DISPATCH_FAIL = [
    "I'm sorry, I was unable to send the provider list. Let me connect you with a representative.",
    "I wasn't able to dispatch the list. Connecting you with a specialist.",
]
