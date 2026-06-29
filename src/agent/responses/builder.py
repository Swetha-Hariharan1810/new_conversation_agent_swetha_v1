"""
response_builder.py — Context-aware conversational response generation.

Single source of truth for all slot-prompt variation.

Public API used by agents and slot infrastructure:
  build_initial_prompt(slot_type)               → str
  build_transition_prompt(slot_type, context)   → str

All selection is pure Python — no LLM calls, no I/O, zero latency.
"""

from __future__ import annotations

import random

from agent.conversation.context import (
    ConversationContext,
)
from agent.slots.types import SlotType

__all__ = [
    "build_initial_prompt",
    "build_transition_prompt",
]

# ---------------------------------------------------------------------------
# Transition templates: moving from the previous confirmed slot to this one
# ---------------------------------------------------------------------------

_TRANSITION_TEMPLATES: dict[SlotType, list[str]] = {
    SlotType.LAST_NAME: [
        "Thank you. And your last name?",
        "Got it. Could I get your last name?",
        "Great. What's your last name?",
        "Thank you. May I have your last name?",
    ],
    SlotType.MEMBER_ID: [
        "Thank you{name_part}. May I have your Member ID?",
        "Great. Could you provide your Member ID number?",
        "Okay{name_part}. May I ask for your Member ID?",
        "Perfect. Could I get your Member ID?",
    ],
    SlotType.DOB: [
        "Thank you. And your date of birth?",
        "Got it{name_part}. What's the date of birth on the account?",
        "Almost there{name_part} — and your date of birth?",
        "And your date of birth?",
    ],
    SlotType.RELATIONSHIP: [
        "Thank you. Are you the plan holder, or are you calling for yourself?",
        "Got it. Could you confirm — are you the plan holder or a subscriber?",
        "And are you the primary account holder?",
    ],
    SlotType.PHONE_NUMBER: [
        "Thank you{name_part}. What is the best number to reach you?",
        "Got it. Could you provide your phone number?",
        "And the phone number on the account?",
    ],
    SlotType.ZIP_CODE: [
        "Thank you. Could you confirm your ZIP code?",
        "Got it. And your ZIP code?",
        "What ZIP code are we working with?",
    ],
    SlotType.EMAIL: [
        "Thank you. And what is your email address?",
        "Got it. Could you provide your email?",
        "And your email address?",
    ],
    SlotType.FAX: [
        "Thank you. And the fax number you'd like us to use?",
        "Got it. Could you provide your fax number?",
    ],
    SlotType.PROVIDER_TYPE: [
        "Thank you{name_part}. What type of provider are you looking for?",
        "Got it. What kind of doctor or specialist do you need?",
        "And what type of provider are you searching for?",
        "What type of care are you looking for — a primary care physician, for example?",
    ],
    SlotType.CLAIM_NUMBER: [
        "Thank you{name_part}. May I have the reference number for the adjustment?",
        "Got it. Could you provide your claim reference number?",
        "And the adjustment reference number?",
    ],
    SlotType.NOTIFICATION_METHOD: [
        "We can also keep you posted on the status of the provider outreach. "
        "I can send the status updates to your email or SMS. How do you want to be notified?",
        "To keep you updated, would you prefer notifications by SMS or email?",
    ],
    SlotType.DELIVERY_METHOD: [
        "Thank you. How would you like us to send this — fax or email?",
        "Got it. Would you prefer fax or email?",
        "And for delivery — fax or email?",
    ],
}

_DEFAULT_TRANSITION = [
    "Thank you{name_part}. Could you provide {slot_label}?",
    "Got it. Could you provide {slot_label}?",
    "Thank you. Could you provide {slot_label}?",
]

# ---------------------------------------------------------------------------
# First-ask templates
# ---------------------------------------------------------------------------

_INITIAL_TEMPLATES: dict[SlotType, list[str]] = {
    SlotType.FIRST_NAME: [
        "Can I get your first name, please?",
        "Could you start with your first name?",
        "To get started, what's your first name?",
        "Please go ahead with your first name.",
    ],
    SlotType.MEMBER_ID: [
        "May I ask for your Member ID, please?",
        "Could you provide your Member ID?",
        "I'll need your Member ID — go ahead whenever you're ready.",
        "Please share your Member ID.",
    ],
    SlotType.DOB: [
        "To validate your account, can I get your date of birth?",
        "Could you provide your date of birth in month, day, year format?",
        "May I have your date of birth?",
        "I'll need your date of birth to confirm your account.",
    ],
    SlotType.PROVIDER_TYPE: [
        "What type of provider are you looking for?",
        "What kind of doctor or specialist do you need?",
        "What type of care are you looking for today?",
        "Are you looking for a primary care physician, a specialist, or another type of provider?",
    ],
    SlotType.FAX: [
        "No problem — what is the correct fax number?",
        "Got it — could I get the updated fax number?",
        "Sure — what fax number should we use?",
    ],
    SlotType.EMAIL: [
        "No problem — what is the correct email address?",
        "Got it — could I get the updated email address?",
        "Sure — what email address should we use?",
    ],
    SlotType.DELIVERY_METHOD: [
        "Would you prefer to receive that by fax or email?",
        "How would you like us to send that — by fax or email?",
        "Should I send that via fax or email?",
    ],
    SlotType.REFERENCE_NUMBER: [
        "May I have the reference number of the adjustment request?",
        "Could you provide the reference number for your adjustment?",
        "I'll need the reference number from your adjustment request — go ahead whenever you're ready.",
        "What is the reference number for the adjustment?",
    ],
    SlotType.NOTIFICATION_METHOD: [
        "How do you want to be notified — via SMS or email?",
        "Would you prefer to receive status updates by SMS or email?",
        "I can send you notifications by SMS or email — which do you prefer?",
    ],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _name_part(context: ConversationContext) -> str:
    if context.should_use_name and context.caller_first_name:
        return f", {context.caller_first_name}"
    return ""


def _slot_label(slot_type: SlotType) -> str:
    return slot_type.value.replace("_", " ")


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------


def build_initial_prompt(slot_type: SlotType) -> str:
    """First ask for a slot at the start of a pipeline."""
    pool = _INITIAL_TEMPLATES.get(slot_type)
    if pool:
        return random.choice(pool)
    return f"Could you provide your {_slot_label(slot_type)}?"


def build_transition_prompt(
    slot_type: SlotType,
    context: ConversationContext,
) -> str:
    """
    Prompt that acknowledges the just-confirmed previous slot and asks for the
    next one. Produces natural flow rather than isolated form-filling.
    """
    np = _name_part(context)
    pool = _TRANSITION_TEMPLATES.get(slot_type, _DEFAULT_TRANSITION)
    template = random.choice(pool)
    return template.format(
        name_part=np,
        slot_label=_slot_label(slot_type),
    )
