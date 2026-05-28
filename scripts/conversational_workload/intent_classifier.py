"""
Deterministic slot classifier for PCP evaluation flow.
Keyword-based matching for stability across LLM response variations.
"""


def classify_ai_slot(ai_message: str, flow: str = "pcp") -> str:  # noqa: C901
    """Return the slot label that best describes what the AI message is collecting."""
    if not ai_message:
        return "other"

    ai_lower = ai_message.lower()

    # Global: intent / greeting
    if any(
        phrase in ai_lower
        for phrase in [
            "how can i assist",
            "how may i assist",
            "how can i help",
            "how i can help",
            "please tell me how i can",
        ]
    ) and "is there anything else" not in ai_lower:
        return "intent_selection"

    # PCP flow
    if "first name" in ai_lower:
        return "first_name"

    if "last name" in ai_lower:
        return "last_name"

    if "member id" in ai_lower and "date of birth" not in ai_lower:
        return "member_id"

    if "date of birth" in ai_lower:
        return "dob"

    if "plan holder" in ai_lower or "subscriber" in ai_lower or "dependent" in ai_lower:
        return "subscriber_confirm"

    if "type of provider" in ai_lower or "what type of provider" in ai_lower:
        return "provider_type"

    if "zip code" in ai_lower:
        return "zip_confirm"

    if "via fax or email" in ai_lower or "fax or email" in ai_lower:
        return "fax_or_email"

    if (
        "fax number" in ai_lower
        and ("is that correct" in ai_lower or "is this correct" in ai_lower or "correct" in ai_lower)
    ):
        return "fax_confirm"

    if "individual deductible" in ai_lower or "family deductible" in ai_lower:
        if "care coach" in ai_lower or "wellness coach" in ai_lower:
            return "benefits_and_coach"
        return "benefits_explanation"

    if "care coach" in ai_lower or "wellness coach" in ai_lower:
        return "coach_interest"

    if "benefits" in ai_lower and (
        "would you" in ai_lower or "also like" in ai_lower or "hear about" in ai_lower
    ):
        return "benefits_offer"

    if "is there anything else" in ai_lower or "anything else i can" in ai_lower:
        return "closing"

    if "thank you for calling" in ai_lower or "have a great day" in ai_lower:
        return "farewell"

    # Correction acknowledgement
    if "correction" in ai_lower or "updated your" in ai_lower or "i've updated" in ai_lower:
        return "correction_ack"

    return "other"
