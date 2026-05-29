"""
Deterministic slot classifier for PCP evaluation flow.
Keyword-based matching — clarification/correction checks run before slot checks
so that re-ask and ack messages are not misclassified as regular slot prompts.
"""


def classify_ai_slot(ai_message: str, flow: str = "pcp") -> str:  # noqa: C901
    """Return the slot label that best describes what the AI message is collecting."""
    if not ai_message:
        return "other"

    ai_lower = ai_message.lower()

    # --- Re-ask / clarification (must precede slot checks) ---
    if any(
        phrase in ai_lower
        for phrase in [
            "i'm sorry, i didn't quite catch",
            "i didn't quite catch",
            "could you repeat",
            "one more time",
            "i wasn't able to catch",
            "didn't catch that",
            "could you say that again",
            "could you please repeat",
        ]
    ):
        return "clarification"

    # --- Correction acknowledgement (must precede slot checks) ---
    if any(
        phrase in ai_lower
        for phrase in [
            "got it, i've updated",
            "got it i've updated",
            "i've updated",
            "i've corrected",
            "i have updated",
            "i have corrected",
            "thank you for the correction",
        ]
    ):
        return "correction_ack"

    # --- Greeting / intent ---
    if (
        any(
            phrase in ai_lower
            for phrase in [
                "how can i assist",
                "how may i assist",
                "how can i help",
                "how i can help",
                "please tell me how i can",
            ]
        )
        and "is there anything else" not in ai_lower
    ):
        return "intent_selection"

    # --- PCP collection slots ---
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

    # Explicit confirmation of an on-file zip (must precede zip_update to avoid
    # "correct zip" in "Is that the correct ZIP code?" matching zip_update)
    if "zip code" in ai_lower and any(
        phrase in ai_lower
        for phrase in ["is that correct", "is this correct", "on file", "to confirm", "for your search"]
    ):
        return "zip_confirm"

    # ZIP re-ask: agent explicitly asks for a new/different zip code
    if any(phrase in ai_lower for phrase in ["5-digit zip", "five-digit zip", "new zip", "provide.*zip"]):
        return "zip_update"

    if "zip code" in ai_lower:
        return "zip_confirm"

    if "via fax or email" in ai_lower or "fax or email" in ai_lower:
        return "fax_or_email"

    # Fax re-ask must precede fax_confirm
    if any(phrase in ai_lower for phrase in ["new fax number", "correct fax number"]):
        return "fax_update"

    if "fax number" in ai_lower and any(
        phrase in ai_lower for phrase in ["is that correct", "is this correct", "on file", "to confirm"]
    ):
        return "fax_confirm"

    # --- Benefits / coaching ---
    if "individual deductible" in ai_lower or "family deductible" in ai_lower:
        return "benefits_explanation"

    if "benefits" in ai_lower and any(
        phrase in ai_lower for phrase in ["would you like to also", "also get the benefits", "hear about"]
    ):
        return "benefits_offer"

    if "care coach" in ai_lower or "wellness coach" in ai_lower:
        return "coach_offer"

    # --- Closing ---
    if "thank you for calling" in ai_lower or "have a great day" in ai_lower:
        return "closing"

    if "is there anything else" in ai_lower or "anything else i can" in ai_lower:
        return "follow_up"

    return "other"
