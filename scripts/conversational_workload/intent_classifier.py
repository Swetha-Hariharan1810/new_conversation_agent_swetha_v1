"""
Deterministic slot classifier for PCP evaluation flow.
Keyword-based matching — clarification/correction checks run before slot checks
so that re-ask and ack messages are not misclassified as regular slot prompts.
"""


def classify_ai_slot(ai_message: str, flow: str = "pcp") -> str:  # noqa: C901
    """Return the slot label that best describes what the AI message is collecting."""
    if flow == "claim":
        return classify_ai_slot_claim(ai_message)

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


# ---------------------------------------------------------------------------
# Claim-adjustment flow dispatcher — called by classify_ai_slot(flow="claim")
# ---------------------------------------------------------------------------


def classify_ai_slot_claim(ai_message: str) -> str:  # noqa: C901
    """Return the slot label for claim-adjustment flow AI messages."""
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

    # --- Identity collection (shared with PCP flow) ---
    if "first name" in ai_lower:
        return "first_name"

    if "last name" in ai_lower:
        return "last_name"

    if "member id" in ai_lower and "date of birth" not in ai_lower:
        return "member_id"

    if "date of birth" in ai_lower:
        return "dob"

    if "phone number" in ai_lower and any(
        phrase in ai_lower for phrase in ["is your phone number", "is this correct", "on file"]
    ):
        return "phone_confirmed"

    # --- Claim-specific slots ---

    # reference_number must precede records_method (both can appear after auth)
    if any(phrase in ai_lower for phrase in ["reference number", "adjustment request"]):
        return "reference_number"

    # upload_link_sent: confirmation that the link was dispatched (contains delivery confirmation language)
    if any(
        phrase in ai_lower
        for phrase in [
            "expect to receive the link",
            "link has been sent",
            "upload link is on its way",
            "you should receive the upload link",
        ]
    ):
        return "upload_link_sent"

    # upload_consent: agent offering to send the upload link (precedes email_confirmed_upload)
    if any(
        phrase in ai_lower
        for phrase in ["send you a link", "upload the records directly", "link where you can upload"]
    ):
        return "upload_consent"

    # email_confirmed_upload: verifying on-file email before sending upload link
    if "email address we have on file" in ai_lower and any(
        phrase in ai_lower
        for phrase in ["correct or has it been changed", "still the right address", "correct email to send"]
    ):
        return "email_confirmed_upload"

    # guide_scheduled: confirmation that Personal Guide outreach is booked
    if any(phrase in ai_lower for phrase in ["personal guide will call", "next 24 hours"]):
        return "guide_scheduled"

    # personal_guide_consent: offering Personal Guide outreach (precedes guide_scheduled)
    if any(
        phrase in ai_lower
        for phrase in ["personal guide", "outreach to your doctor", "would you like us to proceed"]
    ):
        return "personal_guide_consent"

    # N2 preference confirmed — follow-up opening
    if (
        any(
            phrase in ai_lower
            for phrase in [
                "i'll send progress updates",
                "i will send progress updates",
                "send progress updates",
                "sure, i will send sms",
                "sure, i will send email",
                "perfect, i'll send progress",
                "got it, i'll send progress",
                "aside from this",
                "is there anything else i can help you with today",
                "is there anything else i can assist you with",
            ]
        )
        and "sms or email" not in ai_lower
        and "email or sms" not in ai_lower
    ):
        return "follow_up"

    # n2_notification_method: second notification-channel ask (progress updates)
    # Detected when both a notification-channel phrase AND a progress/update phrase are present
    if any(
        phrase in ai_lower for phrase in ["sms or email", "email or sms", "how do you want to be notified"]
    ) and any(phrase in ai_lower for phrase in ["updates", "keep you posted", "progress", "notifications"]):
        return "n2_notification_method"

    # Personal Guide scheduled — N1 notification bridge
    if any(
        phrase in ai_lower
        for phrase in [
            "personal guide will call",
            "personal guide will reach out",
            "team will contact the provider",
            "our team will contact",
            "we can also keep you posted",
            "would you prefer those by email or sms",
            "would you prefer those by sms or email",
        ]
    ):
        return "guide_scheduled"

    # notification_method: first notification-channel ask
    if any(
        phrase in ai_lower for phrase in ["email or sms", "sms or email", "how do you want to be notified"]
    ):
        return "notification_method"

    # timeline_question: agent providing or offering timeline details
    if any(phrase in ai_lower for phrase in ["expected timeline", "how long will it take", "business days"]):
        return "timeline_question"

    # records_method: agent asking member how records will be delivered
    if any(
        phrase in ai_lower
        for phrase in [
            "send it over",
            "upload the records",
            "doctor to send",
            "complete copy of the medical records",
            "medical records for this request",
            "medical records for this adjustment",
            "copy of the medical records",
        ]
    ):
        return "records_method"

    # --- Closing ---
    if "thank you for calling" in ai_lower or "have a great day" in ai_lower:
        return "closing"

    if "is there anything else" in ai_lower or "anything else i can" in ai_lower:
        return "follow_up"

    return "other"
