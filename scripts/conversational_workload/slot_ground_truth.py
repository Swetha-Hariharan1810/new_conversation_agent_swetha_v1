"""
Maps slot labels to expected user responses for the PCP evaluation flow.
Uses entity data to populate dynamic values (name, member ID, etc.).
"""


def _pcp_map(entity) -> dict:
    fax = getattr(entity, "fax_number", None) or "6175554199"
    provider_type = getattr(entity, "provider_type", None) or "Primary Care Physician"

    return {
        "intent_selection": lambda: "I need to find a primary care physician in my area.",
        "first_name": lambda: entity.first_name,
        "last_name": lambda: entity.last_name,
        "member_id": lambda: f"m {' '.join(entity.member_id[1:])}",
        "dob": lambda: f"April twelfth {entity.date_of_birth[:4]}",
        "subscriber_confirm": lambda: "I'm calling for myself.",
        "provider_type": lambda: provider_type,
        "zip_confirm": lambda: "yes that's correct",
        "fax_or_email": lambda: "send it to my fax",
        "fax_confirm": lambda: "yes that's correct",
        "benefits_offer": lambda: "yes please",
        "benefits_explanation": lambda: "yes please",
        "benefits_and_coach": lambda: "yes that sounds interesting",
        "coach_interest": lambda: "yes that sounds interesting",
        "closing": lambda: "can you summarize the PCP benefits?",
        "farewell": lambda: "no thanks that was helpful",
        "correction_ack": lambda: entity.member_id,
        "other": lambda: "okay",
    }


def ground_truth_for_slot(slot: str, entity, flow: str = "pcp") -> str:
    slot_map = _pcp_map(entity)
    handler = slot_map.get(slot, lambda: "okay")
    return handler()
