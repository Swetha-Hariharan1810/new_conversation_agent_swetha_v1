"""
Maps slot labels to expected user responses for the PCP evaluation flow.

Scenario-specific overrides handle clarification re-asks and mid-conversation
corrections. turn_counters tracks how many times each (scenario_tag, slot)
pair has been visited within a single run so the override can return a
different response on the second visit.
"""

from __future__ import annotations


def _pcp_map(entity) -> dict:
    provider = getattr(entity, "provider_type", None) or "Primary Care Physician"
    fax = getattr(entity, "fax_number", None) or "6175554199"
    zip_code = getattr(entity, "zip_code", None) or "12139"

    return {
        "intent_selection": lambda: "Can I get a list of in-network providers, please?",
        "first_name": lambda: entity.first_name,
        "last_name": lambda: entity.last_name,
        "member_id": lambda: entity.member_id,
        "dob": lambda: entity.date_of_birth,
        "subscriber_confirm": lambda: "I'm calling for myself.",
        "provider_type": lambda: f"I'm looking for a {provider}.",
        "zip_confirm": lambda: "Yes, that's correct.",
        "zip_update": lambda: zip_code,
        "fax_or_email": lambda: "Please send it to my fax.",
        "fax_confirm": lambda: "Yes, that fax number is correct.",
        "fax_update": lambda: fax,
        "benefits_offer": lambda: "Yes please.",
        "benefits_explanation": lambda: "Yes please.",
        "coach_offer": lambda: "Yes, that sounds interesting.",
        "follow_up": lambda: "Can you summarize the PCP benefits?",
        "closing": lambda: "No, that was very helpful. Thank you.",
        "clarification": lambda: f"Sorry about that — {entity.member_id}.",
        "correction_ack": lambda: "Yes, that's right now. Thank you.",
        "other": lambda: "Okay.",
    }


def ground_truth_for_slot(
    slot: str,
    entity,
    flow: str = "pcp",
    scenario_tag: str = "",
    turn_counters: dict | None = None,
) -> str:
    if turn_counters is None:
        turn_counters = {}

    visit = turn_counters.get((scenario_tag, slot), 0)

    # ── pcp_clarification_zip ──────────────────────────────────────────────
    if scenario_tag == "pcp_clarification_zip":
        if slot == "zip_confirm":
            return "Hmm, let me check." if visit == 0 else "Yes, that's correct."

    # ── pcp_correction_first_name ──────────────────────────────────────────
    if scenario_tag == "pcp_correction_first_name":
        if slot == "last_name" and visit == 0:
            return "Actually my first name is Emilia, E-M-I-L-I-A. Last name is Carter."
        if slot == "correction_ack" and visit == 0:
            return entity.member_id

    # ── pcp_correction_member_id ───────────────────────────────────────────
    if scenario_tag == "pcp_correction_member_id":
        if slot == "member_id" and visit == 0:
            return "m nine oh seven five oh two"
        if slot == "dob" and visit == 0:
            return "Sorry, that's wrong. It's m nine zero seven five zero three."
        if slot == "correction_ack" and visit == 0:
            return f"April twelfth {entity.date_of_birth[:4]}"

    # ── pcp_clarification_fax ──────────────────────────────────────────────
    if scenario_tag == "pcp_clarification_fax":
        if slot == "fax_confirm":
            return (
                "I'm not sure if that's the right one."
                if visit == 0
                else "Yes, that is the correct fax number."
            )

    # ── Default ────────────────────────────────────────────────────────────
    pcp_map = _pcp_map(entity)
    handler = pcp_map.get(slot, lambda: "Okay.")
    return handler()
