"""
slot_ownership.py — which part of the system can honor a caller-requested
slot update (Phase 3; the SLOT_OWNERSHIP registry the Phase-4 routing uses).

Consumers:
  * slot_manager._handle_answered_followup — an update request the CURRENT
    pipeline cannot honor is parked as a kind="action" item (instead of being
    declined) whenever the registry names an owner for the slot.
  * follow_up.agent — parked kind="action" items are routed to the owning
    flow; only OWNER_HUMAN slots escalate to a representative.

Keep this module dependency-free (core ↔ agents import safety).
"""

from __future__ import annotations

# Sentinel owners. Values that are neither of these are intake intents — the
# flow re-run via reset_for_new_intent re-collects the slot.
OWNER_HUMAN = "human"
OWNER_VERIFICATION = "verification"

# slot name → owner. Unknown slots default to OWNER_HUMAN (safe: never promise
# an update the system cannot perform).
SLOT_OWNERSHIP: dict[str, str] = {
    # Identity slots — re-verification re-collects them from scratch.
    "first_name": OWNER_VERIFICATION,
    "last_name": OWNER_VERIFICATION,
    "member_id": OWNER_VERIFICATION,
    "dob": OWNER_VERIFICATION,
    "relationship": OWNER_VERIFICATION,
    # Claim notification preferences — collected by notification_setup inside
    # the claims flow.
    "notification_method": "claim_services",
    "n2_notification_method": "claim_services",
    # Salesforce-record contact fields — disputes go to a human
    # (mirrors verification.handlers.CALLER_LOCKED_SLOTS).
    "phone_number": OWNER_HUMAN,
    "phone": OWNER_HUMAN,
    "email": OWNER_HUMAN,
    "zip_code": OWNER_HUMAN,
    "fax": OWNER_HUMAN,
}


def slot_update_owner(slot: str) -> str:
    """Owner label for a caller-requested update to ``slot``.

    Returns OWNER_VERIFICATION, an intake intent, or OWNER_HUMAN. Slots the
    registry does not know are OWNER_HUMAN — never route an update nowhere.
    """
    return SLOT_OWNERSHIP.get((slot or "").strip().lower(), OWNER_HUMAN)
