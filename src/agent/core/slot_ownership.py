"""
slot_ownership.py — SLOT_OWNERSHIP registry: which agent owns each slot and
how a caller-requested update to it may be honored (Phase 4, fixes Bug C).

updatable modes:
  in_flow        — the owning agent's own pipeline collects/updates the slot;
                   when another agent is active, the update is parked as a
                   kind="action" item for follow_up (Phase 3 behavior).
  route_to_owner — the update is honored NOW: the current agent hands off to
                   the owner (pending_slot_update carries the way back).
                   Never park or say "later" for these.
  human_only     — only a representative can change it; decline honestly.

invalidates: state keys stale the moment the slot changes; a routing agent
clears them before the hand-off so the owner recomputes them (and dispatch
preconditions can block on them). Keys are literal State keys.

Consumers: core.slot_manager (resolve_update_target / routing / parking),
agents.delivery_management + provider_search (ZIP round-trip),
orchestration.fast_path (return hop), agents.follow_up (parked actions),
agents.verification.handlers (correction scoping).

Keep this module dependency-free (core ↔ agents import safety).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Updatable = Literal["in_flow", "route_to_owner", "human_only"]


@dataclass(frozen=True)
class SlotOwnership:
    agent: str
    updatable: Updatable
    invalidates: tuple[str, ...] = ()


SLOT_OWNERSHIP: dict[str, SlotOwnership] = {
    # ZIP drives the provider search — an update must re-run the search, so
    # the current provider list and the ZIP it was built from are invalidated.
    "zip_code": SlotOwnership(
        agent="provider_search_agent",
        updatable="route_to_owner",
        invalidates=("provider_list_sent", "zip_code_used"),
    ),
    # Delivery contact details — delivery_management's own confirmation flow
    # updates them inline.
    "fax": SlotOwnership(agent="delivery_management_agent", updatable="in_flow"),
    "email": SlotOwnership(agent="delivery_management_agent", updatable="in_flow"),
    # Identity slots — verification's existing correction/detour machinery
    # updates them in flow; elsewhere they park for a re-verify.
    "first_name": SlotOwnership(agent="verification_agent", updatable="in_flow"),
    "last_name": SlotOwnership(agent="verification_agent", updatable="in_flow"),
    "member_id": SlotOwnership(agent="verification_agent", updatable="in_flow"),
    "dob": SlotOwnership(agent="verification_agent", updatable="in_flow"),
    "relationship": SlotOwnership(agent="verification_agent", updatable="in_flow"),
    # Claim notification preferences — collected by notification_setup within
    # the claims flow.
    "notification_method": SlotOwnership(agent="notification_setup_agent", updatable="in_flow"),
    "n2_notification_method": SlotOwnership(agent="notification_setup_agent", updatable="in_flow"),
    # Human-only: system flags and SF-verified phone.
    "phone_number": SlotOwnership(agent="", updatable="human_only"),
    "member_status_verify": SlotOwnership(agent="", updatable="human_only"),
    "call_intent": SlotOwnership(agent="", updatable="human_only"),
}


def get_ownership(slot: str) -> SlotOwnership | None:
    """Registry entry for ``slot``, or None when unknown (treat as human-only)."""
    return SLOT_OWNERSHIP.get((slot or "").strip().lower())


def invalidated_state_updates(slot: str) -> dict:
    """State updates that clear everything an update to ``slot`` invalidates."""
    own = get_ownership(slot)
    if not own:
        return {}
    return {key: (False if key.endswith(("_sent", "_updated")) else "") for key in own.invalidates}


# ── follow_up compatibility layer (Phase 3 parked-action routing) ────────────
# follow_up routes parked kind="action" items by intake intent. Map each
# owning agent to the intent whose flow re-runs it; verification is special
# (re-verify with the CURRENT intent) and human-only stays human.
OWNER_HUMAN = "human"
OWNER_VERIFICATION = "verification"

_AGENT_TO_INTAKE_INTENT: dict[str, str] = {
    "provider_search_agent": "provider_services",
    "delivery_management_agent": "provider_services",
    "notification_setup_agent": "claim_services",
}


def slot_update_owner(slot: str) -> str:
    """Owner label for follow_up's parked-action routing.

    Returns OWNER_VERIFICATION, an intake intent, or OWNER_HUMAN. Unknown and
    human-only slots are OWNER_HUMAN — never route an update nowhere.
    """
    own = get_ownership(slot)
    if not own or own.updatable == "human_only":
        return OWNER_HUMAN
    if own.agent == "verification_agent":
        return OWNER_VERIFICATION
    return _AGENT_TO_INTAKE_INTENT.get(own.agent, OWNER_HUMAN)
