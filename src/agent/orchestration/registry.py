"""
registry.py — Conversation-wide agent / slot / owner / artifact registry (Phase 3D).

Single source of truth for "who owns what" across every agent, so the shared
resolver can route a correction or a parked secondary to the right owner
regardless of which agent is active. Phases 1–3C grew several small, scattered
owner maps (invalidation.INTENT_OWNER_REGISTRY, resolver.KNOWN_AGENTS,
shadow._FIELD_OWNERS); 3D consolidates them here and derives the rest.

To onboard an agent: add its row to ``AGENT_SLOTS`` (the slots it collects) and,
if it produces a downstream artifact, ``AGENT_ARTIFACTS`` + any ``INVALIDATION_EDGES``.
Everything else (slot→owner, artifact→owner, known-agent set, invalidation map)
is derived below.
"""

from __future__ import annotations

# Each agent → the slots it collects. Order matters only for shared-field owner
# resolution (first declarer wins), so the canonical writer is listed first.
AGENT_SLOTS: dict[str, list[str]] = {
    "verification_agent": [
        "first_name",
        "last_name",
        "member_id",
        "dob",
        "relationship",
        "name_confirmed",
        "phone_confirmed",
    ],
    "provider_search_agent": ["provider_type", "zip_code", "zip_confirmed"],
    "delivery_management_agent": [
        "delivery_method",
        "fax",
        "fax_confirmed",
        "email",
        "email_confirmed",
        "benefits_response",
    ],
    "benefits_agent": [],
    "care_wellness_agent": ["care_coach_response"],
    "claim_adjustment_agent": ["reference_number"],
    "records_coordination_agent": ["upload_method", "upload_consent", "personal_guide_consent"],
    "notification_setup_agent": ["notification_method", "phone", "timeline_question"],
    "follow_up_agent": [],
}

# Each agent → downstream artifact(s) it produces/sends.
AGENT_ARTIFACTS: dict[str, list[str]] = {
    "delivery_management_agent": ["provider_list"],
}

# Upstream owner-field → downstream artifact(s) invalidated when it is disputed.
INVALIDATION_EDGES: dict[str, list[str]] = {
    "zip_code": ["provider_list"],
}

ALL_AGENTS: frozenset[str] = frozenset(AGENT_SLOTS)


def _build_slot_owners() -> dict[str, str]:
    owners: dict[str, str] = {}
    for agent, slots in AGENT_SLOTS.items():
        for slot in slots:
            owners.setdefault(slot, agent)  # first declarer wins (canonical writer)
    return owners


SLOT_OWNERS: dict[str, str] = _build_slot_owners()

ARTIFACT_OWNERS: dict[str, str] = {
    artifact: agent for agent, artifacts in AGENT_ARTIFACTS.items() for artifact in artifacts
}

# Field / artifact → responsible agent (consumed by the resolver + invalidation).
INTENT_OWNER_REGISTRY: dict[str, str] = {**SLOT_OWNERS, **ARTIFACT_OWNERS}

# Upstream field → downstream artifacts it invalidates.
INVALIDATION_MAP: dict[str, list[str]] = dict(INVALIDATION_EDGES)


def owner_of(name: str) -> str | None:
    """Agent responsible for an owner field or artifact, if registered."""
    return INTENT_OWNER_REGISTRY.get(name)


def slots_for(agent: str) -> list[str]:
    return list(AGENT_SLOTS.get(agent, []))
