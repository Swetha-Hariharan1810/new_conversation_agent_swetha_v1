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

import re

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
    # Claim flow: the upload link and the Personal Guide outreach are both keyed
    # on the claim reference number, so they are invalidated if it is disputed.
    "records_coordination_agent": ["upload_link", "personal_guide_outreach"],
}

# Upstream owner-field → downstream artifact(s) invalidated when it is disputed.
INVALIDATION_EDGES: dict[str, list[str]] = {
    "zip_code": ["provider_list"],
    # A disputed claim reference number must not be acted on (send link / trigger
    # provider outreach) until it is re-resolved — the claim-flow analog of the
    # stale-delivery guard.
    "reference_number": ["upload_link", "personal_guide_outreach"],
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


# ── Intent-phrase vocabulary (Phase 3: never guess an owner) ────────────────────
# Deterministic keyword STEMS → owning agent. Stems match at a word boundary and
# extend through the word ("reimburs" → "reimbursement", "claim" → "claims"), so
# spoken variants resolve without a model call. A phrase that matches NOTHING
# here has no owner — the decoder must emit UNKNOWN and downstream asks, it
# never routes to "the closest" agent.
INTENT_PHRASES: dict[str, str] = {
    # claims / billing / money back
    "refund": "claim_adjustment_agent",
    "bill": "claim_adjustment_agent",
    "billing": "claim_adjustment_agent",
    "charge": "claim_adjustment_agent",
    "claim": "claim_adjustment_agent",
    "reimburs": "claim_adjustment_agent",
    # benefits / plan coverage
    "deductible": "benefits_agent",
    "copay": "benefits_agent",
    "coverage": "benefits_agent",
    "benefit": "benefits_agent",
    # medical records
    "records": "records_coordination_agent",
    "medical record": "records_coordination_agent",
    "upload": "records_coordination_agent",
    # notifications
    "notify": "notification_setup_agent",
    "notification": "notification_setup_agent",
    "text me": "notification_setup_agent",
    "sms": "notification_setup_agent",
    # provider search
    "provider": "provider_search_agent",
    "doctor": "provider_search_agent",
    "specialist": "provider_search_agent",
    "physician": "provider_search_agent",
    # delivery of the provider list
    "fax": "delivery_management_agent",
    "email": "delivery_management_agent",
    "delivery": "delivery_management_agent",
}


def owner_for_phrase(text: str) -> str | None:
    """Deterministic owner for a side-request phrase, or None (never guess).

    Longest matching stem wins; matching is case-insensitive and anchored at a
    word boundary (the stem may continue through the word, so "reimburs"
    matches "reimbursement"). Returns None when no stem matches — the caller
    must treat that as UNKNOWN, not pick a default owner.
    """
    lowered = (text or "").lower()
    if not lowered:
        return None
    best: tuple[int, str] | None = None
    for stem, owner in INTENT_PHRASES.items():
        if re.search(rf"\b{re.escape(stem)}", lowered) and (best is None or len(stem) > best[0]):
            best = (len(stem), owner)
    return best[1] if best else None


# ── Intent-queue entry shape (Phase 3: park the caller's own words) ─────────────
# ``intent_queue`` entries are {"owner": <agent>, "span": <verbatim user words>}
# so draining can acknowledge the parked request in the caller's own words.
# Bare-string entries (pre-Phase-3 checkpoints) are accepted everywhere.


def queue_entry(owner: str, span: str = "") -> dict:
    """Build an intent-queue entry: the owning agent + the caller's verbatim span."""
    return {"owner": owner, "span": span or ""}


def queue_entry_owner(entry) -> str:
    """Owner agent of a queue entry (dict entry or legacy bare string)."""
    if isinstance(entry, dict):
        return str(entry.get("owner") or "")
    return str(entry or "")


def queue_entry_span(entry) -> str:
    """Verbatim caller span of a queue entry ("" for legacy bare strings)."""
    if isinstance(entry, dict):
        return str(entry.get("span") or "")
    return ""


def queue_owners(queue) -> list[str]:
    """Owner agents of every entry in an intent queue (any entry shape)."""
    return [queue_entry_owner(e) for e in (queue or [])]
