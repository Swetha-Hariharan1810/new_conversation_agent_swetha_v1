"""
pending_intents.py — shared model for everything a member still wants.

A turn can carry several requests. Each becomes a PendingIntent. The list is
persisted in LangGraph state under "pending_intents" as a list of plain dicts.
All reducers here are pure: they take and return a list of dicts and never
mutate the input.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Optional


class IntentKind(str, Enum):
    IN_SCOPE_INDEPENDENT = "in_scope_independent"
    IN_SCOPE_INVALIDATING = "in_scope_invalidating"
    OFF_TOPIC = "off_topic"
    UNSUPPORTED = "unsupported"
    SAFETY = "safety"


class IntentStatus(str, Enum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


# Which committed value is owned by which agent. Used to rewind a correction.
CORRECTION_OWNER: dict[str, str] = {
    "zip_code": "provider_search_agent",
    "fax": "delivery_management_agent",
    "email": "delivery_management_agent",
    "phone_number": "verification_agent",
}

# Which agent handles a fresh independent in scope request, by intent label.
INTENT_AGENT: dict[str, str] = {
    "provider_services": "provider_search_agent",
    "benefits_inquiry": "benefits_agent",
    "care_wellness": "care_wellness_agent",
    "claim_services": "claim_adjustment_agent",
}

_IN_SCOPE = {IntentKind.IN_SCOPE_INDEPENDENT.value, IntentKind.IN_SCOPE_INVALIDATING.value}


@dataclass
class PendingIntent:
    kind: str
    raw_text: str
    target: Optional[str] = None
    status: str = IntentStatus.OPEN.value

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PendingIntent":
        return cls(
            kind=d["kind"],
            raw_text=d.get("raw_text", ""),
            target=d.get("target"),
            status=d.get("status", IntentStatus.OPEN.value),
        )


def _norm_text(raw: Optional[str]) -> str:
    return " ".join((raw or "").lower().split())


def _dedup_key(d: dict) -> tuple:
    """Identity used to collapse duplicates.

    A correction is identified by (kind, target): "change the fax" raised twice
    in one episode is one item regardless of wording. A targetless intent
    (independent / off_topic / unsupported / safety) has no natural identity, so
    its normalised raw_text is part of the key — two DISTINCT requests in one
    turn (e.g. "benefits_inquiry" and "care_wellness") must stay separate.
    """
    target = d.get("target")
    if target:
        return (d.get("kind"), target)
    return (d.get("kind"), None, _norm_text(d.get("raw_text")))


def add_intent(intents: list[dict], intent: PendingIntent) -> list[dict]:
    """Append, collapsing duplicates by kind, target, and normalised text.

    Corrections collapse by (kind, target) so repeated "change the fax" within
    one episode stays a single item. Targetless intents also key on normalised
    raw_text so two distinct requests in one turn are both captured.

    Only collapses against still-open (unresolved) items so repeats within one
    episode stay a single item. If the matching item was already resolved, a
    repeat of the same intent must re-fire: the resolved item is re-opened
    rather than silently dropped.
    """
    new = intent.to_dict()
    new_key = _dedup_key(new)
    out = []
    collapsed = False
    for existing in intents:
        if not collapsed and _dedup_key(existing) == new_key:
            collapsed = True
            if existing.get("status") == IntentStatus.RESOLVED.value:
                # Repeat of an already-resolved intent: re-open it.
                out.append({**existing, **new, "status": IntentStatus.OPEN.value})
                continue
            # Still open: keep the existing item, drop the duplicate.
            out.append(existing)
            continue
        out.append(existing)
    if collapsed:
        return out
    return [*intents, new]


def next_open_correction(intents: list[dict]) -> Optional[dict]:
    for d in intents:
        if (
            d.get("kind") == IntentKind.IN_SCOPE_INVALIDATING.value
            and d.get("status") == IntentStatus.OPEN.value
        ):
            return d
    return None


def next_open_independent(intents: list[dict]) -> Optional[dict]:
    for d in intents:
        if (
            d.get("kind") == IntentKind.IN_SCOPE_INDEPENDENT.value
            and d.get("status") == IntentStatus.OPEN.value
        ):
            return d
    return None


def all_in_scope_resolved(intents: list[dict]) -> bool:
    for d in intents:
        if d.get("kind") in _IN_SCOPE and d.get("status") != IntentStatus.RESOLVED.value:
            return False
    return True


def mark_resolved_by_target(intents: list[dict], target: str) -> list[dict]:
    out = []
    for d in intents:
        if d.get("target") == target and d.get("status") != IntentStatus.RESOLVED.value:
            d = {**d, "status": IntentStatus.RESOLVED.value}
        out.append(d)
    return out


def mark_resolved_by_kind(intents: list[dict], kind: str) -> list[dict]:
    out = []
    for d in intents:
        if d.get("kind") == kind and d.get("status") != IntentStatus.RESOLVED.value:
            d = {**d, "status": IntentStatus.RESOLVED.value}
        out.append(d)
    return out


def mark_independent_resolved_for_agent(intents: list[dict], agent_name: str) -> list[dict]:
    """Resolve any open in_scope_independent intent handled by ``agent_name``.

    A drained independent intent carries its intent label in ``raw_text`` and
    ``get_drain_route`` maps that label to a handler via ``INTENT_AGENT``. When
    that handler completes, the intent it just served must move to resolved;
    otherwise ``all_in_scope_resolved`` never becomes True and ``get_drain_route``
    keeps re-routing to the same handler. Resolving only the intents owned by
    this agent leaves other agents' independent intents open for their own drain.
    """
    out = []
    for d in intents:
        is_independent = d.get("kind") == IntentKind.IN_SCOPE_INDEPENDENT.value
        is_open = d.get("status") != IntentStatus.RESOLVED.value
        handled_here = INTENT_AGENT.get((d.get("raw_text") or "").strip()) == agent_name
        if is_independent and is_open and handled_here:
            d = {**d, "status": IntentStatus.RESOLVED.value}
        out.append(d)
    return out
