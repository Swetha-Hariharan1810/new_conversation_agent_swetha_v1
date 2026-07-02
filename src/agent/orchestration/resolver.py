"""
resolver.py — Deterministic TurnPlan resolver (Phase 3, pure Python).

Takes a ``TurnPlan`` (the multi-intent understanding decode) + the current
``State`` and decides what the turn does, with NO LLM call and NO improvised
text. Every member-facing sentence is later rendered from a single closed set of
speech-act labels — the resolver only *selects* one; it never writes prose.

Guarantees enforced here (all deterministic):
  * ``slot_answer`` is accepted only if it passes the existing normalizer +
    validator for ``awaiting_slot``;
  * a ``secondary_intent`` is dropped unless its ``verbatim_span`` actually occurs
    in this turn's utterance (anti-hallucination) and its ``owner`` resolves in
    the registry;
  * a ``correction`` whose ``owner`` does not resolve is rejected;
  * precedence: safety > invalidating_correction > current-step completion >
    parked independents > closure;
  * surviving independents are enqueued into ``intent_queue``;
  * an invalidating correction flips ``dirty_artifacts`` via invalidation.py and
    sets a rewind target (the owner/step to return to);
  * low-confidence / absent-span / unknown turns route to ``clarify`` /
    ``open_redirect`` — ask, never act.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from agent.llm.schema import GuardType, SecondaryIntentType, TurnPlan
from agent.orchestration.invalidation import (
    INVALIDATION_MAP,
    artifacts_invalidated_by,
    mark_dirty,
    owner_of,
)
from agent.orchestration.registry import (
    AGENT_SLOTS,
    ALL_AGENTS,
    queue_entry,
    queue_entry_owner,
)
from agent.slots import normalizers as _N
from agent.slots import validators as _V

# ── Closed speech-act vocabulary (the ONLY member-facing response labels) ──────
RE_ASK = "re_ask"
CLARIFY = "clarify"
CORRECTION_ACK = "correction_ack"
UNSUPPORTED_DECLINE = "unsupported_decline"
MULTI_INTENT_ACK = "multi_intent_ack"
OPEN_REDIRECT = "open_redirect"
# Phase 2: the utterance answered a DIFFERENT pending slot of the active agent —
# accept that value (no attempt counted on the awaiting slot), then re-ask.
CROSS_SLOT_ACCEPT = "cross_slot_accept"

SPEECH_ACTS = frozenset(
    {
        RE_ASK,
        CLARIFY,
        CORRECTION_ACK,
        UNSUPPORTED_DECLINE,
        MULTI_INTENT_ACK,
        OPEN_REDIRECT,
        CROSS_SLOT_ACCEPT,
    }
)

# Below this, the decode is too uncertain to act on — ask instead.
CONFIDENCE_THRESHOLD = 0.5

# Known routable agents — sourced from the conversation-wide registry (Phase 3D),
# plus the two non-slot terminals (intake/escalation/closure) the resolver may
# route to. Kept as a leaf import (registry has no heavy deps).
KNOWN_AGENTS = ALL_AGENTS | {"intake_agent", "escalation_agent", "closure_agent"}

# slot name → (normalizer, validator), reusing the existing deterministic functions.
SLOT_SPEC = {
    "first_name": (_N.normalize_name, _V.validate_name),
    "last_name": (_N.normalize_name, _V.validate_name),
    "member_id": (_N.normalize_member_id, _V.validate_member_id),
    "dob": (_N.normalize_dob, _V.validate_dob),
    "zip_code": (_N.normalize_zip_code, _V.validate_zip_code),
    "phone_number": (_N.normalize_phone_number, _V.validate_phone_number),
    "fax": (_N.normalize_fax_number, _V.validate_fax_number),
    "email": (_N.normalize_email, _V.validate_email),
    "provider_type": (_N.normalize_provider_type, _V.validate_provider_type),
    "delivery_method": (_N.normalize_delivery_method, _V.validate_delivery_method),
    "reference_number": (_N.normalize_reference_number, _V.validate_reference_number),
    "notification_method": (_N.normalize_notification_method, _V.validate_notification_method),
    "relationship": (_N.normalize_caller_role, _V.validate_relationship),
    # yes/no confirmation slots (handled inline by agents today)
    "zip_confirmed": (_N.normalize_yes_no, _V.validate_yes_no),
    "fax_confirmed": (_N.normalize_yes_no, _V.validate_yes_no),
    "email_confirmed": (_N.normalize_yes_no, _V.validate_yes_no),
    "phone_confirmed": (_N.normalize_yes_no, _V.validate_yes_no),
    "benefits_response": (_N.normalize_yes_no, _V.validate_yes_no),
}

_SAFETY_GUARDS = {GuardType.ABUSE, GuardType.SELF_HARM}


@dataclass
class ResolverOutcome:
    speech_act: Optional[str]  # closed set, or None when safety/escalation dominates
    state_updates: dict = field(default_factory=dict)
    rewind_target: Optional[str] = None  # owner/agent to return to for an invalidating correction
    parked: list = field(default_factory=list)  # resolved owners of independents enqueued
    dirty: dict = field(default_factory=dict)  # dirty_artifacts delta
    # Unsupported/out-of-scope secondaries that co-occurred with a winning act.
    # They get a decline appended to the dominant act's speech, so a multi-intent
    # turn never silently drops an unanswerable request (S5).
    declined: list = field(default_factory=list)
    # Phase 3: per surviving in-scope independent — {owner, span, answer, answerable}.
    # Lets the generator answer a relevant question inline (grounded) or park it.
    independents_detail: list = field(default_factory=list)

    def to_log_dict(self) -> dict:
        """PII-safe summary for shadow logging (no verbatim spans)."""
        return {
            "speech_act": self.speech_act,
            "rewind_target": self.rewind_target,
            "parked": list(self.parked),
            "dirty": [k for k, v in self.dirty.items() if v],
            "declined": list(self.declined),
            "state_update_keys": sorted(self.state_updates.keys()),
        }


# ── Pure helpers ──────────────────────────────────────────────────────────────


def resolve_owner(owner: Optional[str]) -> Optional[str]:
    """Resolve an owner string to an agent name, or None if it doesn't resolve.

    Accepts an agent name directly, or an owner field/artifact that maps to an
    agent via INTENT_OWNER_REGISTRY.
    """
    if not owner:
        return None
    o = owner.strip()
    if o in KNOWN_AGENTS:
        return o
    mapped = owner_of(o)
    if mapped in KNOWN_AGENTS:
        return mapped
    return None


def _span_in_utterance(span: Optional[str], utterance: str) -> bool:
    """Deterministic substring check (case-insensitive to tolerate ASR casing)."""
    if not span:
        return False
    return span.strip().lower() in (utterance or "").lower()


def validate_slot_answer(awaiting_slot: str, answer: Optional[str]) -> tuple[bool, Optional[str]]:
    """Validate a proposed slot answer with the existing normalizer + validator.
    Returns (ok, normalized_value)."""
    if not answer or not awaiting_slot:
        return (False, None)
    spec = SLOT_SPEC.get(awaiting_slot)
    if not spec:
        return (False, None)  # unknown slot → cannot validate → not accepted
    normalizer, validator = spec
    normalized = normalizer(answer)
    if not normalized:
        return (False, None)
    result = validator(normalized)
    ok = result.valid if hasattr(result, "valid") else bool(result)
    return (ok, normalized) if ok else (False, None)


def _invalidating_field_for_owner(agent: Optional[str]) -> Optional[str]:
    """Reverse-lookup: a depended-on field owned by ``agent`` that invalidates a
    downstream artifact (e.g. provider_search_agent → zip_code)."""
    if not agent:
        return None
    for fld, artifacts in INVALIDATION_MAP.items():
        if artifacts and owner_of(fld) == agent:
            return fld
    return None


# Free-text-permissive slots (validate_name accepts any short alphabetic text):
# a cross-slot match against these may come only from the STRUCTURED
# ``plan.slot_answer``, never the raw utterance, or any rambling non-answer
# would "validate" as a name.
_FREE_TEXT_SLOTS = frozenset({"first_name", "last_name"})


def _cross_slot_match(plan: TurnPlan, state: dict, awaiting: str, *, utterance: str) -> Optional[object]:
    """Does this non-answer actually answer a DIFFERENT pending slot of the
    active agent? Returns ``(slot, normalized_value)`` on exactly one match,
    the string ``"ambiguous"`` when several pending slots validate (never
    guess), or ``None`` when nothing matches.

    Candidates tried per slot: the structured ``plan.slot_answer`` first, then
    the raw utterance (strict-validator slots only). Yes/no confirmation slots
    are excluded — a bare "yes" out of context must never confirm a value that
    was not read back this turn.
    """
    if not awaiting:
        return None
    agent = state.get("active_agent") or ""
    matches: dict[str, str] = {}
    for slot in AGENT_SLOTS.get(agent, []):
        if slot == awaiting or (state.get(slot) or ""):
            continue  # not this agent's other slot, or already filled
        spec = SLOT_SPEC.get(slot)
        if not spec or spec[0] is _N.normalize_yes_no:
            continue
        candidates = [plan.slot_answer]
        if slot not in _FREE_TEXT_SLOTS:
            candidates.append(utterance)
        for candidate in candidates:
            ok, normalized = validate_slot_answer(slot, candidate)
            if ok:
                matches[slot] = normalized
                break
    if len(matches) > 1:
        return "ambiguous"
    if matches:
        return next(iter(matches.items()))
    return None


def _park(independents: list, state: dict) -> tuple[list, list]:
    """Enqueue each independent into intent_queue (dedup by owner, order-
    preserving). Entries carry the caller's verbatim span alongside the owner
    ({"owner": …, "span": …}) so draining can acknowledge the parked request in
    the caller's own words; legacy bare-string entries are left untouched.
    Returns (parked_owners, new_queue)."""
    queue = list(state.get("intent_queue") or [])
    owners_in_queue = {queue_entry_owner(e) for e in queue}
    parked: list = []
    for si, owner in independents:
        parked.append(owner)
        if owner not in owners_in_queue:
            queue.append(queue_entry(owner, si.verbatim_span or ""))
            owners_in_queue.add(owner)
    return parked, queue


# ── Main entry point ──────────────────────────────────────────────────────────


def resolve_turn(plan: TurnPlan, state: dict, *, utterance: str) -> ResolverOutcome:  # noqa: C901
    awaiting = state.get("awaiting_slot") or ""
    utter = utterance or ""

    # 1. Filter secondary intents by verbatim span + owner resolution.
    survivors: list = []  # (SecondaryIntent, resolved_owner)
    dropped_for_span = False
    dropped_for_owner = False
    for si in plan.secondary_intents:
        if not _span_in_utterance(si.verbatim_span, utter):
            dropped_for_span = True
            continue
        resolved = resolve_owner(si.owner)
        needs_owner = si.type in (
            SecondaryIntentType.IN_SCOPE_INDEPENDENT,
            SecondaryIntentType.INVALIDATING_CORRECTION,
        )
        if needs_owner and not resolved:
            dropped_for_owner = True
            continue
        if si.owner and not resolved and not needs_owner:
            dropped_for_owner = True
            continue
        survivors.append((si, resolved))

    # 2. Correction owner rejection.
    correction = plan.correction
    correction_owner = resolve_owner(correction.owner) if correction else None
    if correction and not correction_owner:
        correction = None

    # 3. Slot-answer validation (existing normalizer + validator).
    slot_ok, slot_norm = validate_slot_answer(awaiting, plan.slot_answer)

    # ── Precedence ladder ──────────────────────────────────────────────────────

    # (a) SAFETY — escalation dominates; no member speech-act from the closed set.
    safety_secondary = any(si.type == SecondaryIntentType.SAFETY for si, _ in survivors)
    if (plan.guard in _SAFETY_GUARDS and plan.guard_confidence >= 0.7) or safety_secondary:
        return ResolverOutcome(
            speech_act=None,
            state_updates={"escalate": True, "escalation_reason": "safety"},
        )

    # (b) Low confidence / unknown / hallucinated-only → ask, never act.
    if plan.confidence < CONFIDENCE_THRESHOLD:
        return ResolverOutcome(CLARIFY if awaiting else OPEN_REDIRECT)
    if any(si.type == SecondaryIntentType.UNKNOWN for si, _ in survivors):
        return ResolverOutcome(CLARIFY if awaiting else OPEN_REDIRECT)

    independents = [(si, own) for si, own in survivors if si.type == SecondaryIntentType.IN_SCOPE_INDEPENDENT]
    unsupported = [
        si
        for si, _ in survivors
        if si.type in (SecondaryIntentType.OUT_OF_SCOPE, SecondaryIntentType.IN_DOMAIN_UNSUPPORTED)
    ]
    inv_secondaries = [
        (si, own) for si, own in survivors if si.type == SecondaryIntentType.INVALIDATING_CORRECTION
    ]

    # (c) INVALIDATING CORRECTION — refuse to build on a disputed value; rewind.
    inv_field: Optional[str] = None
    inv_owner: Optional[str] = None
    if correction is not None and artifacts_invalidated_by(correction.field):
        inv_field, inv_owner = correction.field, correction_owner
    elif inv_secondaries:
        si, own = inv_secondaries[0]
        derived = _invalidating_field_for_owner(own)
        if derived:
            inv_field, inv_owner = derived, own

    # Unsupported/out-of-scope survivors are declined alongside the winning act
    # (so a multi-intent turn never silently drops an unanswerable request).
    declined = [si.type.value for si in unsupported]

    # Per-independent detail for the Phase 3 generator composition: the resolved
    # owner, the verbatim span, and any grounded inline answer the decode produced
    # from the session snapshot. The compose layer (which reads PARK_ANSWERABLE)
    # decides inline-answer vs. park; the resolver stays pure and always parks.
    independents_detail = [
        {
            "owner": own,
            "span": si.verbatim_span,
            "answer": (getattr(si, "answer", None) or "").strip(),
            "answerable": bool(
                getattr(si, "answerable_from_snapshot", False) and (getattr(si, "answer", None) or "").strip()
            ),
        }
        for si, own in independents
    ]

    if inv_field:
        updates: dict = {}
        if slot_ok:
            updates[awaiting] = slot_norm
        dirty = mark_dirty(state.get("dirty_artifacts"), inv_field)
        updates["dirty_artifacts"] = dirty
        parked, queue = _park(independents, state)
        if parked:
            updates["intent_queue"] = queue
        return ResolverOutcome(
            CORRECTION_ACK,
            updates,
            rewind_target=inv_owner,
            parked=parked,
            dirty=dirty,
            declined=declined,
            independents_detail=independents_detail,
        )

    # (d) Non-invalidating correction — acknowledge + rewind to its owner.
    if correction is not None:
        updates = {}
        if slot_ok:
            updates[awaiting] = slot_norm
        parked, queue = _park(independents, state)
        if parked:
            updates["intent_queue"] = queue
        return ResolverOutcome(
            CORRECTION_ACK,
            updates,
            rewind_target=correction_owner,
            parked=parked,
            declined=declined,
            independents_detail=independents_detail,
        )

    # (e) Current-step completion (slot answered cleanly).
    if slot_ok:
        updates = {awaiting: slot_norm}
        if independents:
            parked, queue = _park(independents, state)
            updates["intent_queue"] = queue
            return ResolverOutcome(
                MULTI_INTENT_ACK,
                updates,
                parked=parked,
                declined=declined,
                independents_detail=independents_detail,
            )
        if unsupported:
            return ResolverOutcome(UNSUPPORTED_DECLINE, updates, declined=declined)
        # Clean single-intent answer — nothing for the resolver to add; proceed.
        return ResolverOutcome(None, updates)

    # (f) Slot NOT answered.
    if independents:
        parked, queue = _park(independents, state)
        return ResolverOutcome(
            MULTI_INTENT_ACK,
            {"intent_queue": queue},
            parked=parked,
            declined=declined,
            independents_detail=independents_detail,
        )
    if unsupported:
        return ResolverOutcome(UNSUPPORTED_DECLINE)
    if dropped_for_span or dropped_for_owner:
        # We had a secondary signal we could not safely act on — ask, never act.
        return ResolverOutcome(CLARIFY if awaiting else OPEN_REDIRECT)

    # (g) Cross-slot answer (Phase 2, Bug 1's trigger): before treating this as a
    # genuine non-answer, check whether it validates against exactly one OTHER
    # pending slot of the active agent. If so, accept that value and re-ask the
    # original awaiting slot — no failed attempt is counted on it. Multiple
    # plausible slots → CLARIFY (never guess).
    cross = _cross_slot_match(plan, state, awaiting, utterance=utter)
    if cross == "ambiguous":
        return ResolverOutcome(CLARIFY)
    if cross:
        slot, value = cross
        return ResolverOutcome(CROSS_SLOT_ACCEPT, {slot: value})

    # (h) Genuine non-answer.
    return ResolverOutcome(RE_ASK if awaiting else OPEN_REDIRECT)
