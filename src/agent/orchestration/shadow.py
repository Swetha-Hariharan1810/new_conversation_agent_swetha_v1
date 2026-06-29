"""
shadow.py — Phase 3A shadow-mode runner for the TurnPlan understanding decode.

Installs the understanding decode + deterministic resolver at the shared
slot-collection chokepoint (``_collect_slot``) so it runs on every slot turn in
every agent — but ONLY logs its decision. The live path keeps running the
existing logic, so there is zero behavior change.

A *decoder* turns (state, utterance, decision) into a ``TurnPlan``. It is
pluggable:

  * Production default: ``None`` — shadow is a complete no-op (no cost, no risk)
    until a decoder is explicitly installed (the LLM decode arrives in Phase 3B).
  * Tests / non-LLM runs: ``heuristic_decoder`` recovers the multi-intent shape
    deterministically from the raw utterance + the existing WorkerResult — the
    very signals today's single-intent path drops — proving the resolver would
    catch them.

``run_shadow`` is wrapped by the caller in try/except and never feeds anything
back into the live turn.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from agent.llm.schema import (
    Correction,
    GuardType,
    SecondaryIntent,
    SecondaryIntentType,
    TurnPlan,
)
from agent.logger import get_logger
from agent.orchestration.invalidation import artifacts_invalidated_by, owner_of
from agent.orchestration.observability import (
    detect_secondary_request,
    matched_span,
    secondary_target,
)
from agent.orchestration.resolver import ResolverOutcome, resolve_turn

logger = get_logger(__name__)

# Decoder: (state, utterance, decision) -> TurnPlan | None
Decoder = Callable[[dict, str, Any], Optional[TurnPlan]]

_decoder: Optional[Decoder] = None


def set_shadow_decoder(decoder: Optional[Decoder]) -> None:
    """Install (or clear, with None) the shadow decoder. Default is None = off."""
    global _decoder
    _decoder = decoder


def clear_shadow_decoder() -> None:
    set_shadow_decoder(None)


def get_shadow_decoder() -> Optional[Decoder]:
    return _decoder


# ── Default deterministic decoder (no LLM) ─────────────────────────────────────

# Which agent owns a corrected identity/contact field, for correction routing.
_FIELD_OWNERS = {
    "zip_code": "provider_search_agent",
    "fax": "delivery_management_agent",
    "email": "delivery_management_agent",
    "first_name": "verification_agent",
    "last_name": "verification_agent",
    "member_id": "verification_agent",
    "dob": "verification_agent",
    "phone_number": "verification_agent",
}


def _owner_for_field(field: str) -> str:
    return owner_of(field) or _FIELD_OWNERS.get(field, "")


def heuristic_decoder(state: dict, utterance: str, decision: Any) -> Optional[TurnPlan]:
    """Recover a TurnPlan deterministically from the utterance + WorkerResult.

    This is the non-LLM stand-in for the understanding decode: it deliberately
    re-reads the raw utterance (via the Phase 2 detector) to recover a secondary
    request that the single-intent WorkerResult dropped — exactly the UAT-007
    failure — and lifts any WorkerResult corrections into a structured Correction.
    """
    awaiting = state.get("awaiting_slot") or ""
    extracted = (getattr(decision, "extracted", None) or {}) if decision else {}
    corrections = (getattr(decision, "corrections", None) or {}) if decision else {}
    guard = getattr(decision, "guard", GuardType.NONE) if decision else GuardType.NONE
    guard_conf = getattr(decision, "guard_confidence", 0.0) if decision else 0.0

    slot_answer = extracted.get(awaiting) if awaiting else None

    correction: Optional[Correction] = None
    if corrections:
        fld = next(iter(corrections))
        correction = Correction(field=fld, owner=_owner_for_field(fld), new_value=corrections.get(fld))

    secondary_intents: list[SecondaryIntent] = []
    if detect_secondary_request(utterance):
        span = matched_span(utterance) or (utterance or "").strip()
        target = secondary_target(utterance)
        if target == "zip_code" or (correction and artifacts_invalidated_by(correction.field)):
            secondary_intents.append(
                SecondaryIntent(
                    type=SecondaryIntentType.INVALIDATING_CORRECTION,
                    owner="provider_search_agent",
                    verbatim_span=span,
                )
            )
            if correction is None and target == "zip_code":
                correction = Correction(field="zip_code", owner="provider_search_agent", new_value=None)
        elif target in ("fax", "email", "phone_number"):
            secondary_intents.append(
                SecondaryIntent(
                    type=SecondaryIntentType.IN_SCOPE_INDEPENDENT,
                    owner="delivery_management_agent",
                    verbatim_span=span,
                )
            )
        else:
            # Generic in-scope side question (e.g. "also what's my deductible?").
            secondary_intents.append(
                SecondaryIntent(
                    type=SecondaryIntentType.IN_SCOPE_INDEPENDENT,
                    owner="benefits_agent",
                    verbatim_span=span,
                )
            )

    if not (slot_answer or secondary_intents or correction or guard != GuardType.NONE):
        return None

    return TurnPlan(
        slot_answer=slot_answer,
        secondary_intents=secondary_intents,
        correction=correction,
        guard=guard,
        guard_confidence=float(guard_conf or 0.0),
        confidence=1.0,
    )


# ── Runner ─────────────────────────────────────────────────────────────────────


def run_shadow(
    state: dict,
    *,
    utterance: str,
    awaiting_slot: str,
    decision: Any = None,
    agent_name: str = "",
) -> Optional[ResolverOutcome]:
    """Run the understanding decode + resolver in shadow and log the decision.

    Returns the ResolverOutcome (for tests) or None when no decoder is installed
    or there is nothing to plan. NEVER mutates the live turn.
    """
    decoder = _decoder
    if decoder is None:
        return None

    plan = decoder(state, utterance, decision)
    if plan is None:
        return None

    outcome = resolve_turn(plan, {**state, "awaiting_slot": awaiting_slot}, utterance=utterance)

    logger.info(
        "turnplan_shadow",
        extra={
            "metric": "turnplan_shadow",
            "agent": agent_name,
            "awaiting_slot": awaiting_slot,
            "n_secondary": len(plan.secondary_intents),
            "has_correction": plan.correction is not None,
            **outcome.to_log_dict(),
        },
    )
    return outcome
