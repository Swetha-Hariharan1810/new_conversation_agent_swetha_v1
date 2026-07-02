"""
test_phase3a_resolver.py — exhaustive unit tests for the deterministic TurnPlan
resolver (Phase 3A). Pure functions; no LLM, no graph, no fixtures.
"""

from __future__ import annotations

import pytest

import tests.golden  # noqa: F401 — ensures src/ is on sys.path
from agent.llm.schema import (
    Correction,
    GuardType,
    SecondaryIntent,
    SecondaryIntentType,
    TurnPlan,
)
from agent.orchestration.resolver import (
    CLARIFY,
    CORRECTION_ACK,
    MULTI_INTENT_ACK,
    OPEN_REDIRECT,
    RE_ASK,
    UNSUPPORTED_DECLINE,
    resolve_owner,
    resolve_turn,
    validate_slot_answer,
)

pytestmark = pytest.mark.regression


def _state(awaiting="dob", **kw):
    base = {"awaiting_slot": awaiting, "dirty_artifacts": {}, "intent_queue": []}
    base.update(kw)
    return base


# ── pure helpers ────────────────────────────────────────────────────────────


def test_resolve_owner():
    assert resolve_owner("provider_search_agent") == "provider_search_agent"
    assert resolve_owner("zip_code") == "provider_search_agent"  # field → owner
    assert resolve_owner("provider_list") == "delivery_management_agent"
    assert resolve_owner("bogus") is None
    assert resolve_owner(None) is None
    assert resolve_owner("") is None


def test_validate_slot_answer():
    assert validate_slot_answer("dob", "09/03/1985") == (True, "09/03/1985")
    assert validate_slot_answer("dob", "not a date") == (False, None)
    assert validate_slot_answer("member_id", "m 714598")[0] is True
    assert validate_slot_answer("zip_code", "94107") == (True, "94107")
    assert validate_slot_answer("zip_code", "9410") == (False, None)
    assert validate_slot_answer("delivery_method", "fax") == (True, "fax")
    assert validate_slot_answer("unknown_slot", "whatever") == (False, None)
    assert validate_slot_answer("dob", None) == (False, None)


# ── slot-answer validation gates acceptance ─────────────────────────────────


def test_clean_slot_answer_accepted_and_proceeds():
    out = resolve_turn(TurnPlan(slot_answer="09/03/1985"), _state("dob"), utterance="September third 1985")
    assert out.speech_act is None  # nothing extra to say — proceed
    assert out.state_updates == {"dob": "09/03/1985"}


def test_invalid_slot_answer_triggers_re_ask():
    out = resolve_turn(TurnPlan(slot_answer="banana"), _state("dob"), utterance="banana")
    assert out.speech_act == RE_ASK
    assert "dob" not in out.state_updates


def test_genuine_non_answer_without_awaiting_is_open_redirect():
    out = resolve_turn(TurnPlan(), _state(awaiting=""), utterance="hi")
    assert out.speech_act == OPEN_REDIRECT


# ── span drop + owner rejection (anti-hallucination) ────────────────────────


def test_secondary_dropped_when_span_absent():
    si = SecondaryIntent(
        type=SecondaryIntentType.IN_SCOPE_INDEPENDENT,
        owner="benefits_agent",
        verbatim_span="what is my deductible",  # NOT in utterance
    )
    out = resolve_turn(TurnPlan(secondary_intents=[si]), _state("dob"), utterance="uh, hold on")
    # Secondary hallucinated → not acted on → ask, never act.
    assert out.speech_act == CLARIFY
    assert out.parked == []


def test_secondary_dropped_when_owner_unresolved():
    si = SecondaryIntent(
        type=SecondaryIntentType.IN_SCOPE_INDEPENDENT,
        owner="ghost_agent",
        verbatim_span="deductible",
    )
    out = resolve_turn(TurnPlan(secondary_intents=[si]), _state("dob"), utterance="my deductible")
    assert out.speech_act == CLARIFY
    assert out.parked == []


def test_correction_rejected_when_owner_unresolved():
    out = resolve_turn(
        TurnPlan(slot_answer="09/03/1985", correction=Correction(field="zip_code", owner="ghost")),
        _state("dob"),
        utterance="September third 1985",
    )
    # Correction dropped → behaves like a clean answer, no dirty flip.
    assert out.speech_act is None
    assert out.dirty == {}
    assert "dirty_artifacts" not in out.state_updates


# ── invalidating correction: dirty flip + rewind ────────────────────────────


def test_invalidating_correction_flips_dirty_and_rewinds():
    out = resolve_turn(
        TurnPlan(
            slot_answer="fax",
            correction=Correction(field="zip_code", owner="provider_search_agent"),
        ),
        _state("delivery_method"),
        utterance="Fax, but I need to update my ZIP code.",
    )
    assert out.speech_act == CORRECTION_ACK
    assert out.dirty == {"provider_list": True}
    assert out.rewind_target == "provider_search_agent"
    assert out.state_updates["dirty_artifacts"] == {"provider_list": True}
    assert out.state_updates["delivery_method"] == "fax"  # primary still captured


def test_invalidating_via_secondary_only_reverse_resolves_field():
    si = SecondaryIntent(
        type=SecondaryIntentType.INVALIDATING_CORRECTION,
        owner="provider_search_agent",
        verbatim_span="update my ZIP",
    )
    out = resolve_turn(
        TurnPlan(secondary_intents=[si]),
        _state("delivery_method"),
        utterance="update my ZIP please",
    )
    assert out.speech_act == CORRECTION_ACK
    assert out.dirty == {"provider_list": True}
    assert out.rewind_target == "provider_search_agent"


def test_non_invalidating_correction_acks_without_dirty():
    out = resolve_turn(
        TurnPlan(correction=Correction(field="first_name", owner="verification_agent", new_value="Dan")),
        _state("dob"),
        utterance="actually my first name is Dan",
    )
    assert out.speech_act == CORRECTION_ACK
    assert out.dirty == {}
    assert out.rewind_target == "verification_agent"


# ── precedence ──────────────────────────────────────────────────────────────


def test_safety_beats_everything():
    out = resolve_turn(
        TurnPlan(
            slot_answer="fax",
            correction=Correction(field="zip_code", owner="provider_search_agent"),
            guard=GuardType.SELF_HARM,
            guard_confidence=1.0,
        ),
        _state("delivery_method"),
        utterance="Fax but update my ZIP — honestly I can't go on",
    )
    assert out.speech_act is None  # escalation, not a member speech-act
    assert out.state_updates.get("escalate") is True
    assert out.dirty == {}  # correction not processed — safety short-circuits


def test_safety_via_secondary_type():
    si = SecondaryIntent(type=SecondaryIntentType.SAFETY, owner=None, verbatim_span="hurt myself")
    out = resolve_turn(TurnPlan(secondary_intents=[si]), _state("dob"), utterance="I might hurt myself")
    assert out.speech_act is None
    assert out.state_updates.get("escalate") is True


def test_invalidating_correction_beats_step_completion():
    # slot answered cleanly AND an invalidating correction present → correction wins.
    out = resolve_turn(
        TurnPlan(
            slot_answer="fax",
            correction=Correction(field="zip_code", owner="provider_search_agent"),
        ),
        _state("delivery_method"),
        utterance="fax but update my zip",
    )
    assert out.speech_act == CORRECTION_ACK
    assert out.dirty == {"provider_list": True}


# ── multi-intent ack + parking independents ─────────────────────────────────


def test_slot_answered_with_independent_parks_and_acks():
    si = SecondaryIntent(
        type=SecondaryIntentType.IN_SCOPE_INDEPENDENT,
        owner="benefits_agent",
        verbatim_span="deductible",
    )
    out = resolve_turn(
        TurnPlan(slot_answer="pediatrician", secondary_intents=[si]),
        _state("provider_type"),
        utterance="pediatrician, also what's my deductible",
    )
    assert out.speech_act == MULTI_INTENT_ACK
    assert out.parked == ["benefits_agent"]
    # Phase 3: queue entries carry the caller's verbatim span alongside the owner.
    assert out.state_updates["intent_queue"] == [{"owner": "benefits_agent", "span": "deductible"}]
    assert out.state_updates["provider_type"] == "Pediatrician"


def test_intent_queue_dedup_and_order():
    si = SecondaryIntent(
        type=SecondaryIntentType.IN_SCOPE_INDEPENDENT,
        owner="benefits_agent",
        verbatim_span="deductible",
    )
    out = resolve_turn(
        TurnPlan(slot_answer="pediatrician", secondary_intents=[si]),
        _state("provider_type", intent_queue=["benefits_agent", "follow_up_agent"]),
        utterance="pediatrician and my deductible",
    )
    assert out.state_updates["intent_queue"] == ["benefits_agent", "follow_up_agent"]  # no dup


def test_slot_not_answered_with_independent_still_acks():
    si = SecondaryIntent(
        type=SecondaryIntentType.IN_SCOPE_INDEPENDENT,
        owner="delivery_management_agent",
        verbatim_span="different fax",
    )
    out = resolve_turn(
        TurnPlan(secondary_intents=[si]),
        _state("benefits_response"),
        utterance="send it to a different fax",
    )
    assert out.speech_act == MULTI_INTENT_ACK
    assert out.parked == ["delivery_management_agent"]


# ── unsupported / out-of-scope ──────────────────────────────────────────────


def test_out_of_scope_secondary_declines():
    si = SecondaryIntent(type=SecondaryIntentType.OUT_OF_SCOPE, owner=None, verbatim_span="weather")
    out = resolve_turn(
        TurnPlan(slot_answer="yes", secondary_intents=[si]),
        _state("fax_confirmed"),
        utterance="yes and what's the weather",
    )
    assert out.speech_act == UNSUPPORTED_DECLINE
    assert out.state_updates["fax_confirmed"] == "yes"


# ── confidence / unknown gating ─────────────────────────────────────────────


def test_low_confidence_routes_to_clarify():
    out = resolve_turn(TurnPlan(slot_answer="09/03/1985", confidence=0.2), _state("dob"), utterance="maybe")
    assert out.speech_act == CLARIFY
    assert out.state_updates == {}  # never act on low confidence


def test_low_confidence_without_awaiting_is_open_redirect():
    out = resolve_turn(TurnPlan(confidence=0.1), _state(awaiting=""), utterance="hmm")
    assert out.speech_act == OPEN_REDIRECT


def test_unknown_secondary_type_routes_to_clarify():
    si = SecondaryIntent(type=SecondaryIntentType.UNKNOWN, owner=None, verbatim_span="thing")
    out = resolve_turn(TurnPlan(secondary_intents=[si]), _state("dob"), utterance="some thing")
    assert out.speech_act == CLARIFY


def test_speech_acts_are_from_closed_set():
    from agent.orchestration.resolver import SPEECH_ACTS

    assert SPEECH_ACTS == {
        "re_ask",
        "clarify",
        "correction_ack",
        "unsupported_decline",
        "multi_intent_ack",
        "open_redirect",
        "cross_slot_accept",
    }
