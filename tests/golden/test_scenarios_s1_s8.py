"""
test_scenarios_s1_s8.py — the Section-11 verification scenarios, deterministic.

These use the UAT-007 cast (Daniel Reed, M714598, DOB Sept 3 1985, ZIP 94107,
fax 415-555-3299, pediatrician) so state is consistent. Each test is the
deterministic counterpart of the listed scenario — it proves the Python
guarantees (resolver decision, gate, span/owner checks, draining, precedence)
without depending on an LLM classifying correctly. The end-to-end (live LLM)
counterparts live in tests/live_e2e/scenarios.py (group N), run via test_live.py.

S6 and S7 are the safety net and run in CI on every commit (see ci.yml).
"""

from __future__ import annotations

import re

import pytest

import tests.golden  # noqa: F401 — ensures src/ is on sys.path
from agent.llm.schema import (
    Correction,
    GuardType,
    SecondaryIntent,
    SecondaryIntentType,
    TurnPlan,
)
from agent.orchestration import shadow as shadow_mod
from agent.orchestration.fast_path import drain_next_intent
from agent.orchestration.resolver import (
    CORRECTION_ACK,
    MULTI_INTENT_ACK,
    resolve_turn,
)
from tests.golden.driver import load_fixture, run_conversation, run_fixture

pytestmark = pytest.mark.regression


def _fixed_decoder(plan):
    return lambda _s, _u, _d: plan


# ── S1 — Canonical UAT-007: slot answer + invalidating correction (P1+3B) ────


async def test_s1_slot_answer_plus_invalidating_correction_end_to_end():
    initial = {
        "messages": [{"role": "assistant", "content": "Your list is ready. Fax or email?"}],
        "member_status_verify": True,
        "member_id": "M714598",
        "first_name": "Daniel",
        "call_intent": "provider_services",
        "active_agent": "delivery_management_agent",
        "next_node": "delivery_management_agent",
        "provider_type": "Pediatrician",
        "zip_code": "94107",
        "zip_code_used": "94107",
        "fax": "415-555-3299",
        "email": "",
        "delivery_method": "",
        "awaiting_slot": "",
        "dirty_artifacts": {},
        "intent_queue": [],
        "slot_attempts": {},
        "is_interrupt": True,
        "app_run_id": "s1",
    }
    turns = [
        {"agent": "delivery_management_agent", "user": "Fax, but I need to update my ZIP code.",
         "extraction": {"extracted": {"delivery_method": "fax"}}},
        {"agent": "provider_search_agent", "user": "It's 94110.",
         "extraction": {"extracted": {"zip_code": "94110"}}},
        {"agent": "delivery_management_agent", "user": "fax",
         "extraction": {"extracted": {"delivery_method": "fax"}}},
        {"agent": "delivery_management_agent", "user": "yes",
         "extraction": {"extracted": {"fax_confirmed": "yes"}}},
        {"agent": "delivery_management_agent", "user": "no thanks",
         "extraction": {"extracted": {"benefits_response": "no"}}},
    ]
    run = await run_conversation(initial, turns, fixture_id="S1")
    # ZIP acknowledged, not dropped; list dispatched only after the new ZIP resolves.
    assert "zip" in run.turns[0].ai.lower() and "fax" in run.turns[0].ai.lower()
    dispatches = run.recorder.for_tool("dispatch_provider_list")
    assert len(dispatches) == 1 and dispatches[0]["zip_code"] == "94110"
    assert run.recorder.for_tool("update_zip_code") == [{"member_id": "M714598", "zip_code": "94110"}]


# ── S2 — Mid-verification identity correction (Phase 3D) ─────────────────────


async def test_s2_mid_verification_identity_correction():
    from agent.core.agent import BaseAgent
    from agent.core.slot_manager import _InternalSlotConfig
    from agent.slots.normalizers import normalize_dob
    from agent.slots.types import SlotType
    from agent.slots.validators import validate_dob

    class _Probe(BaseAgent):
        AGENT_NAME = "verification_agent"

        async def run(self, state):  # pragma: no cover
            return {}

    state = {
        "awaiting_slot": "dob",
        "member_status_verify": False,
        "member_id": "M714598",
        "first_name": "Daniel",
        "slot_attempts": {"member_id": {"attempt_count": 1, "confirmed": True, "last_value": "M714598"}},
    }
    agent = _Probe.from_state(state)
    config = _InternalSlotConfig("dob", "", normalize_dob, validate_dob, SlotType.DOB)
    decision = TurnPlan()  # unused; resolver reads the shared decode of the utterance
    _utter = "September third nineteen eighty-five — wait, my member ID was wrong, it's M714599."
    messages = [
        {"role": "assistant", "content": "And your date of birth?"},
        {"role": "user", "content": _utter},
    ]
    # Inject the decode for this turn: DOB answer + member_id correction.
    plan = TurnPlan(
        slot_answer="09/03/1985",
        correction=Correction(field="member_id", owner="verification_agent", new_value="M714599"),
    )
    shadow_mod.set_shadow_decoder(_fixed_decoder(plan))
    value, interrupt = await agent._collect_slot(
        dict(state), config, messages, pre_extracted="09/03/1985", decision=decision
    )
    assert value == "09/03/1985"  # DOB accepted
    assert interrupt is not None
    assert interrupt["awaiting_slot"] == "member_id"  # rewound to re-validate identity
    assert interrupt["next_node"] == "verification_agent"
    ack = interrupt["messages"]["content"]
    assert re.search(r"member id", ack, re.IGNORECASE)  # correction acknowledged


# ── S3 — Slot answer + fresh in-scope independent (Phase 3C) ─────────────────


async def test_s3_slot_answer_plus_independent_parked_then_drained():
    run = await run_fixture(load_fixture("slot_interrupt_fresh_request"), print_latency=False)
    # benefits parked + acknowledged this turn.
    assert "benefits_agent" in (run.final_state.get("intent_queue") or [])
    assert re.search(r"benefits", run.turns[0].ai, re.IGNORECASE)
    assert run.dropped_request_count == 0
    # actioned (drained) on a subsequent turn.
    drain = drain_next_intent(run.final_state)
    assert drain["next_node"] == "benefits_agent"


# ── S4 — Safety/transfer injected mid-slot — precedence (Phase 3B) ───────────


def test_s4_safety_outranks_slot_and_correction_in_resolver():
    plan = TurnPlan(
        slot_answer="yes",
        correction=Correction(field="zip_code", owner="provider_search_agent"),
        guard=GuardType.SELF_HARM,
        guard_confidence=1.0,
    )
    out = resolve_turn(plan, {"awaiting_slot": "fax_confirmed", "dirty_artifacts": {}}, utterance="...")
    assert out.speech_act is None  # escalation, not a normal turn
    assert out.state_updates.get("escalate") is True
    assert out.dirty == {}  # correction not processed — safety short-circuits


async def test_s4_transfer_guard_escalates_on_slot_turn():
    fixture = {
        "id": "S4-TRANSFER",
        "driver": "delivery_management_agent",
        "initial_state": {
            "messages": [{"role": "assistant", "content": "Is 415-555-3299 correct?"}],
            "member_status_verify": True,
            "member_id": "M714598",
            "call_intent": "provider_services",
            "active_agent": "delivery_management_agent",
            "provider_type": "Pediatrician",
            "zip_code": "94107",
            "zip_code_used": "94107",
            "fax": "415-555-3299",
            "delivery_method": "fax",
            "awaiting_slot": "fax_confirmed",
            "dirty_artifacts": {},
            "slot_attempts": {},
            "is_interrupt": True,
            "app_run_id": "s4",
        },
        "turns": [
            {
                "user": "Yes that's right — actually just transfer me to a human, this is urgent.",
                "extraction": {"guard": "TRANSFER_REQUEST", "guard_confidence": 1.0},
            }
        ],
    }
    run = await run_fixture(fixture, print_latency=False)
    assert run.final_state.get("next_node") == "escalation_agent"
    assert run.recorder.count("dispatch_provider_list") == 0  # slot answer not actioned


# ── S5 — Triple/quad intent — precedence + multi-park drain (Phase 3C/3D) ────


def test_s5_quad_intent_precedence_and_accounting():
    plan = TurnPlan(
        slot_answer="email",
        correction=Correction(field="zip_code", owner="provider_search_agent"),
        secondary_intents=[
            SecondaryIntent(
                type=SecondaryIntentType.IN_SCOPE_INDEPENDENT,
                owner="delivery_management_agent",
                verbatim_span="a different fax number",
            ),
            SecondaryIntent(
                type=SecondaryIntentType.IN_DOMAIN_UNSUPPORTED,
                owner=None,
                verbatim_span="pharmacy copay",
            ),
        ],
    )
    utterance = (
        "Email — but my ZIP is wrong, and can you also send the list to a different fax number, "
        "and what's my pharmacy copay?"
    )
    out = resolve_turn(plan, {"awaiting_slot": "delivery_method", "dirty_artifacts": {}, "intent_queue": []},
                       utterance=utterance)
    # Precedence: invalidating correction wins; slot held; independent parked; unsupported declined.
    assert out.speech_act == CORRECTION_ACK
    assert out.state_updates.get("delivery_method") == "email"  # slot answer held
    assert out.dirty == {"provider_list": True}  # zip correction → rewind/refresh
    assert out.rewind_target == "provider_search_agent"
    assert "delivery_management_agent" in out.parked  # alternate fax parked
    assert out.declined == ["in_domain_unsupported"]  # pharmacy copay declined, not fabricated
    # 4 distinct outcomes accounted for; nothing silently dropped.


async def test_s5_quad_intent_every_request_gets_spoken_outcome():
    plan = TurnPlan(
        slot_answer="email",
        correction=Correction(field="zip_code", owner="provider_search_agent"),
        secondary_intents=[
            SecondaryIntent(type=SecondaryIntentType.IN_SCOPE_INDEPENDENT,
                            owner="delivery_management_agent", verbatim_span="different fax"),
            SecondaryIntent(type=SecondaryIntentType.IN_DOMAIN_UNSUPPORTED,
                            owner=None, verbatim_span="pharmacy copay"),
        ],
    )
    shadow_mod.set_shadow_decoder(_fixed_decoder(plan))
    fixture = {
        "id": "S5-E2E",
        "driver": "delivery_management_agent",
        "initial_state": {
            "messages": [{"role": "assistant", "content": "Fax or email?"}],
            "member_status_verify": True, "member_id": "M714598", "call_intent": "provider_services",
            "active_agent": "delivery_management_agent", "provider_type": "Pediatrician",
            "zip_code": "94107", "zip_code_used": "94107", "fax": "415-555-3299", "email": "",
            "delivery_method": "", "awaiting_slot": "", "dirty_artifacts": {}, "intent_queue": [],
            "slot_attempts": {}, "is_interrupt": True, "app_run_id": "s5",
        },
        "turns": [{
            "user": "Email — but my ZIP is wrong, and send to a different fax, and what's my pharmacy copay?",
            "extraction": {"extracted": {"delivery_method": "email"}},
        }],
    }
    run = await run_fixture(fixture, print_latency=False)
    ai = run.turns[0].ai.lower()
    # The acknowledgement covers the ZIP correction, the parked fax, AND declines the copay.
    assert "zip" in ai
    assert "delivery_management_agent" in (run.final_state.get("intent_queue") or [])
    assert re.search(r"not able|can't|outside", ai)  # pharmacy copay declined inline
    assert run.recorder.count("dispatch_provider_list") == 0  # never on the disputed ZIP
    assert run.dropped_request_count == 0


# ── S6 — Misclassification backstop — gate holds without the model (P1, det.) ─


@pytest.mark.guards
async def test_s6_gate_holds_when_correction_misclassified():
    # The decode MISLABELS the ZIP correction as an in-scope independent (wrong
    # enum). The resolver therefore does NOT flip dirty from it...
    mislabeled = TurnPlan(
        slot_answer="fax",
        secondary_intents=[
            SecondaryIntent(
                type=SecondaryIntentType.IN_SCOPE_INDEPENDENT,
                owner="provider_search_agent",
                verbatim_span="update my ZIP code",
            )
        ],
    )
    out = resolve_turn(
        mislabeled,
        {"awaiting_slot": "delivery_method", "dirty_artifacts": {"provider_list": True}},
        utterance="Fax, but update my ZIP code",
    )
    assert out.speech_act == MULTI_INTENT_ACK  # parked, not rewound (worse-UX)
    assert out.dirty == {}  # the misclassification did NOT clear/flip via this path

    # ...but the Phase 1 gate reads ONLY dirty_artifacts, so delivery is still
    # blocked because provider_list is dirty (set by the invalidation map upstream).
    fixture = {
        "id": "S6-GATE",
        "driver": "delivery_management_agent",
        "initial_state": {
            "messages": [{"role": "assistant", "content": "Is 415-555-3299 correct?"}],
            "member_status_verify": True, "member_id": "M714598", "call_intent": "provider_services",
            "active_agent": "delivery_management_agent", "provider_type": "Pediatrician",
            "zip_code": "94107", "zip_code_used": "94107", "fax": "415-555-3299",
            "delivery_method": "fax", "awaiting_slot": "fax_confirmed",
            "dirty_artifacts": {"provider_list": True}, "slot_attempts": {},
            "is_interrupt": True, "app_run_id": "s6",
        },
        "turns": [{"user": "yes", "extraction": {"extracted": {"fax_confirmed": "yes"}}}],
    }
    run = await run_fixture(fixture, print_latency=False)
    assert run.recorder.count("dispatch_provider_list") == 0  # safety holds regardless of classification
    assert run.final_state.get("next_node") == "provider_search_agent"  # routed to re-resolve


# ── S7 — Span-check drops a hallucinated intent (Phase 3A, deterministic) ─────


@pytest.mark.guards
def test_s7_span_and_owner_checks_drop_invented_intents():
    plan = TurnPlan(
        slot_answer="email",
        secondary_intents=[
            # span absent from the utterance → hallucinated
            SecondaryIntent(type=SecondaryIntentType.IN_SCOPE_INDEPENDENT,
                            owner="benefits_agent", verbatim_span="cancel my policy"),
            # owner not in the registry → unresolvable
            SecondaryIntent(type=SecondaryIntentType.IN_SCOPE_INDEPENDENT,
                            owner="ghost_agent", verbatim_span="Email is fine"),
        ],
    )
    out = resolve_turn(plan, {"awaiting_slot": "delivery_method", "dirty_artifacts": {}, "intent_queue": []},
                       utterance="Email is fine, thanks.")
    # Neither invented intent survives → clean answer, nothing parked/acknowledged.
    assert out.parked == []
    assert out.speech_act is None  # just the validated slot answer proceeds
    assert out.state_updates == {"delivery_method": "email"}


# ── S8 — Single-intent regression + one decode per turn (all phases) ─────────


async def test_s8_single_intent_unchanged_and_one_decode():
    calls = {"n": 0}
    base = shadow_mod.heuristic_decoder

    def _counting(state, utterance, decision):
        calls["n"] += 1
        return base(state, utterance, decision)

    shadow_mod.set_shadow_decoder(_counting)
    fixture = {
        "id": "S8-CLEAN",
        "driver": "provider_search_agent",
        "initial_state": {
            "messages": [{"role": "assistant", "content": "What type of provider?"}],
            "member_status_verify": True, "member_id": "M714598", "call_intent": "provider_services",
            "active_agent": "provider_search_agent", "provider_type": "", "zip_code": "94107",
            "zip_code_used": "", "awaiting_slot": "provider_type", "dirty_artifacts": {},
            "intent_queue": [], "slot_attempts": {}, "is_interrupt": True, "app_run_id": "s8",
        },
        "turns": [{"user": "a pediatrician", "extraction": {"extracted": {"provider_type": "pediatrician"}}}],
    }
    run = await run_fixture(fixture, print_latency=False)
    assert calls["n"] == 1  # exactly one understanding decode this turn
    assert run.final_state.get("provider_type") == "Pediatrician"
    assert run.turns[0].awaiting_slot == "zip_confirmed"  # advances normally, no parking
    assert run.dropped_request_count == 0
