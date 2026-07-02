"""
test_phase3d.py — Phase 3D: roll the resolver live across every agent.

Covers the consolidated registry, the cross-agent roll at the shared chokepoint
(corrections + side-questions on both answered and non-answered slot turns),
intent-queue draining, the follow_up migration onto the unified path, and the
one-understanding-decode-per-turn regression guard.
"""

from __future__ import annotations

import pytest

import tests.golden  # noqa: F401 — ensures src/ is on sys.path
from agent.llm.schema import SecondaryIntent, SecondaryIntentType, TurnPlan
from agent.orchestration import shadow as shadow_mod
from agent.orchestration.fast_path import drain_next_intent
from agent.orchestration.registry import queue_owners
from tests.golden.driver import load_fixture, run_fixture

pytestmark = pytest.mark.regression


# ── conversation-wide registry ───────────────────────────────────────────────


def test_registry_registers_every_agent_and_owner():
    from agent.orchestration.registry import (
        ALL_AGENTS,
        INTENT_OWNER_REGISTRY,
        INVALIDATION_MAP,
        owner_of,
    )

    for agent in (
        "verification_agent",
        "provider_search_agent",
        "delivery_management_agent",
        "benefits_agent",
        "care_wellness_agent",
        "claim_adjustment_agent",
        "records_coordination_agent",
        "notification_setup_agent",
        "follow_up_agent",
    ):
        assert agent in ALL_AGENTS

    # Owners resolve across agents (corrections/secondaries can route anywhere).
    assert owner_of("zip_code") == "provider_search_agent"
    assert owner_of("member_id") == "verification_agent"
    assert owner_of("fax") == "delivery_management_agent"
    assert owner_of("reference_number") == "claim_adjustment_agent"
    assert owner_of("notification_method") == "notification_setup_agent"
    assert owner_of("provider_list") == "delivery_management_agent"

    assert INVALIDATION_MAP["zip_code"] == ["provider_list"]
    # Claim-flow parity: a disputed reference invalidates its claim artifacts.
    assert set(INVALIDATION_MAP["reference_number"]) == {"upload_link", "personal_guide_outreach"}
    # The scattered owner maps now derive from one source.
    assert INTENT_OWNER_REGISTRY["provider_list"] == "delivery_management_agent"


def test_resolver_known_agents_sourced_from_registry():
    from agent.orchestration.registry import ALL_AGENTS
    from agent.orchestration.resolver import KNOWN_AGENTS

    assert ALL_AGENTS <= KNOWN_AGENTS
    assert {"intake_agent", "escalation_agent", "closure_agent"} <= KNOWN_AGENTS


# ── cross-agent roll: side-question on a NON-answered slot turn ───────────────


def _provider_type_state() -> dict:
    return {
        "messages": [{"role": "assistant", "content": "What type of provider are you looking for?"}],
        "member_status_verify": True,
        "member_id": "M714598",
        "call_intent": "provider_services",
        "active_agent": "provider_search_agent",
        "provider_type": "",
        "zip_code": "94107",
        "zip_code_used": "",
        "awaiting_slot": "provider_type",
        "dirty_artifacts": {},
        "intent_queue": [],
        "slot_attempts": {},
        "is_interrupt": True,
        "app_run_id": "p3d",
    }


def _fixed_decoder(plan: TurnPlan):
    return lambda _state, _utterance, _decision: plan


async def test_out_of_scope_on_unanswered_slot_gets_spoken_outcome_and_reasks():
    plan = TurnPlan(
        secondary_intents=[
            SecondaryIntent(type=SecondaryIntentType.OUT_OF_SCOPE, owner=None, verbatim_span="weather")
        ]
    )
    shadow_mod.set_shadow_decoder(_fixed_decoder(plan))
    fixture = {
        "id": "P3D-OOS-NOANSWER",
        "driver": "provider_search_agent",
        "initial_state": _provider_type_state(),
        "turns": [{"user": "what's the weather like?", "extraction": {}}],
    }
    run = await run_fixture(fixture, print_latency=False)

    turn0 = run.turns[0]
    # Spoken outcome for the unanswerable question, and the slot is re-asked.
    assert turn0.ai.strip()
    assert turn0.awaiting_slot == "provider_type"  # still collecting
    assert "provider" in turn0.ai.lower()  # the re-ask
    assert run.recorder.count("dispatch_provider_list") == 0


async def test_independent_on_unanswered_slot_is_parked_and_reasks():
    plan = TurnPlan(
        secondary_intents=[
            SecondaryIntent(
                type=SecondaryIntentType.IN_SCOPE_INDEPENDENT,
                owner="delivery_management_agent",
                verbatim_span="a different fax",
            )
        ]
    )
    shadow_mod.set_shadow_decoder(_fixed_decoder(plan))
    fixture = {
        "id": "P3D-INDEP-NOANSWER",
        "driver": "provider_search_agent",
        "initial_state": _provider_type_state(),
        "turns": [{"user": "can you use a different fax for that?", "extraction": {}}],
    }
    run = await run_fixture(fixture, print_latency=False)

    assert "delivery_management_agent" in queue_owners(run.final_state.get("intent_queue"))
    assert run.turns[0].awaiting_slot == "provider_type"  # re-asked, not abandoned


# ── intent-queue draining (drained on a later turn) ──────────────────────────


def test_drain_next_intent_pops_owner():
    out = drain_next_intent({"intent_queue": ["benefits_agent", "delivery_management_agent"]})
    assert out == {
        "next_node": "benefits_agent",
        "intent_queue": ["delivery_management_agent"],
        "is_interrupt": False,
        "drained_intent_reason": "",  # legacy bare-string entry carries no span
    }
    assert drain_next_intent({"intent_queue": []}) is None
    assert drain_next_intent({}) is None
    # Unknown entries are skipped, not routed to.
    assert drain_next_intent({"intent_queue": ["not_an_agent"]}) is None


async def test_parked_independent_is_drained_on_a_later_turn():
    # Park a benefits question during provider_type collection (Phase 3C live).
    run = await run_fixture(load_fixture("slot_interrupt_fresh_request"), print_latency=False)
    assert "benefits_agent" in queue_owners(run.final_state.get("intent_queue"))

    # On a later (completion) turn, the orchestrator drains it to its owner.
    drain = drain_next_intent(run.final_state)
    assert drain["next_node"] == "benefits_agent"
    assert "benefits_agent" not in queue_owners(drain["intent_queue"])


# ── follow_up migrated onto the unified path ─────────────────────────────────


def test_follow_up_parks_cross_domain_side_request():
    from agent.agents.follow_up.agent import FollowUpAgent

    agent = FollowUpAgent()
    state = {"intent_queue": [], "messages": [], "app_run_id": "fu", "slot_attempts": {}}
    out = agent._resolve_cross_domain_side_request(
        state, "Sounds good, but can you send the list to a different fax number?", 2
    )
    assert out is not None
    assert "delivery_management_agent" in queue_owners(out["intent_queue"])


def test_follow_up_leaves_answerable_questions_to_qa_path():
    from agent.agents.follow_up.agent import FollowUpAgent

    agent = FollowUpAgent()
    state = {"intent_queue": [], "messages": [], "app_run_id": "fu", "slot_attempts": {}}
    # A benefits question follow_up can answer itself must NOT be parked.
    assert agent._resolve_cross_domain_side_request(state, "but what's my OOP max?", 2) is None


# ── regression guard: exactly one understanding decode per slot turn ──────────


async def test_one_understanding_decode_per_turn_no_fanout():
    calls = {"n": 0}
    base = shadow_mod.heuristic_decoder

    def _counting(state, utterance, decision):
        calls["n"] += 1
        return base(state, utterance, decision)

    shadow_mod.set_shadow_decoder(_counting)
    # A multi-intent turn that parks an independent — must still decode once,
    # with no per-parked-intent fan-out.
    await run_fixture(load_fixture("slot_interrupt_fresh_request"), print_latency=False)
    assert calls["n"] == 1


async def test_clean_single_intent_decodes_once_and_does_not_act():
    calls = {"n": 0}
    base = shadow_mod.heuristic_decoder

    def _counting(state, utterance, decision):
        calls["n"] += 1
        return base(state, utterance, decision)

    shadow_mod.set_shadow_decoder(_counting)
    fixture = {
        "id": "P3D-CLEAN",
        "driver": "provider_search_agent",
        "initial_state": _provider_type_state(),
        "turns": [{"user": "Pediatrician", "extraction": {"extracted": {"provider_type": "pediatrician"}}}],
    }
    run = await run_fixture(fixture, print_latency=False)
    assert calls["n"] == 1
    # Clean single answer advances normally (resolver does not act).
    assert run.final_state.get("provider_type") == "Pediatrician"
    assert run.turns[0].awaiting_slot == "zip_confirmed"
