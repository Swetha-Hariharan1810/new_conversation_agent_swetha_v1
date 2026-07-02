"""
test_phase3_intent_ownership.py — Phase 3: never guess an owner, know more.

Hermetic. Covers:
  * ``INTENT_PHRASES`` / ``owner_for_phrase``: deterministic keyword-stem →
    owner resolution (longest match, word boundary, case-insensitive), None
    when nothing matches.
  * The heuristic decoder no longer hallucinates routing: a side request whose
    phrase resolves goes to its real owner (refund → claim_adjustment_agent,
    not benefits_agent); one that doesn't resolve is emitted UNKNOWN with no
    owner, which the resolver turns into CLARIFY — ask, never act.
  * intent_queue entries carry the caller's verbatim span; draining surfaces it
    as ``drained_intent_reason`` and the receiving agent opens with a grounded
    one-clause bridge when UNIFIED_VOICE is on (consumed after one turn).
"""

from __future__ import annotations

import pytest

import tests.golden  # noqa: F401 — ensures src/ is on sys.path
from agent.llm.schema import SecondaryIntentType, TurnPlan
from agent.orchestration import shadow as shadow_mod
from agent.orchestration.fast_path import drain_next_intent
from agent.orchestration.registry import owner_for_phrase, queue_entry, queue_owners

pytestmark = pytest.mark.regression


# ── owner_for_phrase vocabulary ────────────────────────────────────────────────


def test_owner_for_phrase_resolves_known_stems():
    assert owner_for_phrase("I want a refund on my last bill") == "claim_adjustment_agent"
    assert owner_for_phrase("I was reimbursed the wrong amount") == "claim_adjustment_agent"
    assert owner_for_phrase("there's a weird CHARGE on my statement") == "claim_adjustment_agent"
    assert owner_for_phrase("what's my deductible?") == "benefits_agent"
    assert owner_for_phrase("does my coverage include that?") == "benefits_agent"
    assert owner_for_phrase("upload my medical records") == "records_coordination_agent"
    assert owner_for_phrase("can you text me updates") == "notification_setup_agent"
    assert owner_for_phrase("I prefer SMS") == "notification_setup_agent"
    assert owner_for_phrase("find me a specialist") == "provider_search_agent"
    assert owner_for_phrase("send it by fax instead") == "delivery_management_agent"


def test_owner_for_phrase_never_guesses():
    assert owner_for_phrase("what's the weather like up there?") is None
    assert owner_for_phrase("") is None
    assert owner_for_phrase(None) is None
    # Stems anchor at a word boundary — no mid-word matches ("...ebill..." etc.).
    assert owner_for_phrase("the exhibillity of it all") is None


# ── heuristic decoder: real owner or UNKNOWN — never a default ────────────────


def _heuristic_plan(utterance: str) -> TurnPlan | None:
    return shadow_mod.heuristic_decoder(
        {"awaiting_slot": "provider_type", "active_agent": "provider_search_agent"},
        utterance,
        None,
    )


def test_heuristic_routes_refund_to_claim_adjustment():
    """The old else-branch would have parked this at benefits_agent."""
    plan = _heuristic_plan("also I want a refund on my last bill")
    assert plan is not None and plan.secondary_intents
    si = plan.secondary_intents[0]
    assert si.type == SecondaryIntentType.IN_SCOPE_INDEPENDENT
    assert si.owner == "claim_adjustment_agent"


def test_heuristic_emits_unknown_when_no_phrase_matches_and_resolver_clarifies():
    from agent.orchestration.resolver import CLARIFY, resolve_turn

    utterance = "by the way, what's the weather like up there?"
    plan = _heuristic_plan(utterance)
    assert plan is not None and plan.secondary_intents
    si = plan.secondary_intents[0]
    assert si.type == SecondaryIntentType.UNKNOWN
    assert not si.owner  # no hallucinated owner

    # The resolver turns UNKNOWN into CLARIFY — ask, never act, never misroute.
    out = resolve_turn(
        plan,
        {"awaiting_slot": "provider_type", "dirty_artifacts": {}, "intent_queue": []},
        utterance=utterance,
    )
    assert out.speech_act == CLARIFY
    assert not out.parked
    assert "intent_queue" not in out.state_updates


# ── span rides the queue: park → drain → bridge ────────────────────────────────


def test_drain_surfaces_the_parked_span_as_reason():
    queue = [queue_entry("claim_adjustment_agent", "a refund on my last bill")]
    out = drain_next_intent({"intent_queue": queue})
    assert out["next_node"] == "claim_adjustment_agent"
    assert out["intent_queue"] == []
    assert out["drained_intent_reason"] == "a refund on my last bill"


def test_park_to_drain_span_roundtrip_through_resolver():
    from agent.llm.schema import SecondaryIntent

    plan = TurnPlan(
        slot_answer="pediatrician",
        secondary_intents=[
            SecondaryIntent(
                type=SecondaryIntentType.IN_SCOPE_INDEPENDENT,
                owner="claim_adjustment_agent",
                verbatim_span="a refund on my last bill",
            )
        ],
    )
    from agent.orchestration.resolver import resolve_turn

    out = resolve_turn(
        plan,
        {"awaiting_slot": "provider_type", "dirty_artifacts": {}, "intent_queue": []},
        utterance="a pediatrician please — oh and a refund on my last bill",
    )
    assert queue_owners(out.state_updates["intent_queue"]) == ["claim_adjustment_agent"]
    drain = drain_next_intent({"intent_queue": out.state_updates["intent_queue"]})
    assert drain["drained_intent_reason"] == "a refund on my last bill"


# ── BaseAgent.execute bridges the drained request (UNIFIED_VOICE) ──────────────


def _probe_agent(first_message: str):
    from agent.core.agent import BaseAgent

    class _Probe(BaseAgent):
        AGENT_NAME = "claim_adjustment_agent"

        async def run(self, state):
            return self.ask_member(state, first_message)

    return _Probe.from_state({})


async def test_drained_turn_opens_with_grounded_bridge(monkeypatch):
    monkeypatch.setenv("UNIFIED_VOICE", "true")
    agent = _probe_agent("Could I get your claim reference number?")
    state = {"drained_intent_reason": "a refund on my last bill", "messages": []}
    result = await agent.execute(state)
    spoken = result["messages"]["content"]
    # One-clause bridge in the caller's own words, then the agent's opener.
    assert "a refund on my last bill" in spoken
    assert spoken.endswith("Could I get your claim reference number?")
    # The reason is consumed — the bridge speaks exactly once.
    assert result["drained_intent_reason"] == ""


async def test_no_bridge_when_flag_off_but_reason_still_consumed(monkeypatch):
    monkeypatch.setenv("UNIFIED_VOICE", "false")
    agent = _probe_agent("Could I get your claim reference number?")
    state = {"drained_intent_reason": "a refund on my last bill", "messages": []}
    result = await agent.execute(state)
    assert result["messages"]["content"] == "Could I get your claim reference number?"
    assert result["drained_intent_reason"] == ""


async def test_no_span_falls_back_to_generic_bridge(monkeypatch):
    monkeypatch.setenv("UNIFIED_VOICE", "true")
    # A legacy bare-string queue entry drains with reason "" → no bridge text to
    # ground, so nothing is prepended (the empty reason is simply not bridged).
    agent = _probe_agent("Could I get your claim reference number?")
    state = {"drained_intent_reason": "", "messages": []}
    result = await agent.execute(state)
    assert result["messages"]["content"] == "Could I get your claim reference number?"
