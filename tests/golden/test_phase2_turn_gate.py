"""
test_phase2_turn_gate.py — Phase 2: one understanding decode per turn, at ONE
chokepoint (generalizes both bugs).

Hermetic. Covers:
  * The turn gate (``understand_turn``): fast path (zero LLM cost), decode
    budget with heuristic fallback (metric="turnplan_timeout"), and the
    idempotence cache (a second caller for the same user message never decodes
    again — a different awaiting slot re-runs only the pure resolver).
  * Hand-coded yes/no confirmations flow through the resolver (Bug 2): a
    zip-confirmation "yes" with a parked side request produces accept →
    park-ack → next-step ask in ONE turn, enqueues the owner, applies the
    accept's state updates, and routes to the next step's agent.
  * CROSS_SLOT_ACCEPT: a non-answer that validates against exactly one OTHER
    pending slot of the active agent is accepted (no failed attempt) and the
    awaiting slot is re-asked; multiple plausible slots → CLARIFY, never guess.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import tests.golden  # noqa: F401 — ensures src/ is on sys.path
from agent.llm.schema import EventType, GuardType, SecondaryIntent, SecondaryIntentType, TurnPlan
from agent.orchestration import shadow as shadow_mod
from agent.orchestration.registry import queue_owners
from tests.golden.driver import run_fixture

pytestmark = pytest.mark.regression


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@contextmanager
def _capture_gate_logs():
    lg = logging.getLogger("agent.orchestration.turn_gate")
    handler = _ListHandler()
    prev = lg.level
    lg.setLevel(logging.DEBUG)
    lg.addHandler(handler)
    try:
        yield handler.records
    finally:
        lg.removeHandler(handler)
        lg.setLevel(prev)


def _decision(event_type: EventType = EventType.ANSWERED, extracted: dict | None = None):
    return SimpleNamespace(
        event_type=event_type,
        extracted=extracted or {},
        corrections={},
        guard=GuardType.NONE,
        guard_confidence=0.0,
    )


# ── the gate: fast path / timeout / idempotence ────────────────────────────────


async def test_fast_path_never_awaits_the_llm_decoder():
    """A plain single-intent answer must not pay the async (LLM) decode."""
    from agent.orchestration.turn_gate import understand_turn

    called = {"n": 0}

    async def _llm_decoder(_state, _utterance, _decision):
        called["n"] += 1
        return TurnPlan(slot_answer="never used")

    shadow_mod.set_shadow_decoder(_llm_decoder)
    state = {"awaiting_slot": "member_id", "active_agent": "verification_agent"}
    plan, outcome = await understand_turn(
        state,
        utterance="M714598",
        awaiting_slot="member_id",
        decision=_decision(extracted={"member_id": "M714598"}),
    )
    assert called["n"] == 0  # the async decoder was never awaited
    assert plan is not None and plan.slot_answer == "M714598"  # heuristic decode
    assert outcome is not None and outcome.state_updates.get("member_id") == "M714598"


async def test_timeout_falls_back_to_heuristic_with_metric(monkeypatch):
    monkeypatch.setenv("TURNPLAN_TIMEOUT_MS", "50")
    from agent.orchestration.turn_gate import understand_turn

    async def _slow_decoder(_state, _utterance, _decision):
        await asyncio.sleep(1.0)
        return TurnPlan(slot_answer="too late")

    shadow_mod.set_shadow_decoder(_slow_decoder)
    state = {"awaiting_slot": "member_id", "active_agent": "verification_agent"}
    # ANSWERED_WITH_FOLLOWUP defeats the fast path → the (slow) full decode runs.
    with _capture_gate_logs() as records:
        plan, outcome = await understand_turn(
            state,
            utterance="M714598 and one more thing",
            awaiting_slot="member_id",
            decision=_decision(EventType.ANSWERED_WITH_FOLLOWUP, {"member_id": "M714598"}),
        )
    assert plan is not None and plan.slot_answer == "M714598"  # heuristic fallback
    assert outcome is not None
    assert any(getattr(r, "metric", None) == "turnplan_timeout" for r in records)


async def test_idempotence_one_decode_per_turn():
    from agent.orchestration.turn_gate import understand_turn

    calls = {"n": 0}
    plan = TurnPlan(slot_answer="yes")

    def _counting_decoder(_state, _utterance, _decision):
        calls["n"] += 1
        return plan

    shadow_mod.set_shadow_decoder(_counting_decoder)
    state = {"awaiting_slot": "zip_confirmed", "active_agent": "provider_search_agent"}

    p1, o1 = await understand_turn(state, utterance="yes", awaiting_slot="zip_confirmed")
    p2, o2 = await understand_turn(state, utterance="yes", awaiting_slot="zip_confirmed")
    assert calls["n"] == 1  # second call served from the per-turn cache
    assert p1 is p2 and o1 is o2

    # A different slot focus re-runs only the pure resolver — still one decode.
    p3, _o3 = await understand_turn(state, utterance="yes", awaiting_slot="fax_confirmed")
    assert calls["n"] == 1
    assert p3 is plan

    # A NEW user message decodes again.
    await understand_turn(state, utterance="no", awaiting_slot="zip_confirmed")
    assert calls["n"] == 2


async def test_gate_respects_kill_switch():
    from agent.orchestration.turn_gate import understand_turn

    shadow_mod.set_shadow_decoder(None)
    assert await understand_turn({}, utterance="yes", awaiting_slot="zip_confirmed") == (None, None)


# ── Bug 2: hand-coded zip_confirmed through the resolver ───────────────────────


def _zip_confirmed_state() -> dict:
    return {
        "messages": [{"role": "assistant", "content": "Is the ZIP 94107 on file still correct?"}],
        "member_status_verify": True,
        "member_id": "M714598",
        "call_intent": "provider_services",
        "active_agent": "provider_search_agent",
        "provider_type": "Pediatrician",
        "zip_code": "94107",
        "zip_code_used": "",
        "awaiting_slot": "zip_confirmed",
        "dirty_artifacts": {},
        "intent_queue": [],
        "slot_attempts": {},
        "is_interrupt": True,
        "app_run_id": "p2-gate",
    }


def _plan_with_parked_benefits(slot_answer: str) -> TurnPlan:
    return TurnPlan(
        slot_answer=slot_answer,
        secondary_intents=[
            SecondaryIntent(
                type=SecondaryIntentType.IN_SCOPE_INDEPENDENT,
                owner="benefits_agent",
                verbatim_span="benefits",
            )
        ],
    )


async def test_zip_yes_with_parked_secondary_accepts_and_asks_next_step(monkeypatch):
    monkeypatch.setenv("MULTI_INTENT_LIVE", "false")   # templated-speech kill switch
    """Bug 2: 'yes' + a side request must produce accept → park-ack → the NEXT
    step's ask (delivery method) in ONE turn, with the accept taking effect and
    the next turn routed to the delivery agent — never a re-ask of zip_confirmed."""
    calls = {"n": 0}
    plan = _plan_with_parked_benefits("yes")

    def _decoder(_state, _utterance, _decision):
        calls["n"] += 1
        return plan

    shadow_mod.set_shadow_decoder(_decoder)
    fixture = {
        "id": "P2-ZIP-YES-PARKED",
        "driver": "provider_search_agent",
        "initial_state": _zip_confirmed_state(),
        "turns": [
            {
                "user": "yes, and I also have a question about my benefits",
                "extraction": {"extracted": {"zip_confirmed": "yes"}},
            }
        ],
    }
    run = await run_fixture(fixture, print_latency=False)
    turn0 = run.turns[0]

    # ONE decode for the whole turn (guards + hand-coded branch share the cache).
    assert calls["n"] == 1
    # The accept took effect: ZIP resolved, list clean, delivery is next.
    assert run.final_state.get("zip_code_used") == "94107"
    assert not (run.final_state.get("dirty_artifacts") or {}).get("provider_list")
    # The parked owner is enqueued for draining.
    assert "benefits_agent" in queue_owners(run.final_state.get("intent_queue"))
    # One sentence: park-ack + the NEXT step's ask — not a zip_confirmed re-ask.
    assert "benefits" in turn0.ai.lower()
    assert "delivery method" in turn0.ai.lower()
    assert turn0.awaiting_slot == "delivery_method"
    assert run.final_state.get("next_node") == "delivery_management_agent"
    assert run.recorder.count("dispatch_provider_list") == 0


async def test_zip_no_with_parked_secondary_marks_dirty_and_asks_new_zip(monkeypatch):
    monkeypatch.setenv("MULTI_INTENT_LIVE", "false")   # templated-speech kill switch
    """'no' + a side request: the decline still takes effect (list marked stale,
    new ZIP is the next step) and the side request is parked, in ONE sentence."""
    shadow_mod.set_shadow_decoder(lambda _s, _u, _d: _plan_with_parked_benefits("no"))
    fixture = {
        "id": "P2-ZIP-NO-PARKED",
        "driver": "provider_search_agent",
        "initial_state": _zip_confirmed_state(),
        "turns": [
            {
                "user": "no, and I also have a question about my benefits",
                "extraction": {"extracted": {"zip_confirmed": "no"}},
            }
        ],
    }
    run = await run_fixture(fixture, print_latency=False)
    turn0 = run.turns[0]

    assert "benefits_agent" in queue_owners(run.final_state.get("intent_queue"))
    # The dispute took effect: derived list stale, and the new ZIP is asked next.
    assert (run.final_state.get("dirty_artifacts") or {}).get("provider_list") is True
    assert turn0.awaiting_slot == "zip_code"
    assert "zip" in turn0.ai.lower()
    assert "benefits" in turn0.ai.lower()
    assert run.recorder.count("dispatch_provider_list") == 0


# ── CROSS_SLOT_ACCEPT ──────────────────────────────────────────────────────────


def test_cross_slot_answer_accepted_for_exactly_one_pending_slot():
    from agent.orchestration.resolver import CROSS_SLOT_ACCEPT, resolve_turn

    state = {
        "awaiting_slot": "zip_confirmed",
        "active_agent": "provider_search_agent",
        "provider_type": "",
        "zip_code": "94107",  # filled → not a cross-slot candidate
        "intent_queue": [],
    }
    out = resolve_turn(TurnPlan(), state, utterance="I'm looking for a primary care physician")
    assert out.speech_act == CROSS_SLOT_ACCEPT
    assert out.state_updates.get("provider_type")


def test_cross_slot_ambiguous_falls_back_to_clarify():
    from agent.orchestration.resolver import CLARIFY, resolve_turn

    # Both name slots are pending and both validate "Maria" — never guess.
    state = {
        "awaiting_slot": "member_id",
        "active_agent": "verification_agent",
        "first_name": "",
        "last_name": "",
        "dob": "",
        "relationship": "",
        "intent_queue": [],
    }
    out = resolve_turn(TurnPlan(slot_answer="Maria"), state, utterance="Maria")
    assert out.speech_act == CLARIFY


def test_bare_yes_never_cross_accepts_a_confirmation_slot():
    from agent.orchestration.resolver import RE_ASK, resolve_turn

    # Awaiting provider_type; "yes" must not confirm the (unread) ZIP.
    state = {
        "awaiting_slot": "provider_type",
        "active_agent": "provider_search_agent",
        "provider_type": "",
        "zip_code": "94107",
        "zip_confirmed": "",
        "intent_queue": [],
    }
    out = resolve_turn(TurnPlan(), state, utterance="yes")
    assert out.speech_act == RE_ASK


async def test_apply_cross_slot_accept_speaks_value_and_reasks_awaiting():
    from agent.conversation.context import ConversationContext
    from agent.core.agent import BaseAgent
    from agent.orchestration.resolver import CROSS_SLOT_ACCEPT, ResolverOutcome

    class _Probe(BaseAgent):
        AGENT_NAME = "provider_search_agent"

        async def run(self, state):  # pragma: no cover - not exercised
            return {}

    agent = _Probe.from_state({})
    outcome = ResolverOutcome(CROSS_SLOT_ACCEPT, {"provider_type": "Primary Care Physician"})
    interrupt = await agent._apply_resolver_outcome(
        {"awaiting_slot": "zip_confirmed"},
        ConversationContext(),
        "zip_confirmed",
        None,
        None,
        outcome,
        slot_answered=False,
        slot_label="ZIP code confirmation",
    )
    assert interrupt is not None
    # The cross value is accepted into state; the awaiting slot stays open —
    # and no failed attempt was counted against it.
    assert interrupt["provider_type"] == "Primary Care Physician"
    assert interrupt["awaiting_slot"] == "zip_confirmed"
    spoken = interrupt["messages"]["content"]
    assert "Primary Care Physician" in spoken
    assert "ZIP code confirmation" in spoken
    assert (interrupt.get("slot_attempts") or {}).get("zip_confirmed", {}).get("attempt_count", 0) == 0
