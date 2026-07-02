"""
test_phase4_rollout.py — Phase 4: flags flipped, latency proven.

Hermetic. Covers the rollout acceptance criteria:
  * The rebuild is ON by default (UNIFIED_VOICE / TURNPLAN_DECODE=live /
    MULTI_INTENT_LIVE) — asserted in test_phase0_flags — and a full agent turn
    under the NEW defaults still makes exactly ONE understanding decode (the
    idempotence guard's counter proves it: guards + hand-coded branch +
    pipeline share the cache).
  * ``understand_turn`` emits metric="turn_gate_latency_ms" with a
    fast_path=true/false tag on every decoded turn.
  * Fast-path turns add < 1ms of gate overhead (pure-Python decode + resolve;
    no LLM call).
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager

import pytest

import tests.golden  # noqa: F401 — ensures src/ is on sys.path
from agent.llm.schema import SecondaryIntent, SecondaryIntentType, TurnPlan
from agent.orchestration import shadow as shadow_mod
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


def _latency_records(records) -> list[logging.LogRecord]:
    return [r for r in records if getattr(r, "metric", None) == "turn_gate_latency_ms"]


# ── the latency metric: emitted per decoded turn, tagged by path ───────────────


async def test_gate_emits_latency_metric_tagged_fast_path():
    from agent.orchestration.turn_gate import understand_turn

    shadow_mod.set_shadow_decoder(shadow_mod.heuristic_decoder)
    with _capture_gate_logs() as records:
        # Plain single-intent answer → fast path.
        await understand_turn({"awaiting_slot": "member_id"}, utterance="M714598", awaiting_slot="member_id")
    lat = _latency_records(records)
    assert len(lat) == 1
    assert lat[0].fast_path is True
    assert lat[0].latency_ms >= 0.0


async def test_gate_emits_decode_tag_on_full_decode():
    from types import SimpleNamespace

    from agent.llm.schema import EventType, GuardType
    from agent.orchestration.turn_gate import understand_turn

    shadow_mod.set_shadow_decoder(shadow_mod.heuristic_decoder)
    decision = SimpleNamespace(
        event_type=EventType.ANSWERED_WITH_FOLLOWUP,  # defeats the fast path
        extracted={"member_id": "M714598"},
        corrections={},
        guard=GuardType.NONE,
        guard_confidence=0.0,
    )
    with _capture_gate_logs() as records:
        await understand_turn(
            {"awaiting_slot": "member_id"},
            utterance="M714598 and also my deductible",
            awaiting_slot="member_id",
            decision=decision,
        )
    lat = _latency_records(records)
    assert len(lat) == 1
    assert lat[0].fast_path is False


async def test_cached_second_call_emits_no_second_latency_record():
    """The idempotence cache means a repeat caller costs nothing — and is not
    double-counted in the latency metric."""
    from agent.orchestration.turn_gate import understand_turn

    shadow_mod.set_shadow_decoder(shadow_mod.heuristic_decoder)
    state = {"awaiting_slot": "member_id"}
    with _capture_gate_logs() as records:
        await understand_turn(state, utterance="M714598", awaiting_slot="member_id")
        await understand_turn(state, utterance="M714598", awaiting_slot="member_id")
    assert len(_latency_records(records)) == 1


# ── acceptance: fast-path turns add < 1ms ──────────────────────────────────────


async def test_fast_path_gate_overhead_under_one_ms():
    from agent.orchestration.turn_gate import understand_turn

    shadow_mod.set_shadow_decoder(shadow_mod.heuristic_decoder)
    # Warm imports/caches once so the measurement is the steady-state cost.
    await understand_turn({"awaiting_slot": "member_id"}, utterance="warmup", awaiting_slot="member_id")

    n = 50
    start = time.perf_counter()
    for i in range(n):
        # Fresh state each iteration — every call does the full fast-path work.
        await understand_turn(
            {"awaiting_slot": "member_id"}, utterance=f"M71459{i}", awaiting_slot="member_id"
        )
    avg_ms = (time.perf_counter() - start) * 1000.0 / n
    assert avg_ms < 1.0, f"fast-path gate overhead {avg_ms:.3f}ms ≥ 1ms"


# ── acceptance: one understanding decode per turn under the NEW defaults ───────


async def test_full_turn_under_new_defaults_decodes_exactly_once():
    """A hand-coded confirmation turn under the rollout defaults (UNIFIED_VOICE,
    MULTI_INTENT_LIVE, TURNPLAN_DECODE all live) exercises the guards, the
    hand-coded zip_confirmed branch, and the compose path — and still pays
    exactly ONE understanding decode (the idempotence guard's counter)."""
    calls = {"n": 0}
    plan = TurnPlan(
        slot_answer="yes",
        secondary_intents=[
            SecondaryIntent(
                type=SecondaryIntentType.IN_SCOPE_INDEPENDENT,
                owner="benefits_agent",
                verbatim_span="benefits",
            )
        ],
    )

    def _counting_decoder(_state, _utterance, _decision):
        calls["n"] += 1
        return plan

    shadow_mod.set_shadow_decoder(_counting_decoder)
    fixture = {
        "id": "P4-ONE-DECODE",
        "driver": "provider_search_agent",
        "initial_state": {
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
            "app_run_id": "p4-one-decode",
        },
        "turns": [
            {
                "user": "yes, and I also have a question about my benefits",
                "extraction": {"extracted": {"zip_confirmed": "yes"}},
            }
        ],
    }
    run = await run_fixture(fixture, print_latency=False)

    assert calls["n"] == 1, f"expected exactly one understanding decode, got {calls['n']}"
    # The Bug 2 routing still holds on the composed (default) path.
    assert run.turns[0].ai.strip()
    assert run.final_state.get("next_node") == "delivery_management_agent"
    assert run.final_state.get("awaiting_slot") == "delivery_method"
    assert run.final_state.get("zip_code_used") == "94107"
