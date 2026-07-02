"""
test_golden_baseline.py — Phase 0 golden baseline for the context-retention
(multi-intent) defect.

These tests are GREEN now and describe *today's* behavior — including the two
known UAT-007 failures and the conversation-wide multi-intent drops. As each
later phase lands, the failing assertions flip (marked inline with PHASE-FLIP).

Determinism: the LLM and Salesforce seams are replaced with scripted fakes
(see driver.py). No env vars, no network, no checkpointer. Run with:

    uv run pytest tests/golden -s

The ``-s`` flag surfaces the per-turn wall-clock latency probe printed by the
driver — the seed of the latency bench Phase 3/4 will assert a budget against.
"""

from __future__ import annotations

import re

import pytest

import tests.golden  # noqa: F401 — ensures src/ is on sys.path
from agent.orchestration.registry import queue_owners
from tests.golden.driver import (
    build_result,
    load_fixture,
    run_fixture,
)

pytestmark = pytest.mark.regression


def _signal(state: dict) -> dict:
    return state.get("last_agent_signal") or {}


# ──────────────────────────────────────────────────────────────────────────────
# UAT-007 — the headline defect, both known failures asserted explicitly
# ──────────────────────────────────────────────────────────────────────────────


async def test_uat_007_zip_request_acknowledged_and_routed():
    """Phase 3B: F1 is CLOSED — the ZIP-update request is acknowledged in the SAME
    turn the member says 'Fax, but I need to update my ZIP code', and the call
    routes to update the ZIP and rebuild before delivery. F2 stays closed."""
    fixture = load_fixture("uat_007_multi_intent")
    run = await run_fixture(fixture)

    turn0 = run.turns[0]

    # ── F1 CLOSED: both intents acknowledged in one templated reply ─────────────
    assert re.search(r"zip", turn0.ai, re.IGNORECASE), f"ZIP not acknowledged: {turn0.ai!r}"
    assert re.search(r"fax", turn0.ai, re.IGNORECASE), f"fax answer not acknowledged: {turn0.ai!r}"

    # Slot answer accepted (not dropped) and routed to the ZIP owner to re-resolve.
    assert run.final_state.get("delivery_method") == "fax"
    assert run.final_state.get("next_node") == "provider_search_agent"
    assert turn0.awaiting_slot == "zip_code"
    assert run.final_state.get("dirty_artifacts", {}).get("provider_list") is True

    # ── F2 stays closed: nothing dispatched on the disputed ZIP ─────────────────
    assert run.recorder.count("dispatch_provider_list") == 0
    assert not run.final_state.get("provider_list_sent")

    # The silent drop is gone.
    assert run.dropped_request_count == 0

    # One understanding decode; per-turn wall-clock stays within a generous
    # deterministic budget (real budget asserted in Phase 4).
    assert len(run.latencies_ms) == 1
    assert run.latencies_ms[0] < 250


# ──────────────────────────────────────────────────────────────────────────────
# Conversation-wide: the same drop at the provider-search slot stage
# ──────────────────────────────────────────────────────────────────────────────


async def test_provider_search_fresh_request_acknowledged():
    """Phase 3C: the bundled in-scope request (a benefits question) is no longer
    dropped — it is acknowledged and parked for draining, while the slot answer is
    accepted. The drop is gone."""
    fixture = load_fixture("slot_interrupt_fresh_request")
    run = await run_fixture(fixture)

    turn0 = run.turns[0]
    # Slot answer accepted.
    assert run.final_state.get("provider_type") == "Pediatrician"
    # Secondary acknowledged (multi-intent ack mentions the benefits question)...
    assert re.search(r"benefits", turn0.ai, re.IGNORECASE), f"benefits not acknowledged: {turn0.ai!r}"
    # ...and parked for draining (enqueued), not dropped.
    assert "benefits_agent" in queue_owners(run.final_state.get("intent_queue"))
    assert run.dropped_request_count == 0


# ──────────────────────────────────────────────────────────────────────────────
# Conversation-wide: answer + correction in one breath during verification
# ──────────────────────────────────────────────────────────────────────────────


async def test_mid_verification_correction_acknowledged_and_rewound():
    """Phase 3D: a correction bundled with a slot answer is no longer dropped on
    the valid-extraction path — the DOB is confirmed, the Member ID correction is
    acknowledged, and the turn rewinds to the corrected value's owner."""
    fixture = load_fixture("mid_verification_correction")
    probe = fixture["collector_probe"]

    # Drive the load-bearing per-turn collector directly (the "relevant agent"
    # component). No graph, no LLM-1 — the decision object is supplied verbatim.
    from agent.core.agent import BaseAgent
    from agent.core.slot_manager import _InternalSlotConfig
    from agent.slots.normalizers import normalize_dob
    from agent.slots.types import SlotType
    from agent.slots.validators import validate_dob

    class _ProbeAgent(BaseAgent):
        AGENT_NAME = "verification_agent"

        async def run(self, state):  # pragma: no cover - not used
            return {}

    agent = _ProbeAgent.from_state(probe["state"])
    config = _InternalSlotConfig(
        slot_name="dob",
        prompt="",
        normalizer=normalize_dob,
        validator=validate_dob,
        slot_type=SlotType.DOB,
    )
    decision = build_result(probe["decision"])
    messages = [
        {"role": "assistant", "content": "And your date of birth?"},
        {"role": "user", "content": fixture["turns"][0]["user"]},
    ]

    value, interrupt = await agent._collect_slot(
        dict(probe["state"]),
        config,
        messages,
        pre_extracted=probe["pre_extracted"],
        decision=decision,
    )

    # DOB confirmed...
    assert value == "09/03/1985"
    # ...AND the Member ID correction is acknowledged and rewound to its owner.
    assert interrupt is not None
    assert interrupt["awaiting_slot"] == "member_id"
    assert interrupt["next_node"] == "verification_agent"
    ack = interrupt["messages"]["content"] if isinstance(interrupt.get("messages"), dict) else ""
    assert re.search(r"member id", ack, re.IGNORECASE), f"correction not acknowledged: {ack!r}"


# ──────────────────────────────────────────────────────────────────────────────
# Safety injected mid-flow — handled correctly today (GREEN floor, no regression)
# ──────────────────────────────────────────────────────────────────────────────


async def test_safety_phrase_classified_and_escalated():
    fixture = load_fixture("safety_injected_midflow")
    run = await run_fixture(fixture)

    state = run.final_state
    assert state.get("next_node") == "escalation_agent"
    reason = (_signal(state).get("escalation_reason") or "")
    assert "self_harm" in reason.lower(), f"expected self-harm escalation reason, got {reason!r}"
    # A spoken outcome was produced (escalation pre-message), not dropped.
    assert (state.get("escalation_pre_message") or "").strip(), "safety produced no spoken outcome"
    # No provider list was dispatched on a safety turn.
    assert run.recorder.count("dispatch_provider_list") == 0


# ──────────────────────────────────────────────────────────────────────────────
# Unsupported question invisible to the schema — silently re-asked (KNOWN FAILURE)
# ──────────────────────────────────────────────────────────────────────────────


async def test_unsupported_question_silently_reasked():
    fixture = load_fixture("unsupported_injected_midflow")
    run = await run_fixture(fixture)

    turn0 = run.turns[0]
    # Agent re-reads the same fax confirmation and never addresses the copay question.
    assert turn0.awaiting_slot == "fax_confirmed"
    assert re.search(r"fax", turn0.ai, re.IGNORECASE), f"expected a fax re-ask, got {turn0.ai!r}"
    assert not re.search(r"copay|specialist", turn0.ai, re.IGNORECASE), (
        f"F1 regressed (good!) — copay question acknowledged: {turn0.ai!r}"
    )
    assert run.recorder.count("dispatch_provider_list") == 0


# ──────────────────────────────────────────────────────────────────────────────
# Latency probe — every agent-driven fixture prints a per-turn wall-clock number
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "fixture_name",
    [
        "uat_007_multi_intent",
        "slot_interrupt_fresh_request",
        "safety_injected_midflow",
        "unsupported_injected_midflow",
    ],
)
async def test_latency_probe_emits_per_turn_wall_clock(fixture_name, capsys):
    fixture = load_fixture(fixture_name)
    run = await run_fixture(fixture)

    # Every turn has a measured wall-clock latency.
    assert run.latencies_ms
    assert all(isinstance(ms, float) and ms >= 0 for ms in run.latencies_ms)

    # And the probe line was printed for each turn (Phase 3/4 will assert a budget).
    out = capsys.readouterr().out
    assert out.count("[golden-latency]") == len(fixture["turns"])
    assert "wall_clock=" in out


# ──────────────────────────────────────────────────────────────────────────────
# Fixture hygiene — every fixture is well-formed and replayable
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "fixture_name",
    [
        "uat_007_multi_intent",
        "slot_interrupt_fresh_request",
        "mid_verification_correction",
        "safety_injected_midflow",
        "unsupported_injected_midflow",
    ],
)
def test_fixture_is_well_formed(fixture_name):
    fixture = load_fixture(fixture_name)
    assert fixture["id"]
    assert fixture["title"]
    assert fixture["driver"]
    assert "turns" in fixture and fixture["turns"]
    # A failure ledger must be present: open defects (known_failures) and/or
    # closed ones (resolved_failures, once a phase fixes them).
    assert "known_failures" in fixture or "resolved_failures" in fixture
    # Every scripted extraction must build into a valid structured result.
    for turn in fixture["turns"]:
        build_result(turn.get("extraction"), schema=fixture.get("schema", "worker"))
