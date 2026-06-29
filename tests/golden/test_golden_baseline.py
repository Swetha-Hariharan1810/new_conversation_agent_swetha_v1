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


async def test_uat_007_zip_disputed_blocks_dispatch_but_request_still_silently_dropped():
    """Phase 1: F2 is CLOSED (no dispatch on a disputed ZIP); F1 is still OPEN
    (the ZIP-update request is silently dropped — that is Phase 3)."""
    fixture = load_fixture("uat_007_multi_intent")
    run = await run_fixture(fixture)

    turn0 = run.turns[0]
    turn1 = run.turns[1]

    # ── F1 (STILL OPEN): the ZIP-update request is never acknowledged ───────────
    # The member said "Fax, but I need to update my ZIP code." Turn 0's reply
    # confirms the fax and never mentions the ZIP. PHASE-FLIP: Phase 3.
    assert turn0.awaiting_slot == "fax_confirmed"
    assert run.final_state.get("delivery_method") == "fax"
    assert not re.search(r"zip", turn0.ai, re.IGNORECASE), (
        f"F1 fixed early? — ZIP acknowledged at turn 0 (that's Phase 3): {turn0.ai!r}"
    )

    # ── F2 (CLOSED in Phase 1): delivery on the disputed ZIP is impossible ──────
    # The stale-delivery gate refuses to dispatch and redirects to the ZIP owner.
    assert run.recorder.count("dispatch_provider_list") == 0, (
        "F2 regressed — a provider list was dispatched while the ZIP was disputed."
    )
    assert turn1.awaiting_slot == "zip_code"
    assert run.final_state.get("next_node") == "provider_search_agent"
    assert re.search(r"zip", turn1.ai, re.IGNORECASE), (
        f"expected a redirect asking for the current ZIP, got {turn1.ai!r}"
    )
    assert not run.final_state.get("provider_list_sent")
    assert run.final_state.get("zip_code") == "94107"  # unchanged and NOT dispatched

    # Latency probe produced a wall-clock number for every turn (no new LLM call).
    assert len(run.latencies_ms) == 2
    assert all(ms >= 0 for ms in run.latencies_ms)


# ──────────────────────────────────────────────────────────────────────────────
# Conversation-wide: the same drop at the provider-search slot stage
# ──────────────────────────────────────────────────────────────────────────────


async def test_provider_search_fresh_request_dropped():
    fixture = load_fixture("slot_interrupt_fresh_request")
    run = await run_fixture(fixture)

    turn0 = run.turns[0]
    # provider_type captured; agent advanced to ZIP confirmation.
    assert run.final_state.get("provider_type") == "Pediatrician"
    assert turn0.awaiting_slot == "zip_confirmed"
    # The bundled in-scope benefits question was dropped — never acknowledged.
    assert not re.search(r"deductible", turn0.ai, re.IGNORECASE), (
        f"F1 regressed (good!) — deductible question acknowledged: {turn0.ai!r}"
    )
    # The agent reply is the ZIP confirmation (mentions the ZIP on file).
    assert re.search(r"94107", turn0.ai), f"expected ZIP confirmation prompt, got {turn0.ai!r}"


# ──────────────────────────────────────────────────────────────────────────────
# Conversation-wide: answer + correction in one breath during verification
# ──────────────────────────────────────────────────────────────────────────────


async def test_mid_verification_correction_dropped():
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

    fake_recorder_unused = None  # collector path touches no storage tools
    value, interrupt = await agent._collect_slot(
        dict(probe["state"]),
        config,
        messages,
        pre_extracted=probe["pre_extracted"],
        decision=decision,
    )

    # DOB confirmed cleanly...
    assert value == "09/03/1985"
    assert interrupt is None
    # ...but the bundled member_id CORRECTION was dropped: the collector never
    # touched the member_id slot (no acknowledgement, no new value applied).
    member_id_slot = agent._slots.get("member_id")
    assert member_id_slot is None or member_id_slot.last_value != "M999999", (
        "F1 regressed (good!) — the member_id correction was applied; baseline drops it."
    )
    assert "member_id" not in agent._newly_confirmed
    assert fake_recorder_unused is None


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
    # known_failures must be present (possibly empty for the GREEN control case).
    assert "known_failures" in fixture
    # Every scripted extraction must build into a valid structured result.
    for turn in fixture["turns"]:
        build_result(turn.get("extraction"), schema=fixture.get("schema", "worker"))
