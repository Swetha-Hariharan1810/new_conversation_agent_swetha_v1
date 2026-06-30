"""
test_claim_flow_guard.py — claim-flow parity for the stale-value guard.

The provider flow refuses to deliver a provider list on a disputed ZIP (Phase 1).
The claim flow has the same shape: the upload link and the Personal Guide outreach
are both keyed on the claim reference number, so they must never be actioned on a
disputed reference. These deterministic tests prove the gate holds regardless of
classification (the claim-flow analog of S6) and that the resolver flips the right
dirty artifacts on an invalidating reference correction.
"""

from __future__ import annotations

import pytest

import tests.golden  # noqa: F401 — ensures src/ is on sys.path
from agent.llm.schema import Correction, SecondaryIntent, SecondaryIntentType, TurnPlan
from agent.orchestration.resolver import CORRECTION_ACK, resolve_turn
from tests.golden.driver import ToolRecorder, deterministic_env

pytestmark = pytest.mark.regression


# ── registry: the claim reference owns/invalidates the claim artifacts ────────


def test_reference_number_invalidates_claim_artifacts():
    from agent.orchestration.invalidation import artifacts_invalidated_by
    from agent.orchestration.registry import INVALIDATION_MAP, owner_of

    assert set(artifacts_invalidated_by("reference_number")) == {"upload_link", "personal_guide_outreach"}
    assert owner_of("reference_number") == "claim_adjustment_agent"
    assert owner_of("upload_link") == "records_coordination_agent"
    assert owner_of("personal_guide_outreach") == "records_coordination_agent"
    assert "reference_number" in INVALIDATION_MAP


# ── resolver: an invalidating reference correction flips dirty + rewinds ──────


def test_resolver_flips_claim_artifacts_on_reference_correction():
    plan = TurnPlan(
        slot_answer="yes",
        correction=Correction(field="reference_number", owner="claim_adjustment_agent", new_value="42695818"),
    )
    out = resolve_turn(
        plan,
        {"awaiting_slot": "upload_consent", "dirty_artifacts": {}, "intent_queue": []},
        utterance="yes, but my reference number was wrong",
    )
    assert out.speech_act == CORRECTION_ACK
    assert out.dirty.get("upload_link") is True
    assert out.dirty.get("personal_guide_outreach") is True
    assert out.rewind_target == "claim_adjustment_agent"


# ── gate: upload link refused on a disputed reference (S6 analog) ─────────────


@pytest.mark.guards
async def test_gate_blocks_upload_link_when_reference_disputed():
    from agent.agents.records_coordination.agent import RecordsCoordinationAgent

    recorder = ToolRecorder()
    state = {
        "messages": [{"role": "assistant", "content": "Shall I send the upload link?"}],
        "member_status_verify": True,
        "member_id": "M714598",
        "reference_number": "42695817",
        "claim_status": "open for Review",
        "email": "daniel.reed@example.com",
        "dirty_artifacts": {"upload_link": True},  # reference disputed upstream
        "slot_attempts": {},
        "app_run_id": "claim-gate",
    }
    agent = RecordsCoordinationAgent.from_state(state)
    with deterministic_env(_fake_llm(), recorder):
        out = await agent._send_link_and_proceed(state, "daniel.reed@example.com")

    # No link dispatched; routed back to re-resolve the reference number.
    assert recorder.count("send_claim_upload_link") == 0
    assert out["next_node"] == "claim_adjustment_agent"
    assert out["awaiting_slot"] == "reference_number"
    assert out["reference_number"] == ""  # forces re-collection


@pytest.mark.guards
async def test_gate_blocks_personal_guide_when_reference_disputed():
    from agent.agents.records_coordination.agent import RecordsCoordinationAgent

    recorder = ToolRecorder()
    state = {
        "messages": [{"role": "assistant", "content": "Shall I have a Personal Guide reach out?"}],
        "member_status_verify": True,
        "member_id": "M714598",
        "reference_number": "42695817",
        "claim_status": "open for Review",
        "dirty_artifacts": {"personal_guide_outreach": True},
        "slot_attempts": {},
        "app_run_id": "claim-gate-2",
    }
    agent = RecordsCoordinationAgent.from_state(state)
    with deterministic_env(_fake_llm(), recorder):
        out = await agent._trigger_guide_and_proceed(state)

    assert recorder.count("trigger_claim_personal_guide") == 0
    assert out["next_node"] == "claim_adjustment_agent"
    assert out["awaiting_slot"] == "reference_number"


# ── gate allows the action when the reference is clean ───────────────────────


async def test_upload_link_sent_when_reference_clean():
    from agent.agents.records_coordination.agent import RecordsCoordinationAgent

    recorder = ToolRecorder()
    state = {
        "messages": [{"role": "assistant", "content": "Shall I send the upload link?"}],
        "member_status_verify": True,
        "member_id": "M714598",
        "reference_number": "42695817",
        "claim_status": "open for Review",
        "email": "daniel.reed@example.com",
        "dirty_artifacts": {},  # clean
        "slot_attempts": {},
        "app_run_id": "claim-clean",
    }
    agent = RecordsCoordinationAgent.from_state(state)
    with deterministic_env(_fake_llm(), recorder):
        out = await agent._send_link_and_proceed(state, "daniel.reed@example.com")

    assert recorder.count("send_claim_upload_link") == 1
    assert out.get("upload_link_sent") is True
    assert out.get("awaiting_slot") == "personal_guide_consent"


# ── S6 analog: gate holds even if a reference correction is misclassified ─────


@pytest.mark.guards
def test_misclassified_reference_correction_still_cannot_act():
    # The decode MISLABELS the reference correction as an in-scope independent.
    mislabeled = TurnPlan(
        slot_answer="yes",
        secondary_intents=[
            SecondaryIntent(
                type=SecondaryIntentType.IN_SCOPE_INDEPENDENT,
                owner="claim_adjustment_agent",
                verbatim_span="reference number was wrong",
            )
        ],
    )
    out = resolve_turn(
        mislabeled,
        {"awaiting_slot": "upload_consent", "dirty_artifacts": {"upload_link": True}},
        utterance="yes, but my reference number was wrong",
    )
    # Resolver does not flip dirty from the mislabeled intent...
    assert out.dirty == {}
    # ...but the gate reads ONLY dirty_artifacts, which is already set, so the
    # upload link is still blocked (proven by test_gate_blocks_* above).


def _fake_llm():
    """A FakeLLM whose extraction queue is never consulted (gate returns before
    any LLM call)."""
    from tests.golden.driver import FakeLLM

    return FakeLLM()
