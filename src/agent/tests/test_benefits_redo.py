"""
Benefits / Care-Coach offer honors a delivery redo immediately (Phase 4 —
BUG-2).

"Send that list to my email instead of fax", voiced while the Care Coach
offer is pending, must hand off to delivery_management NOW — across every
extraction variant — and the round trip must bring the member back to the
Care Coach offer exactly once (never lost, never asked twice in a row).
"""

import logging

import pytest

from agent.agents.benefits.agent import BenefitsAgent
from agent.agents.delivery_management.agent import DeliveryManagementAgent
from agent.core.signal import AgentStatus
from agent.llm.schema import EventType, RequestKind, WorkerResult
from agent.orchestration.fast_path import get_fast_path_route

REDO_UTTERANCE = "please send that list to my email instead of fax"

_BENEFIT_AMOUNTS = {
    "individual_deductible": "750",
    "family_deductible": "2500",
    "coinsurance_percent": "20",
    "individual_oop_max": "3000",
    "family_oop_max": "7000",
}

# ── harness ──────────────────────────────────────────────────────────────────


def _mk_benefits(monkeypatch, result: WorkerResult) -> BenefitsAgent:
    import agent.agents.benefits.agent as ba

    async def fake_extract(*a, **k):
        return result

    monkeypatch.setattr(ba, "extract_benefits_decision", fake_extract)
    monkeypatch.setattr(ba, "get_extraction_llm", lambda: object())
    return BenefitsAgent()


def _state(user=REDO_UTTERANCE, **over):
    state = {
        "messages": [
            {"role": "assistant", "content": "Would you like details about our Care Coach Guides?"},
            {"role": "user", "content": user},
        ],
        "awaiting_slot": "care_coach_response",
        "provider_list_sent": True,
        "benefits_offer_made": True,
        "benefits_explained": True,
        "delivery_method": "fax",
        "fax": "5551234567",
        "email": "emily@example.com",
        "provider_type": "dentist",
        "slot_attempts": {},
        "parked_followups": [],
        **_BENEFIT_AMOUNTS,
    }
    state.update(over)
    return state


# ── BUG-2: mid-offer delivery redo, all extraction variants ──────────────────


class TestBug2CareCoachRedo:
    @pytest.mark.parametrize(
        "result",
        [
            # Ideal extraction (prompt contract)
            WorkerResult(update_target="delivery_method", request_kind=RequestKind.REDO),
            # List-phrased target — resolves via the redo/provider_list →
            # delivery registry equivalence
            WorkerResult(update_target="provider_list", request_kind=RequestKind.REDO),
            # LLM produced nothing — regex fallback fills redo/delivery
            WorkerResult(),
            # Mislabeled WAIT — veto + fill
            WorkerResult(event_type=EventType.WAIT),
            # Misread as a Care Coach decline — the redo hook runs BEFORE the
            # yes/no extraction, so the hand-off still wins
            WorkerResult(extracted={"care_coach_response": "no"}),
        ],
    )
    async def test_redo_hands_off_to_delivery(self, monkeypatch, result):
        agent = _mk_benefits(monkeypatch, result)
        out = await agent.run(_state())
        assert out["next_node"] == "delivery_management_agent"
        assert out["pending_cross_agent_request"] == {
            "kind": "redo",
            "target": "delivery",  # canonical topic — delivery's redo_active keys off it
            "return_to_agent": "benefits_agent",
            "return_awaiting": "care_coach_response",
        }

    @pytest.mark.parametrize(
        "utterance",
        [
            REDO_UTTERANCE,
            "can you resend the list",
            "send it to my email instead",
            "actually email is better",
        ],
    )
    async def test_regex_only_variants_never_park(self, monkeypatch, utterance, caplog):
        # The "unknown topic → park" branch must be unreachable for any
        # delivery-phrased redo.
        agent = _mk_benefits(monkeypatch, WorkerResult())
        with caplog.at_level(logging.WARNING):
            out = await agent.run(_state(user=utterance))
        assert out["next_node"] == "delivery_management_agent"
        assert not out.get("parked_followups")
        assert "in just a moment" not in out.get("messages", {}).get("content", "").lower()
        assert not any("parked as question" in r.message for r in caplog.records)

    async def test_unknown_topic_still_parks_with_warning(self, monkeypatch, caplog):
        # Genuinely unknown topics keep the degrade path — now at WARNING
        # with the raw target for observability.
        result = WorkerResult(update_target="claim history", request_kind=RequestKind.REPLAY)
        agent = _mk_benefits(monkeypatch, result)
        # The "agent" logger is configured propagate=false — re-enable so
        # caplog's root handler sees the record.
        monkeypatch.setattr(logging.getLogger("agent"), "propagate", True)
        with caplog.at_level(logging.WARNING):
            out = await agent.run(_state(user="can you go over my claim history again?"))
        assert out["parked_followups"][0]["kind"] == "question"
        assert out["awaiting_slot"] == "care_coach_response"
        warning = next(r for r in caplog.records if "parked as question" in r.message)
        assert warning.levelno == logging.WARNING
        assert getattr(warning, "target", "") == "claim history"

    async def test_care_coach_yes_no_still_works(self, monkeypatch):
        # Control: the redo hook must not swallow plain answers.
        agent = _mk_benefits(monkeypatch, WorkerResult(extracted={"care_coach_response": "no"}))
        out = await agent.run(_state(user="no thank you"))
        assert out["last_agent_signal"]["status"] == AgentStatus.COMPLETE
        assert out.get("proactive_offer_available") is not True


# ── BUG-2 round trip: benefits → delivery re-dispatch → back to the offer ────


async def test_round_trip_regex_only_redo(monkeypatch):
    # (a) benefits hands off — extraction produced NOTHING; the regex
    # fallback alone carries the turn.
    import agent.agents.delivery_management.agent as dma

    benefits = _mk_benefits(monkeypatch, WorkerResult())
    state = _state(member_status_verify=True)
    hop = await benefits.run(state)
    assert hop["next_node"] == "delivery_management_agent"
    assert hop["pending_cross_agent_request"]["target"] == "delivery"
    # The hop turn itself asks nothing — the owner speaks next.
    assert "messages" not in hop

    # (b) delivery finishes the re-dispatch — silent COMPLETE, request kept
    async def fake_dispatch(agent, st, method, dest):
        return None

    monkeypatch.setattr(dma, "dispatch_provider_list", fake_dispatch)
    delivery_state = {**state, **{k: v for k, v in hop.items() if k != "messages"}}
    done = await DeliveryManagementAgent()._proceed_to_dispatch(delivery_state, "email", "emily@example.com")
    assert done["last_agent_signal"]["status"] == AgentStatus.COMPLETE

    # (c) fast-path returns control to benefits
    fp_state = {
        **delivery_state,
        "last_agent_signal": done["last_agent_signal"],
        "active_agent": "delivery_management_agent",
        "delivery_method": "email",
    }
    assert get_fast_path_route(fp_state) == "benefits_agent"

    # (d) orchestrator consumes the request and restores the offer slot
    import agent.orchestration.orchestration as orch

    monkeypatch.setattr(orch, "get_routing_llm", lambda: object())
    updates = await orch.Orchestrator().run(fp_state)
    assert updates["awaiting_slot"] == "care_coach_response"
    assert updates["slot_update_resume"] is True

    # (e) benefits resumes: brief ack + the Care Coach offer re-asked ONCE
    resumed = {**fp_state, **updates, "messages": [{"role": "user", "content": "sure"}]}
    result = await BenefitsAgent().run(resumed)
    message = result["messages"]["content"]
    assert "email" in message.lower()  # the re-send is acknowledged
    assert "coach" in message.lower()  # the offer is not lost
    assert message.count("?") == 1  # exactly one ask — never twice in a row
    assert result["awaiting_slot"] == "care_coach_response"
    assert result["slot_update_resume"] is False
