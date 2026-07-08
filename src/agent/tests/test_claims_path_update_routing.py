"""
Claims-path parity (Phase 7): notification_setup, claim_adjustment, and
records_coordination get the Phase 1–5 patterns — reconcile wiring, hoisted
update routing, never-verbatim-repeat guards, notification channel switch,
redo/replay capabilities, and follow_up parked/live routing — each scenario
parametrized over extraction variants (mirroring Phase 0's structure).
"""

import pytest

from agent.agents.claim_adjustment.agent import ClaimAdjustmentAgent
from agent.agents.follow_up.agent import FollowUpAgent
from agent.agents.notification_setup.agent import NotificationSetupAgent
from agent.agents.records_coordination.agent import RecordsCoordinationAgent
from agent.core.slot_ownership import resolve_capability
from agent.llm.schema import (
    EventType,
    FollowUpIntent,
    FollowUpResult,
    RequestKind,
    WorkerResult,
)
from agent.orchestration.fast_path import get_fast_path_route

# ── harnesses ────────────────────────────────────────────────────────────────


def _mk_claim(monkeypatch, result: WorkerResult) -> ClaimAdjustmentAgent:
    import agent.agents.claim_adjustment.agent as caa

    async def fake_extract(*a, **k):
        return result.model_copy(deep=True)

    monkeypatch.setattr(caa, "extract_claim_adjustment_decision", fake_extract)
    monkeypatch.setattr(caa, "get_extraction_llm", lambda: object())
    return ClaimAdjustmentAgent()


def _claim_state(user: str, **over) -> dict:
    state = {
        "messages": [
            {"role": "assistant", "content": "Could I get the reference number from your letter?"},
            {"role": "user", "content": user},
        ],
        "awaiting_slot": "reference_number",
        "call_intent": "claim_services",
        "member_status_verify": True,
        "first_name": "Emily",
        "last_name": "Carter",
        "zip_code": "90210",
        "parked_followups": [],
    }
    state.update(over)
    return state


def _mk_notification(monkeypatch, result: WorkerResult) -> NotificationSetupAgent:
    import agent.agents.notification_setup.agent as nsa

    async def fake_extract(*a, **k):
        return result.model_copy(deep=True)

    monkeypatch.setattr(nsa, "extract_notification_decision", fake_extract)
    monkeypatch.setattr(nsa, "get_extraction_llm", lambda: object())
    return NotificationSetupAgent()


def _notification_state(user: str, awaiting="phone_confirmed", **over) -> dict:
    state = {
        "messages": [
            {"role": "assistant", "content": "Is 555-987-6543 the right number for SMS updates?"},
            {"role": "user", "content": user},
        ],
        "awaiting_slot": awaiting,
        "call_intent": "claim_services",
        "member_status_verify": True,
        "notification_channel": "sms",
        "phone_number": "5559876543",
        "email": "emily@example.com",
        "zip_code": "90210",
        "parked_followups": [],
    }
    state.update(over)
    return state


# ══ 1. ZIP / last-name update while awaiting reference_number ════════════════


class TestClaimAdjustmentUpdateRouting:
    @pytest.mark.parametrize(
        "result,utterance",
        [
            (WorkerResult(update_target="zip_code", request_kind=RequestKind.UPDATE), "my zip changed"),
            (WorkerResult(), "wait — my zip code changed, i moved"),
            (WorkerResult(event_type=EventType.WAIT), "hold on, my zip code changed"),
            (WorkerResult(event_type=EventType.AMBIGUOUS), "my address changed"),
        ],
    )
    async def test_zip_update_routes_to_provider_search(self, monkeypatch, result, utterance):
        agent = _mk_claim(monkeypatch, result)
        out = await agent.run(_claim_state(utterance))
        assert out["next_node"] == "provider_search_agent"
        assert out["pending_cross_agent_request"] == {
            "kind": "update",
            "target": "zip_code",
            "return_to_agent": "claim_adjustment_agent",
            "return_awaiting": "reference_number",
        }

    @pytest.mark.parametrize(
        "result,utterance",
        [
            (
                WorkerResult(update_target="last_name", request_kind=RequestKind.UPDATE),
                "i need to update my last name",
            ),
            (WorkerResult(), "i need to update my last name"),
            (WorkerResult(event_type=EventType.WAIT), "wait, my last name is wrong"),
        ],
    )
    async def test_last_name_update_routes_to_verification(self, monkeypatch, result, utterance):
        # Identity updates mid-claims-flow route to verification with the slot
        # cleared for re-collection — never a park, never a decline.
        agent = _mk_claim(monkeypatch, result)
        out = await agent.run(_claim_state(utterance))
        assert out["next_node"] == "verification_agent"
        assert out["pending_cross_agent_request"]["target"] == "last_name"
        assert out["pending_cross_agent_request"]["return_awaiting"] == "reference_number"
        assert out["last_name"] == ""  # re-collected, not skipped
        assert out["member_status_verify"] is False  # re-verify required
        assert out["name_confirmed"] is False  # readback re-runs

    async def test_unclassifiable_turn_still_retries(self, monkeypatch):
        async def fake_generate(**kwargs):
            return "Could you read me the reference number from the letter?"

        import agent.llm.response_generator as rg

        monkeypatch.setattr(rg, "generate_recovery_message", fake_generate)
        agent = _mk_claim(monkeypatch, WorkerResult())
        out = await agent.run(_claim_state("hmm let me think about that"))
        assert out["awaiting_slot"] == "reference_number"
        assert "pending_cross_agent_request" not in out


# ══ 2. "wait — my address changed" while awaiting phone_confirmed ════════════


class TestNotificationUpdateRouting:
    @pytest.mark.parametrize(
        "result",
        [
            WorkerResult(update_target="zip_code", request_kind=RequestKind.UPDATE),
            WorkerResult(),
            WorkerResult(event_type=EventType.WAIT),
            # Misread as a decline of the phone — the routed update wins
            # before the "no" path can ask for a new phone number.
            WorkerResult(extracted={"contact_confirmed": "no"}),
        ],
    )
    async def test_address_change_routes_never_repeats_readback(self, monkeypatch, result):
        agent = _mk_notification(monkeypatch, result)
        out = await agent.run(_notification_state("wait — my address changed"))
        assert out["next_node"] == "provider_search_agent"
        assert out["pending_cross_agent_request"]["target"] == "zip_code"
        assert out["pending_cross_agent_request"]["return_awaiting"] == "phone_confirmed"
        # Never the verbatim phone read-back and never the new-phone ask.
        message = out["messages"]["content"].lower()
        assert "555-987" not in message
        assert "correct phone" not in message


# ══ 3. "actually email me instead" while awaiting phone_confirmed ════════════


class TestNotificationChannelSwitch:
    @pytest.mark.parametrize(
        "result,utterance",
        [
            (WorkerResult(extracted={"notification_method": "email"}), "actually email me instead"),
            (WorkerResult(), "actually email me instead"),
            (WorkerResult(), "email works better for me"),
            (WorkerResult(event_type=EventType.WAIT), "actually just email me instead"),
            (WorkerResult(event_type=EventType.AMBIGUOUS), "can you email me instead"),
        ],
    )
    async def test_switch_to_email_never_asks_for_phone(self, monkeypatch, result, utterance):
        agent = _mk_notification(monkeypatch, result)
        out = await agent.run(_notification_state(utterance))
        assert out["awaiting_slot"] == "email_confirmed"
        assert out["notification_channel"] == "email"
        message = out["messages"]["content"].lower()
        assert "emily at example dot com" in message
        assert "phone" not in message  # never "what is the correct phone number?"

    async def test_switch_carries_new_email_value(self, monkeypatch):
        result = WorkerResult(extracted={"email": "new.addr@example.com"})
        agent = _mk_notification(monkeypatch, result)
        out = await agent.run(_notification_state("just email me at new.addr@example.com"))
        assert out["awaiting_slot"] == "email_confirmed"
        assert out["pending_email"] == "new.addr@example.com"
        assert out["notification_channel"] == "email"

    async def test_phone_dispute_still_declines_honestly(self, monkeypatch):
        # phone_number stays human_only in the registry — a plain decline of
        # the SF phone keeps the existing new-phone ask, no switch.
        result = WorkerResult(extracted={"contact_confirmed": "no"})
        agent = _mk_notification(monkeypatch, result)
        out = await agent.run(_notification_state("no that's not my number anymore"))
        assert out["awaiting_slot"] == "phone"
        assert out.get("notification_channel", "sms") == "sms"

    async def test_switch_from_email_back_to_sms(self, monkeypatch):
        agent = _mk_notification(monkeypatch, WorkerResult())
        state = _notification_state(
            "actually text me instead", awaiting="email_confirmed", notification_channel="email"
        )
        out = await agent.run(state)
        assert out["awaiting_slot"] == "phone_confirmed"
        assert out["notification_channel"] == "sms"
        assert "555-987-6543" in out["messages"]["content"]


# ══ 4. Post-setup "change my notification to email" in follow_up ═════════════


def _followup_state(user: str, **over) -> dict:
    state = {
        "messages": [
            {"role": "assistant", "content": "Aside from this, is there anything else I can help with?"},
            {"role": "user", "content": user},
        ],
        "follow_up_turn_count": 1,
        "follow_up_cannot_answer_count": 0,
        "call_intent": "claim_services",
        "member_status_verify": True,
        "notification_channel": "sms",
        "claim_notification_contact": "5559876543",
        "claim_flow_complete": True,
        "claim_status": "in review with our adjustment team",
        "reference_number": "CLM443019",
        "last_update_date": "July 1",
        "email": "emily@example.com",
        "phone_number": "5559876543",
        "parked_followups": [],
    }
    state.update(over)
    return state


def _mk_followup(monkeypatch, result: FollowUpResult | None):
    import agent.agents.follow_up.agent as fua

    async def fake_extract(*a, **k):
        if result is None:  # pragma: no cover
            raise AssertionError("LLM path must not run")
        return result.model_copy(deep=True)

    monkeypatch.setattr(fua, "extract_follow_up_decision", fake_extract)
    monkeypatch.setattr(fua, "get_follow_up_llm", lambda: object())
    return FollowUpAgent()


class TestNotificationRedoRoundTrip:
    def test_registry_entries(self):
        assert resolve_capability("redo", "notification").agent == "notification_setup_agent"
        assert resolve_capability("redo", "notification_method").agent == "notification_setup_agent"
        assert resolve_capability("replay", "claim_status").agent == "claim_adjustment_agent"
        assert resolve_capability("replay", "my claim").agent == "claim_adjustment_agent"
        assert resolve_capability("replay", "notification") is None  # redo-only capability

    @pytest.mark.parametrize(
        "result",
        [
            FollowUpResult(
                follow_up_intent=FollowUpIntent.UPDATE_REQUEST,
                request_kind=RequestKind.REDO,
                request_target="notification",
            ),
            FollowUpResult(
                follow_up_intent=FollowUpIntent.UPDATE_REQUEST,
                request_kind=RequestKind.UPDATE,
                request_target="notification_method",
            ),
        ],
    )
    async def test_follow_up_hops_to_notification_setup(self, monkeypatch, result):
        agent = _mk_followup(monkeypatch, result)
        out = await agent.run(_followup_state("change my notification to email"))
        assert out["next_node"] == "notification_setup_agent"
        assert out["pending_cross_agent_request"]["kind"] == "redo"
        assert out["pending_cross_agent_request"]["target"] == "notification"
        assert out["pending_cross_agent_request"]["return_to_agent"] == "follow_up_agent"

    async def test_round_trip_back_to_follow_up(self, monkeypatch):
        import agent.agents.notification_setup.agent as nsa

        # (a) follow_up hands off
        follow_up = _mk_followup(
            monkeypatch,
            FollowUpResult(
                follow_up_intent=FollowUpIntent.UPDATE_REQUEST,
                request_kind=RequestKind.REDO,
                request_target="notification",
            ),
        )
        state = _followup_state("change my notification to email")
        # N2 is also complete post-setup — the redo must still re-enter.
        state["claim_timeline_notification_channel"] = "sms"
        hop = await follow_up.run(state)
        assert hop["next_node"] == "notification_setup_agent"

        # (b) notification_setup re-collects the method from the same turn —
        # never re-running the timeline question.
        async def fake_extract_method(*a, **k):
            return WorkerResult(extracted={"notification_method": "email"})

        monkeypatch.setattr(nsa, "extract_notification_decision", fake_extract_method)
        monkeypatch.setattr(nsa, "get_extraction_llm", lambda: object())
        ns_state = {**state, **{k: v for k, v in hop.items() if k != "messages"}}
        confirm = await NotificationSetupAgent().run(ns_state)
        assert confirm["awaiting_slot"] == "email_confirmed"
        assert "emily at example dot com" in confirm["messages"]["content"].lower()

        # (c) member confirms — preference saved, redo closed, control handed
        # back to follow_up with NO timeline question.
        async def fake_extract_yes(*a, **k):
            return WorkerResult(extracted={"contact_confirmed": "yes"})

        async def fake_save(agent, st, method, contact):
            return None

        monkeypatch.setattr(nsa, "extract_notification_decision", fake_extract_yes)
        monkeypatch.setattr(nsa, "save_notification_preference", fake_save)
        done_state = {
            **ns_state,
            **{k: v for k, v in confirm.items() if k != "messages"},
            "messages": [
                {"role": "assistant", "content": confirm["messages"]["content"]},
                {"role": "user", "content": "yes that's correct"},
            ],
        }
        done = await NotificationSetupAgent().run(done_state)
        assert done["next_node"] == "follow_up_agent"
        assert done["notification_channel"] == "email"
        assert done["pending_cross_agent_request"] == {}
        message = done["messages"]["content"].lower()
        assert "timeline" not in message  # the timeline question is never re-run
        assert done["awaiting_slot"] == ""


# ══ 5. Parked "when will I hear about my claim?" → claim_status replay ═══════


class TestParkedClaimQuestionReplay:
    async def test_parked_claim_question_hops_to_claim_adjustment(self, monkeypatch):
        agent = _mk_followup(monkeypatch, None)  # LLM must not run
        state = _followup_state(
            "yes one more thing",
            parked_followups=[
                {"query": "when will I hear about my claim?", "kind": "question", "target": ""}
            ],
        )
        hop = await agent.run(state)
        assert hop["next_node"] == "claim_adjustment_agent"
        assert hop["pending_cross_agent_request"]["kind"] == "replay"
        assert hop["pending_cross_agent_request"]["target"] == "claim_status"
        assert hop["parked_followups"] == []

        # Second leg: claim_adjustment answers from real state — never a
        # follow_up-LLM-fabricated answer.
        ca_state = {**state, **{k: v for k, v in hop.items() if k != "messages"}}
        out = await ClaimAdjustmentAgent().run(ca_state)
        message = out["messages"]["content"].lower()
        assert "in review" in message and "clm443019" in message.lower()
        assert out["next_node"] == "follow_up_agent"
        assert out["pending_cross_agent_request"] == {}

    async def test_claim_data_missing_stays_for_llm(self, monkeypatch):
        captured: dict = {}

        async def fake_extract(*a, **k):
            captured.update(k)
            return FollowUpResult(follow_up_intent=FollowUpIntent.QUESTION, answer=None)

        import agent.agents.follow_up.agent as fua

        monkeypatch.setattr(fua, "extract_follow_up_decision", fake_extract)
        monkeypatch.setattr(fua, "get_follow_up_llm", lambda: object())
        agent = FollowUpAgent()
        state = _followup_state(
            "hello",
            claim_status="",  # no adjustment data this call
            parked_followups=[
                {"query": "when will I hear about my claim?", "kind": "question", "target": ""}
            ],
        )
        out = await agent.run(state)
        assert captured["parked_followups"] == ["when will I hear about my claim?"]
        assert out["follow_up_cannot_answer_count"] == 1


# ══ Round trips: routed ZIP update returns to each claims agent ══════════════


async def _complete_zip_round_trip(monkeypatch, route: dict, state: dict) -> dict:
    """Run the owner + fast-path + orchestrator legs of a routed ZIP update."""
    import agent.orchestration.orchestration as orch
    from agent.agents.provider_search.agent import ProviderSearchAgent

    ps_state = {**state, **{k: v for k, v in route.items() if k != "messages"}}
    done = ps_state | ProviderSearchAgent()._signal_done(ps_state, "dentist", "60660")
    fp_state = {**done, "active_agent": "provider_search_agent", "zip_code": "60660"}
    assert get_fast_path_route(fp_state) == route["pending_cross_agent_request"]["return_to_agent"]
    monkeypatch.setattr(orch, "get_routing_llm", lambda: object())
    updates = await orch.Orchestrator().run(fp_state)
    assert updates["slot_update_resume"] is True
    assert updates["awaiting_slot"] == route["pending_cross_agent_request"]["return_awaiting"]
    return {**fp_state, **updates, "messages": [{"role": "user", "content": "60660"}]}


async def test_round_trip_zip_from_claim_adjustment(monkeypatch):
    agent = _mk_claim(monkeypatch, WorkerResult())
    state = _claim_state("wait — my zip code changed, i moved")
    route = await agent.run(state)
    resumed_state = await _complete_zip_round_trip(monkeypatch, route, state)
    resumed = await ClaimAdjustmentAgent().run(resumed_state)
    assert resumed["awaiting_slot"] == "reference_number"
    assert resumed["slot_update_resume"] is False
    assert "all set" in resumed["messages"]["content"].lower()


async def test_round_trip_zip_from_notification_setup(monkeypatch):
    agent = _mk_notification(monkeypatch, WorkerResult())
    state = _notification_state("wait — my address changed")
    route = await agent.run(state)
    resumed_state = await _complete_zip_round_trip(monkeypatch, route, state)
    resumed = await NotificationSetupAgent().run(resumed_state)
    assert resumed["awaiting_slot"] == "phone_confirmed"
    assert resumed["slot_update_resume"] is False
    message = resumed["messages"]["content"]
    assert "555-987-6543" in message  # the preserved phone read-back re-asked
    assert message.lower().startswith("all set")


async def test_round_trip_zip_from_records_coordination(monkeypatch):
    import agent.agents.records_coordination.agent as rca

    async def fake_extract(*a, **k):
        return WorkerResult()

    monkeypatch.setattr(rca, "extract_records_decision", fake_extract)
    monkeypatch.setattr(rca, "get_extraction_llm", lambda: object())
    agent = RecordsCoordinationAgent()
    state = {
        "messages": [
            {"role": "assistant", "content": "Would you like me to send you the upload link?"},
            {"role": "user", "content": "hold on, my zip code changed"},
        ],
        "awaiting_slot": "upload_consent",
        "call_intent": "claim_services",
        "member_status_verify": True,
        "email": "emily@example.com",
        "zip_code": "90210",
        "parked_followups": [],
    }
    route = await agent.run(state)
    assert route["next_node"] == "provider_search_agent"
    assert route["pending_cross_agent_request"]["return_awaiting"] == "upload_consent"

    resumed_state = await _complete_zip_round_trip(monkeypatch, route, state)
    resumed = await RecordsCoordinationAgent().run(resumed_state)
    assert resumed["awaiting_slot"] == "upload_consent"
    assert resumed["slot_update_resume"] is False
    assert "upload" in resumed["messages"]["content"].lower()
