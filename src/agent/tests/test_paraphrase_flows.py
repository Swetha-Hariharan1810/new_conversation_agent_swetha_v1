"""
Paraphrase end-to-end flows — every fixed scenario re-exercised with NOVEL
wording that appears in neither the extraction-prompt examples nor the other
test files.

Worst case by construction: the extraction LLM is faked to return an EMPTY
result (or a bare value) on the paraphrased turn, so the deterministic layer
(request_detection + the branch-level switch/route/park machinery) must carry
the meaning alone. If a paraphrase only worked because the prompt listed it,
these tests would fail.

Each scenario drives the REAL agents across the REAL hop mechanics
(fast-path + orchestrator resume) — multi-turn, cross-agent, end to end.
"""

import pytest

from agent.agents.benefits.agent import BenefitsAgent
from agent.agents.claim_adjustment.agent import ClaimAdjustmentAgent
from agent.agents.delivery_management.agent import DeliveryManagementAgent
from agent.agents.follow_up.agent import FollowUpAgent
from agent.agents.notification_setup.agent import NotificationSetupAgent
from agent.agents.verification.agent import VerificationAgent
from agent.core.signal import AgentStatus
from agent.llm.schema import FollowUpResult, WorkerResult
from agent.orchestration.fast_path import get_fast_path_route

# ── shared plumbing ──────────────────────────────────────────────────────────


def _fake_empty(monkeypatch, module, fn_name: str, llm_name: str):
    """Extraction returns an empty result — the regex layer must carry it."""

    async def fake_extract(*a, **k):
        return WorkerResult()

    monkeypatch.setattr(module, fn_name, fake_extract)
    monkeypatch.setattr(module, llm_name, lambda: object())


async def _zip_round_trip(monkeypatch, route: dict, state: dict, new_zip="30301") -> dict:
    """Owner + fast-path + orchestrator legs of a routed ZIP update."""
    import agent.orchestration.orchestration as orch
    from agent.agents.provider_search.agent import ProviderSearchAgent

    ps_state = {**state, **{k: v for k, v in route.items() if k != "messages"}}
    done = ps_state | ProviderSearchAgent()._signal_done(ps_state, "dentist", new_zip)
    fp_state = {**done, "active_agent": "provider_search_agent", "zip_code": new_zip}
    assert get_fast_path_route(fp_state) == route["pending_cross_agent_request"]["return_to_agent"]
    monkeypatch.setattr(orch, "get_routing_llm", lambda: object())
    updates = await orch.Orchestrator().run(fp_state)
    return {**fp_state, **updates, "messages": [{"role": "user", "content": new_zip}]}


# ══ BUG-5 paraphrases: ZIP change at the fax read-back, full round trip ══════


class TestZipChangeParaphrases:
    @pytest.mark.parametrize(
        "utterance",
        [
            "one sec — we've recently moved, so that old zip won't work",
            "hang on, my postal code is different now",
            "hold on a moment — I have a new zip code now",
            "could you correct my zip please",
        ],
    )
    async def test_route_and_resume_at_fax_readback(self, monkeypatch, utterance):
        import agent.agents.delivery_management.agent as dma

        _fake_empty(monkeypatch, dma, "extract_delivery_management_decision", "get_extraction_llm")
        state = {
            "messages": [
                {"role": "assistant", "content": "The fax on file is 555-123-4567. Is this correct?"},
                {"role": "user", "content": utterance},
            ],
            "awaiting_slot": "fax_confirmed",
            "member_status_verify": True,
            "delivery_method": "fax",
            "fax": "5551234567",
            "email": "jane.doe@example.com",
            "zip_code": "90210",
            "zip_code_used": "90210",
            "parked_followups": [],
        }
        route = await DeliveryManagementAgent().run(state)
        assert route["next_node"] == "provider_search_agent"
        assert route["pending_cross_agent_request"]["target"] == "zip_code"
        assert route["zip_code_used"] == ""  # stale list invalidated

        resumed_state = await _zip_round_trip(monkeypatch, route, state)
        resumed = await DeliveryManagementAgent().run(resumed_state)
        assert resumed["awaiting_slot"] == "fax_confirmed"  # exactly where we left off
        assert resumed["slot_update_resume"] is False
        message = resumed["messages"]["content"]
        assert "30301" in message and "555" in message  # new ZIP acked, fax re-asked


# ══ BUG-3 paraphrases: channel switch at the fax read-back ═══════════════════


class TestChannelSwitchParaphrases:
    @pytest.mark.parametrize(
        "utterance",
        [
            "you know what, just shoot it over by email instead",
            "actually, email would be better for me",
        ],
    )
    async def test_switch_pre_dispatch(self, monkeypatch, utterance):
        import agent.agents.delivery_management.agent as dma

        _fake_empty(monkeypatch, dma, "extract_delivery_management_decision", "get_extraction_llm")
        state = {
            "messages": [
                {"role": "assistant", "content": "The fax on file is 555-123-4567. Is this correct?"},
                {"role": "user", "content": utterance},
            ],
            "awaiting_slot": "fax_confirmed",
            "member_status_verify": True,
            "delivery_method": "fax",
            "fax": "5551234567",
            "email": "jane.doe@example.com",
            "parked_followups": [],
        }
        out = await DeliveryManagementAgent().run(state)
        assert out["awaiting_slot"] == "email_confirmed"
        assert out["delivery_method"] == "email"
        assert "jane dot doe at example dot com" in out["messages"]["content"]


# ══ BUG-4 paraphrases: identity update mid-verification, then continue ═══════


class TestIdentityUpdateParaphrases:
    @pytest.mark.parametrize(
        "utterance,with_answer",
        [
            # NOTE: "one more thing" is an INTERRUPTION guard keyword — the
            # paraphrases deliberately avoid it so the guard layer stays out
            # of the picture.
            ("m nine zero seven five zero three — and also, my last name is different now", True),
            ("one second — I've got to correct my last name too", False),
            ("wait, my last name is spelled wrong on the account", False),
        ],
    )
    async def test_detour_then_conversation_continues(self, monkeypatch, utterance, with_answer):
        import agent.agents.verification.agent as va
        import agent.llm.response_generator as rg

        extraction_by_turn = {"queue": []}

        async def fake_extract(*a, **k):
            return extraction_by_turn["queue"].pop(0)

        async def fake_generate(**kwargs):
            return "Got it — and what should the new value be?"

        monkeypatch.setattr(va, "extract_verification_decision", fake_extract)
        monkeypatch.setattr(va, "get_extraction_llm", lambda: object())
        monkeypatch.setattr(rg, "generate_recovery_message", fake_generate)

        # Turn 1: the paraphrased request — the LLM extracts at most the value.
        extraction_by_turn["queue"].append(
            WorkerResult(extracted={"member_id": "m nine zero seven five zero three"})
            if with_answer
            else WorkerResult()
        )
        state = {
            "messages": [
                {"role": "assistant", "content": "Could I have your member ID?"},
                {"role": "user", "content": utterance},
            ],
            "awaiting_slot": "member_id",
            "first_name": "Emily",
            "last_name": "Carter",
            "name_confirmed": True,
            "call_intent": "provider_services",
            "parked_followups": [],
            "ambiguous_counts": {},
        }
        detour = await VerificationAgent().run(state)
        assert detour["awaiting_slot"] == "last_name"
        assert detour["last_name"] == ""
        assert detour["first_name"] == "Emily"  # cascade never fires on the detour
        if with_answer:
            assert detour["member_id"] == "M907503"  # the answer was still captured
            assert detour["correction_return_to"] == "dob"
        else:
            assert detour["correction_return_to"] == "member_id"

        # Turn 2: the member gives the new last name — the pipeline collects it
        # and the conversation continues at the preserved slot.
        extraction_by_turn["queue"].append(WorkerResult(extracted={"last_name": "Carter-Smith"}))
        turn2_state = {
            **state,
            **{k: v for k, v in detour.items() if k != "messages"},
            "messages": [
                {"role": "assistant", "content": detour["messages"]["content"]},
                {"role": "user", "content": "it's carter dash smith now"},
            ],
        }
        out = await VerificationAgent().run(turn2_state)
        # New name pair → the confirmation readback re-runs before anything else.
        assert out["awaiting_slot"] == "name_confirmed"
        assert "carter-smith" in out["messages"]["content"].lower()


# ══ BUG-2 paraphrases: care-coach redo, full round trip ══════════════════════


class TestCareCoachRedoParaphrases:
    @pytest.mark.parametrize(
        "utterance",
        [
            "hmm, on second thought could you shoot that list over by email instead of faxing it?",
            "would you resend the list again?",
        ],
    )
    async def test_redo_round_trip(self, monkeypatch, utterance):
        import agent.agents.benefits.agent as ba
        import agent.agents.delivery_management.agent as dma
        import agent.orchestration.orchestration as orch

        _fake_empty(monkeypatch, ba, "extract_benefits_decision", "get_extraction_llm")
        state = {
            "messages": [
                {"role": "assistant", "content": "Would you like details about our Care Coach Guides?"},
                {"role": "user", "content": utterance},
            ],
            "awaiting_slot": "care_coach_response",
            "member_status_verify": True,
            "provider_list_sent": True,
            "benefits_offer_made": True,
            "benefits_explained": True,
            "delivery_method": "fax",
            "fax": "5551234567",
            "email": "emily@example.com",
            "provider_type": "dentist",
            "slot_attempts": {},
            "parked_followups": [],
        }
        hop = await BenefitsAgent().run(state)
        assert hop["next_node"] == "delivery_management_agent"
        assert hop["pending_cross_agent_request"]["kind"] == "redo"
        assert hop["pending_cross_agent_request"]["target"] == "delivery"

        # Delivery re-dispatches, fast-path + orchestrator hand back, and the
        # Care Coach offer is re-asked exactly once.
        async def fake_dispatch(agent, st, method, dest):
            return None

        monkeypatch.setattr(dma, "dispatch_provider_list", fake_dispatch)
        delivery_state = {**state, **{k: v for k, v in hop.items() if k != "messages"}}
        done = await DeliveryManagementAgent()._proceed_to_dispatch(
            delivery_state, "email", "emily@example.com"
        )
        assert done["last_agent_signal"]["status"] == AgentStatus.COMPLETE

        fp_state = {
            **delivery_state,
            "last_agent_signal": done["last_agent_signal"],
            "active_agent": "delivery_management_agent",
            "delivery_method": "email",
        }
        assert get_fast_path_route(fp_state) == "benefits_agent"
        monkeypatch.setattr(orch, "get_routing_llm", lambda: object())
        updates = await orch.Orchestrator().run(fp_state)
        resumed = await BenefitsAgent().run(
            {**fp_state, **updates, "messages": [{"role": "user", "content": "sure"}]}
        )
        assert resumed["awaiting_slot"] == "care_coach_response"
        assert resumed["messages"]["content"].count("?") == 1  # asked once, not twice


# ══ BUG-1 paraphrases: parked questions answered from real state ═════════════


class TestParkedQuestionParaphrases:
    async def test_alert_question_replays_provider_list(self, monkeypatch):
        import agent.agents.follow_up.agent as fua

        async def boom(*a, **k):  # pragma: no cover
            raise AssertionError("the generation path must never see an owned parked question")

        monkeypatch.setattr(fua, "extract_follow_up_decision", boom)
        monkeypatch.setattr(fua, "get_follow_up_llm", lambda: object())
        state = {
            "messages": [
                {"role": "assistant", "content": "Anything else I can help with?"},
                {"role": "user", "content": "yeah just checking on that"},
            ],
            "follow_up_turn_count": 1,
            "follow_up_cannot_answer_count": 0,
            "call_intent": "provider_services",
            "member_status_verify": True,
            "provider_list_sent": True,
            "delivery_method": "fax",
            "fax": "5551234567",
            "zip_code_used": "90210",
            "provider_type": "dentist",
            "parked_followups": [
                {
                    "query": "am I going to get an alert once the list goes out?",
                    "kind": "question",
                    "target": "",
                }
            ],
        }
        hop = await FollowUpAgent().run(state)
        assert hop["next_node"] == "delivery_management_agent"
        assert hop["pending_cross_agent_request"]["target"] == "provider_list"

        # Delivery answers from REAL state — channel and destination can never
        # be invented.
        out = await DeliveryManagementAgent().run(
            {**state, **{k: v for k, v in hop.items() if k != "messages"}}
        )
        message = out["messages"]["content"].lower()
        assert "fax" in message and "5551234567" in message and "90210" in message

    async def test_deductible_figure_question_replays_benefits(self, monkeypatch):
        import agent.agents.follow_up.agent as fua

        async def boom(*a, **k):  # pragma: no cover
            raise AssertionError("owned parked questions never reach the LLM")

        monkeypatch.setattr(fua, "extract_follow_up_decision", boom)
        monkeypatch.setattr(fua, "get_follow_up_llm", lambda: object())
        state = {
            "messages": [
                {"role": "assistant", "content": "Anything else I can help with?"},
                {"role": "user", "content": "one thing actually"},
            ],
            "follow_up_turn_count": 1,
            "follow_up_cannot_answer_count": 0,
            "call_intent": "provider_services",
            "benefits_explained": True,
            "parked_followups": [
                {"query": "how much was that deductible figure again?", "kind": "question", "target": ""}
            ],
        }
        hop = await FollowUpAgent().run(state)
        assert hop["next_node"] == "benefits_agent"
        assert hop["pending_cross_agent_request"]["target"] == "benefits"


# ══ Claims-path paraphrases ══════════════════════════════════════════════════


class TestClaimsPathParaphrases:
    async def test_moved_houses_routes_zip_from_reference_ask(self, monkeypatch):
        import agent.agents.claim_adjustment.agent as caa

        _fake_empty(monkeypatch, caa, "extract_claim_adjustment_decision", "get_extraction_llm")
        state = {
            "messages": [
                {"role": "assistant", "content": "Could I get the reference number from your letter?"},
                {"role": "user", "content": "hang on — we moved houses recently, the zip's different"},
            ],
            "awaiting_slot": "reference_number",
            "call_intent": "claim_services",
            "member_status_verify": True,
            "zip_code": "90210",
            "parked_followups": [],
        }
        route = await ClaimAdjustmentAgent().run(state)
        assert route["next_node"] == "provider_search_agent"
        assert route["pending_cross_agent_request"]["return_awaiting"] == "reference_number"

        resumed_state = await _zip_round_trip(monkeypatch, route, state)
        resumed = await ClaimAdjustmentAgent().run(resumed_state)
        assert resumed["awaiting_slot"] == "reference_number"
        assert "reference" in resumed["messages"]["content"].lower()

    async def test_change_surname_first_routes_to_verification(self, monkeypatch):
        import agent.agents.claim_adjustment.agent as caa

        _fake_empty(monkeypatch, caa, "extract_claim_adjustment_decision", "get_extraction_llm")
        state = {
            "messages": [
                {"role": "assistant", "content": "Could I get the reference number from your letter?"},
                {"role": "user", "content": "I have to change my last name first"},
            ],
            "awaiting_slot": "reference_number",
            "call_intent": "claim_services",
            "member_status_verify": True,
            "first_name": "James",
            "last_name": "Wilson",
            "name_confirmed": True,
            "parked_followups": [],
        }
        out = await ClaimAdjustmentAgent().run(state)
        assert out["next_node"] == "verification_agent"
        assert out["last_name"] == ""
        assert out["member_status_verify"] is False

    @pytest.mark.parametrize(
        "utterance",
        [
            "honestly, email works better for me at this point",
            "you know, I'd rather you email me instead of texting",
        ],
    )
    async def test_notification_switch_paraphrases(self, monkeypatch, utterance):
        import agent.agents.notification_setup.agent as nsa

        _fake_empty(monkeypatch, nsa, "extract_notification_decision", "get_extraction_llm")
        state = {
            "messages": [
                {"role": "assistant", "content": "Is 555-987-6543 the right number for SMS updates?"},
                {"role": "user", "content": utterance},
            ],
            "awaiting_slot": "phone_confirmed",
            "call_intent": "claim_services",
            "member_status_verify": True,
            "notification_channel": "sms",
            "phone_number": "5559876543",
            "email": "emily@example.com",
            "parked_followups": [],
        }
        out = await NotificationSetupAgent().run(state)
        assert out["awaiting_slot"] == "email_confirmed"
        assert out["notification_channel"] == "email"
        assert "phone" not in out["messages"]["content"].lower()

    async def test_switch_back_to_text_from_email_confirm(self, monkeypatch):
        import agent.agents.notification_setup.agent as nsa

        _fake_empty(monkeypatch, nsa, "extract_notification_decision", "get_extraction_llm")
        state = {
            "messages": [
                {"role": "assistant", "content": "The email on file is emily at example dot com — correct?"},
                {"role": "user", "content": "can you just text me instead"},
            ],
            "awaiting_slot": "email_confirmed",
            "call_intent": "claim_services",
            "member_status_verify": True,
            "notification_channel": "email",
            "phone_number": "5559876543",
            "email": "emily@example.com",
            "parked_followups": [],
        }
        out = await NotificationSetupAgent().run(state)
        assert out["awaiting_slot"] == "phone_confirmed"
        assert out["notification_channel"] == "sms"
        assert "555-987-6543" in out["messages"]["content"]

    async def test_parked_adjustment_question_replays_claim_status(self, monkeypatch):
        import agent.agents.follow_up.agent as fua

        async def boom(*a, **k):  # pragma: no cover
            raise AssertionError("owned parked questions never reach the LLM")

        monkeypatch.setattr(fua, "extract_follow_up_decision", boom)
        monkeypatch.setattr(fua, "get_follow_up_llm", lambda: object())
        state = {
            "messages": [
                {"role": "assistant", "content": "Anything else I can help with?"},
                {"role": "user", "content": "yes, quickly"},
            ],
            "follow_up_turn_count": 1,
            "follow_up_cannot_answer_count": 0,
            "call_intent": "claim_services",
            "member_status_verify": True,
            "claim_status": "in review with our adjustment team",
            "reference_number": "CLM443019",
            "last_update_date": "July 1",
            "parked_followups": [
                {
                    "query": "any idea when someone will get back to me about the adjustment?",
                    "kind": "question",
                    "target": "",
                }
            ],
        }
        hop = await FollowUpAgent().run(state)
        assert hop["next_node"] == "claim_adjustment_agent"
        assert hop["pending_cross_agent_request"]["target"] == "claim_status"

        out = await ClaimAdjustmentAgent().run({**state, **{k: v for k, v in hop.items() if k != "messages"}})
        message = out["messages"]["content"].lower()
        assert "in review" in message and "5 to 10 business days" in message

    async def test_live_benefits_replay_paraphrase_via_follow_up(self, monkeypatch):
        # Live (non-parked) replay: the wording matches the regex table, so
        # even an empty extraction routes it through the capability registry.
        import agent.agents.follow_up.agent as fua

        async def fake_extract(*a, **k):
            return FollowUpResult()  # LLM produced nothing usable

        monkeypatch.setattr(fua, "extract_follow_up_decision", fake_extract)
        monkeypatch.setattr(fua, "get_follow_up_llm", lambda: object())
        state = {
            "messages": [
                {"role": "assistant", "content": "Anything else I can help with?"},
                {"role": "user", "content": "please go over my benefits once more"},
            ],
            "follow_up_turn_count": 1,
            "follow_up_cannot_answer_count": 0,
            "call_intent": "provider_services",
            "member_status_verify": True,
            "benefits_explained": True,
            "parked_followups": [],
        }
        out = await FollowUpAgent().run(state)
        # With no LLM fields, the utterance itself must still be honored via
        # the follow_up regex backfill — a benefits replay hop.
        assert out.get("next_node") == "benefits_agent"
        assert out["pending_cross_agent_request"]["kind"] == "replay"

    async def test_live_notification_change_paraphrase_via_follow_up(self, monkeypatch):
        # Post-setup channel change phrased as an update, with an EMPTY
        # extraction: the follow_up backfill upgrades it to UPDATE_REQUEST and
        # the capability gate hops it to notification_setup as a redo.
        import agent.agents.follow_up.agent as fua

        async def fake_extract(*a, **k):
            return FollowUpResult()

        monkeypatch.setattr(fua, "extract_follow_up_decision", fake_extract)
        monkeypatch.setattr(fua, "get_follow_up_llm", lambda: object())
        state = {
            "messages": [
                {"role": "assistant", "content": "Anything else I can help with?"},
                {"role": "user", "content": "i need to change my notification preference"},
            ],
            "follow_up_turn_count": 1,
            "follow_up_cannot_answer_count": 0,
            "call_intent": "claim_services",
            "member_status_verify": True,
            "notification_channel": "sms",
            "claim_flow_complete": True,
            "parked_followups": [],
        }
        out = await FollowUpAgent().run(state)
        assert out.get("next_node") == "notification_setup_agent"
        assert out["pending_cross_agent_request"]["kind"] == "redo"
        assert out["pending_cross_agent_request"]["target"] == "notification"


# ══ Negative controls: near-miss paraphrases keep their normal paths ═════════


class TestParaphraseNegativeControls:
    async def test_plain_confirm_with_send_words_still_confirms(self, monkeypatch):
        import agent.agents.delivery_management.agent as dma

        async def fake_extract(*a, **k):
            return WorkerResult(extracted={"fax_confirmed": "yes"})

        async def fake_dispatch(agent, st, method, dest):
            return None

        monkeypatch.setattr(dma, "extract_delivery_management_decision", fake_extract)
        monkeypatch.setattr(dma, "get_extraction_llm", lambda: object())
        monkeypatch.setattr(dma, "dispatch_provider_list", fake_dispatch)
        state = {
            "messages": [
                {"role": "assistant", "content": "The fax on file is 555-123-4567. Is this correct?"},
                {"role": "user", "content": "yes the fax is correct, send it there"},
            ],
            "awaiting_slot": "fax_confirmed",
            "member_status_verify": True,
            "delivery_method": "fax",
            "fax": "5551234567",
            "zip_code": "90210",
            "provider_type": "dentist",
            "parked_followups": [],
        }
        out = await DeliveryManagementAgent().run(state)
        # Dispatch proceeded — no switch, no route, no redo.
        assert out.get("provider_list_sent") is True
        assert "pending_cross_agent_request" not in out

    async def test_third_party_moved_is_not_a_zip_update(self, monkeypatch):
        import agent.agents.delivery_management.agent as dma
        import agent.llm.response_generator as rg

        async def fake_extract(*a, **k):
            return WorkerResult()

        async def fake_generate(**kwargs):
            return "Sorry — is 555-123-4567 still the right fax?"

        monkeypatch.setattr(dma, "extract_delivery_management_decision", fake_extract)
        monkeypatch.setattr(dma, "get_extraction_llm", lambda: object())
        monkeypatch.setattr(rg, "generate_recovery_message", fake_generate)
        state = {
            "messages": [
                {"role": "assistant", "content": "The fax on file is 555-123-4567. Is this correct?"},
                {"role": "user", "content": "my sister just moved in with me"},
            ],
            "awaiting_slot": "fax_confirmed",
            "member_status_verify": True,
            "delivery_method": "fax",
            "fax": "5551234567",
            "zip_code": "90210",
            "parked_followups": [],
        }
        out = await DeliveryManagementAgent().run(state)
        assert out["awaiting_slot"] == "fax_confirmed"  # ordinary retry
        assert "pending_cross_agent_request" not in out
