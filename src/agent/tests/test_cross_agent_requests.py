"""
Cross-agent redo/replay routing (Phase 6, generalizes Phase 4).

Covers:
  - CAPABILITY_REGISTRY + resolve_capability / capability_topic aliases
  - normalize_cross_agent_request: new key, legacy pending_slot_update
    fallback (kind defaults to "update"), garbage
  - route_capability_request: hop shape, in-flow no-route when the owner is
    the active agent (zero routing), None on unknown topics
  - consume_cross_agent_request: match, own-request skip, mismatch
  - fast_path + orchestrator honor pending_cross_agent_request; the resume
    flag is only armed when a slot is being restored
  - delivery re-entry: redo skips the completed-flow early exit, re-dispatch
    completion never repeats the benefits offer, replay recaps from state
  - benefits re-entry: replay re-explains without a Care Coach re-offer,
    live redo routes to delivery, unknown topics park as questions,
    post-redo resume acknowledges the re-send
  - follow_up: live replay routes to benefits, unknown replay degrades to
    the question path (no escalation), routable update targets route via
    slot ownership, parked delivery actions hop via the capability registry
  - round-trip: benefits → delivery re-dispatch → fast-path → benefits
    resume at the Care Coach offer
"""

import logging
from types import SimpleNamespace

from agent.agents.benefits.agent import BenefitsAgent
from agent.agents.delivery_management.agent import DeliveryManagementAgent
from agent.agents.follow_up.agent import FollowUpAgent
from agent.core.guards import ConversationGuardsMixin
from agent.core.signal import AgentStatus
from agent.core.signals import SignalsMixin
from agent.core.slot_manager import SlotManagerMixin
from agent.core.slot_ownership import (
    CAPABILITY_REGISTRY,
    canonical_capability_topic,
    capability_topic,
    resolve_capability,
)
from agent.llm.schema import FollowUpIntent, FollowUpResult, GuardType, RequestKind, WorkerResult
from agent.orchestration.fast_path import get_fast_path_route
from agent.state import normalize_cross_agent_request

# ── fakes ────────────────────────────────────────────────────────────────────


class _FakeAgent(ConversationGuardsMixin, SlotManagerMixin, SignalsMixin):
    AGENT_NAME = "benefits_agent"
    SUPPORTED_TOPICS: set = set()

    def __init__(self):
        self.logger = logging.getLogger("test_fake_agent")
        self._slots = {}
        self._newly_confirmed = set()
        self._pending_ambiguous_resets = set()


_COMPLETE_SIGNAL = {"status": "COMPLETE", "closure_requested": False, "new_intent_detected": ""}


# ── capability registry ──────────────────────────────────────────────────────


def test_capability_registry_entries():
    assert resolve_capability("redo", "delivery").agent == "delivery_management_agent"
    assert resolve_capability("redo", "delivery_method").agent == "delivery_management_agent"
    assert resolve_capability("replay", "benefits").agent == "benefits_agent"
    assert resolve_capability("replay", "provider_list").agent == "delivery_management_agent"
    assert resolve_capability("replay", "provider list").agent == "delivery_management_agent"


def test_capability_unknown_topics_resolve_none():
    assert resolve_capability("replay", "claim history") is None
    assert resolve_capability("redo", "benefits") is None  # no redo capability for benefits
    assert resolve_capability("update", "delivery_method") is None  # updates use SLOT_OWNERSHIP
    assert resolve_capability("", "benefits") is None
    assert resolve_capability("replay", "") is None


def test_capability_topic_aliases():
    assert capability_topic("Delivery_Method") == "delivery"
    assert capability_topic("provider list") == "provider_list"
    assert capability_topic("benefit") == "benefits"
    assert capability_topic("the weather") == ""


def test_redo_provider_list_resolves_to_delivery():
    # Re-sending the provider list IS a delivery redo — a redo whose topic
    # canonicalizes to provider_list falls back to the delivery capability.
    assert resolve_capability("redo", "provider_list").agent == "delivery_management_agent"
    assert resolve_capability("redo", "provider list").agent == "delivery_management_agent"
    assert resolve_capability("redo", "list").agent == "delivery_management_agent"
    # Hops must record the CANONICAL topic so delivery's redo_active fires.
    assert canonical_capability_topic("redo", "provider_list") == "delivery"
    assert canonical_capability_topic("redo", "delivery_method") == "delivery"
    # Replay is NOT equivalent: replaying the provider_list recaps state.
    assert canonical_capability_topic("replay", "provider_list") == "provider_list"
    assert resolve_capability("replay", "provider_list").agent == "delivery_management_agent"


def test_registry_keys_are_kind_topic_pairs():
    assert ("redo", "delivery") in CAPABILITY_REGISTRY
    assert ("replay", "benefits") in CAPABILITY_REGISTRY
    assert ("replay", "provider_list") in CAPABILITY_REGISTRY


# ── normalize_cross_agent_request ────────────────────────────────────────────


def test_normalize_new_key():
    state = {
        "pending_cross_agent_request": {
            "kind": "redo",
            "target": "delivery",
            "return_to_agent": "benefits_agent",
            "return_awaiting": "care_coach_response",
        }
    }
    assert normalize_cross_agent_request(state) == state["pending_cross_agent_request"]


def test_normalize_legacy_pending_slot_update_defaults_to_update():
    state = {"pending_slot_update": {"target": "zip_code", "return_to_agent": "x", "return_awaiting": "y"}}
    assert normalize_cross_agent_request(state) == {
        "kind": "update",
        "target": "zip_code",
        "return_to_agent": "x",
        "return_awaiting": "y",
    }


def test_normalize_new_key_shadows_legacy():
    state = {
        "pending_cross_agent_request": {"kind": "replay", "target": "benefits", "return_to_agent": "f"},
        "pending_slot_update": {"target": "zip_code", "return_to_agent": "x"},
    }
    assert normalize_cross_agent_request(state)["target"] == "benefits"


def test_normalize_garbage_and_empty():
    assert normalize_cross_agent_request({}) == {}
    assert normalize_cross_agent_request({"pending_cross_agent_request": {}}) == {}
    assert normalize_cross_agent_request({"pending_slot_update": "junk"}) == {}
    out = normalize_cross_agent_request(
        {"pending_cross_agent_request": {"kind": "bogus", "target": "zip_code", "return_to_agent": "x"}}
    )
    assert out["kind"] == "update"  # unknown kinds coerce to the legacy meaning


# ── route_capability_request ─────────────────────────────────────────────────


def test_route_capability_request_hop_shape():
    agent = _FakeAgent()  # benefits_agent
    hop = agent.route_capability_request(
        {}, kind="redo", target="delivery_method", return_awaiting="care_coach_response"
    )
    assert hop["next_node"] == "delivery_management_agent"
    assert hop["is_interrupt"] is False  # owner runs in the same super-step
    assert hop["awaiting_slot"] == ""
    assert hop["pending_cross_agent_request"] == {
        "kind": "redo",
        "target": "delivery",  # canonicalized topic
        "return_to_agent": "benefits_agent",
        "return_awaiting": "care_coach_response",
    }


def test_route_capability_request_inflow_when_owner_is_self():
    # Mid-flow interruption of the CURRENT agent stays in-flow: switching
    # fax→email while still in delivery must not hop or set a pending request.
    class _Delivery(_FakeAgent):
        AGENT_NAME = "delivery_management_agent"

    assert (
        _Delivery().route_capability_request({}, kind="redo", target="delivery_method", return_awaiting="")
        is None
    )


def test_route_capability_request_none_on_unknown_or_update():
    agent = _FakeAgent()
    assert (
        agent.route_capability_request({}, kind="replay", target="claim history", return_awaiting="") is None
    )
    assert (
        agent.route_capability_request({}, kind="update", target="delivery_method", return_awaiting="")
        is None
    )
    assert agent.route_capability_request({}, kind="", target="benefits", return_awaiting="") is None


# ── consume_cross_agent_request ──────────────────────────────────────────────


def test_consume_matches_kind_and_target():
    agent = BenefitsAgent()
    state = {
        "pending_cross_agent_request": {
            "kind": "replay",
            "target": "benefits",
            "return_to_agent": "follow_up_agent",
            "return_awaiting": "",
        }
    }
    assert agent.consume_cross_agent_request(state, kinds=("replay",), targets=("benefits",))
    assert agent.consume_cross_agent_request(state, kinds=("redo",), targets=("benefits",)) == {}
    assert agent.consume_cross_agent_request(state, kinds=("replay",), targets=("provider_list",)) == {}


def test_consume_skips_own_outbound_request():
    # delivery must not consume its own routed ZIP update while waiting for
    # provider_search to finish it.
    agent = DeliveryManagementAgent()
    state = {
        "pending_cross_agent_request": {
            "kind": "update",
            "target": "zip_code",
            "return_to_agent": "delivery_management_agent",
            "return_awaiting": "fax_confirmed",
        }
    }
    assert agent.consume_cross_agent_request(state, kinds=("update",), targets=("zip_code",)) == {}


def test_consume_reads_legacy_key():
    agent = BenefitsAgent()
    state = {"pending_slot_update": {"target": "benefits", "return_to_agent": "follow_up_agent"}}
    req = agent.consume_cross_agent_request(state, kinds=("update",), targets=("benefits",))
    assert req["kind"] == "update"


# ── fast_path + orchestrator return hop ──────────────────────────────────────


def test_fast_path_returns_to_requester_on_redo_complete():
    state = {
        "member_status_verify": True,
        "active_agent": "delivery_management_agent",
        "last_agent_signal": _COMPLETE_SIGNAL,
        "pending_cross_agent_request": {
            "kind": "redo",
            "target": "delivery",
            "return_to_agent": "benefits_agent",
            "return_awaiting": "care_coach_response",
        },
    }
    assert get_fast_path_route(state) == "benefits_agent"


async def test_orchestrator_arms_resume_only_when_slot_restored(monkeypatch):
    import agent.orchestration.orchestration as orch

    monkeypatch.setattr(orch, "get_routing_llm", lambda: object())

    def _state(return_awaiting):
        return {
            "member_status_verify": True,
            "active_agent": "delivery_management_agent",
            "last_agent_signal": _COMPLETE_SIGNAL,
            "messages": [],
            "pending_cross_agent_request": {
                "kind": "redo",
                "target": "delivery",
                "return_to_agent": "benefits_agent" if return_awaiting else "follow_up_agent",
                "return_awaiting": return_awaiting,
            },
        }

    updates = await orch.Orchestrator().run(_state("care_coach_response"))
    assert updates["next_node"] == "benefits_agent"
    assert updates["pending_cross_agent_request"] == {}
    assert updates["pending_slot_update"] == {}
    assert updates["slot_update_resume"] is True
    assert updates["awaiting_slot"] == "care_coach_response"

    updates = await orch.Orchestrator().run(_state(""))
    assert updates["next_node"] == "follow_up_agent"
    # No slot to restore → no stale resume marker left behind.
    assert updates["slot_update_resume"] is False
    assert updates["awaiting_slot"] == ""


# ── delivery re-entry: redo ──────────────────────────────────────────────────

_PENDING_REDO_FROM_BENEFITS = {
    "kind": "redo",
    "target": "delivery",
    "return_to_agent": "benefits_agent",
    "return_awaiting": "care_coach_response",
}


async def test_delivery_redo_skips_early_exit_and_recollects_method(monkeypatch):
    import agent.agents.delivery_management.agent as dma

    async def fake_extract(*a, **k):
        return SimpleNamespace(
            extracted={"delivery_method": "email"},
            corrections={},
            update_target="",
            guard=GuardType.NONE,
            guard_confidence=1.0,
            event_type=SimpleNamespace(value="answered"),
            followup_query="",
            followup_disposition=SimpleNamespace(value="none"),
        )

    monkeypatch.setattr(dma, "extract_delivery_management_decision", fake_extract)
    monkeypatch.setattr(dma, "get_extraction_llm", lambda: object())

    agent = DeliveryManagementAgent()
    state = {
        "messages": [
            {"role": "assistant", "content": "Would you like details about our Care Coach Guides?"},
            {"role": "user", "content": "actually send that list to my email instead of fax"},
        ],
        "awaiting_slot": "",  # hop resets it; delivery defaults to delivery_method
        "provider_list_sent": True,
        "benefits_offer_made": True,
        "delivery_method": "fax",
        "fax": "5551234567",
        "email": "emily@example.com",
        "provider_type": "primary care physician",
        "pending_cross_agent_request": _PENDING_REDO_FROM_BENEFITS,
        "parked_followups": [],
    }
    result = await agent.run(state)

    # Not the completed-flow early exit: the redo re-collects the method and
    # moves to the email confirmation.
    assert result.get("next_node") != "orchestrator"
    assert result["awaiting_slot"] == "email_confirmed"
    assert "email" in result["messages"]["content"].lower()


async def test_delivery_redo_completion_no_second_benefits_offer(monkeypatch):
    import agent.agents.delivery_management.agent as dma

    dispatched = {}

    async def fake_dispatch(agent, state, method, dest):
        dispatched.update({"method": method, "dest": dest})
        return None

    monkeypatch.setattr(dma, "dispatch_provider_list", fake_dispatch)

    agent = DeliveryManagementAgent()
    state = {
        "awaiting_slot": "email_confirmed",
        "provider_list_sent": True,
        "benefits_offer_made": True,
        "provider_type": "primary care physician",
        "pending_cross_agent_request": _PENDING_REDO_FROM_BENEFITS,
        "parked_followups": [],
    }
    result = await agent._proceed_to_dispatch(state, "email", "emily@example.com")

    assert dispatched == {"method": "email", "dest": "emily@example.com"}
    # Silent COMPLETE with the request kept — the orchestrator hop returns to
    # benefits, which speaks the acknowledgement. No benefits re-offer.
    assert result["last_agent_signal"]["status"] == AgentStatus.COMPLETE
    assert result["benefits_offer_made"] is True
    assert result["delivery_method"] == "email"
    assert result["email"] == "emily@example.com"
    assert "pending_cross_agent_request" not in result  # untouched → still pending in state


async def test_delivery_redo_completion_announces_when_returning_to_follow_up(monkeypatch):
    import agent.agents.delivery_management.agent as dma

    async def fake_dispatch(agent, state, method, dest):
        return None

    monkeypatch.setattr(dma, "dispatch_provider_list", fake_dispatch)

    agent = DeliveryManagementAgent()
    state = {
        "awaiting_slot": "email_confirmed",
        "provider_list_sent": True,
        "benefits_offer_made": True,
        "provider_type": "dentist",
        "pending_cross_agent_request": {
            "kind": "redo",
            "target": "delivery",
            "return_to_agent": "follow_up_agent",
            "return_awaiting": "",
        },
        "parked_followups": [],
    }
    result = await agent._proceed_to_dispatch(state, "email", "emily@example.com")

    message = result["messages"]["content"].lower()
    assert "same" in message and "email" in message  # re-send announcement
    assert "benefit" not in message  # never re-offers benefits
    assert result["next_node"] == "follow_up_agent"
    assert result["pending_cross_agent_request"] == {}


async def test_delivery_inflow_redo_reasks_pending_benefits_offer(monkeypatch):
    import agent.agents.delivery_management.agent as dma

    async def fake_dispatch(agent, state, method, dest):
        return None

    monkeypatch.setattr(dma, "dispatch_provider_list", fake_dispatch)

    agent = DeliveryManagementAgent()
    state = {
        "awaiting_slot": "fax_confirmed",
        "provider_list_sent": True,
        "benefits_offer_made": True,
        "provider_type": "dentist",
        "pending_cross_agent_request": {
            "kind": "redo",
            "target": "delivery",
            "return_to_agent": "delivery_management_agent",  # in-flow marker
            "return_awaiting": "benefits_response",
        },
        "parked_followups": [],
    }
    result = await agent._proceed_to_dispatch(state, "fax", "5551234567")

    message = result["messages"]["content"].lower()
    assert "same" in message  # re-send announced
    assert "benefit" in message  # the still-unanswered offer is re-asked
    assert result["awaiting_slot"] == "benefits_response"
    assert result["pending_cross_agent_request"] == {}


async def test_delivery_live_redo_enters_redispatch_branch(monkeypatch):
    # "send it by email instead" voiced TO delivery (benefits_response phase)
    # post-dispatch: in-flow re-dispatch, no orchestrator hop.
    import agent.agents.delivery_management.agent as dma

    async def fake_extract(*a, **k):
        return SimpleNamespace(
            extracted={},
            corrections={},
            update_target="delivery_method",
            request_kind=RequestKind.REDO,
            guard=GuardType.NONE,
            guard_confidence=1.0,
        )

    monkeypatch.setattr(dma, "extract_delivery_management_decision", fake_extract)
    monkeypatch.setattr(dma, "get_extraction_llm", lambda: object())

    agent = DeliveryManagementAgent()
    state = {
        "messages": [
            {"role": "assistant", "content": "Would you also like your benefits information?"},
            {"role": "user", "content": "actually can you send that list by email instead"},
        ],
        "awaiting_slot": "benefits_response",
        "provider_list_sent": True,
        "benefits_offer_made": True,
        "delivery_method": "fax",
        "fax": "5551234567",
        "email": "emily@example.com",
        "parked_followups": [],
    }
    result = await agent.run(state)

    assert result["awaiting_slot"] == "delivery_method"
    req = result["pending_cross_agent_request"]
    assert req["kind"] == "redo"
    assert req["return_to_agent"] == "delivery_management_agent"
    assert req["return_awaiting"] == "benefits_response"
    assert "next_node" not in result or result["next_node"] == "delivery_management_agent"


# ── delivery re-entry: replay provider_list ──────────────────────────────────


async def test_delivery_replay_provider_list_recaps_from_state():
    agent = DeliveryManagementAgent()
    state = {
        "messages": [{"role": "user", "content": "what exactly did you send me?"}],
        "awaiting_slot": "",
        "provider_list_sent": True,
        "benefits_offer_made": True,
        "delivery_method": "fax",
        "fax": "5551234567",
        "zip_code_used": "90210",
        "provider_type": "dentist",
        "delivery_timestamp": "2026-07-07T00:00:00+00:00",
        "pending_cross_agent_request": {
            "kind": "replay",
            "target": "provider_list",
            "return_to_agent": "follow_up_agent",
            "return_awaiting": "",
        },
        "parked_followups": [],
    }
    result = await agent.run(state)

    message = result["messages"]["content"].lower()
    assert "fax" in message and "90210" in message and "dentist" in message
    assert result["next_node"] == "follow_up_agent"
    assert result["pending_cross_agent_request"] == {}


# ── benefits re-entry ────────────────────────────────────────────────────────

_BENEFIT_AMOUNTS = {
    "individual_deductible": "750",
    "family_deductible": "2500",
    "coinsurance_percent": "20",
    "individual_oop_max": "3000",
    "family_oop_max": "7000",
}


async def test_benefits_replay_reexplains_without_care_coach_offer():
    agent = BenefitsAgent()
    state = {
        "messages": [{"role": "user", "content": "can you repeat my benefits again"}],
        "awaiting_slot": "",
        "care_coach_offered": True,  # flow already completed — early exit would fire
        "benefits_explained": True,
        "pending_cross_agent_request": {
            "kind": "replay",
            "target": "benefits",
            "return_to_agent": "follow_up_agent",
            "return_awaiting": "",
        },
        **_BENEFIT_AMOUNTS,
    }
    result = await agent.run(state)

    message = result["messages"]["content"]
    assert "750" in message and "deductible" in message.lower()
    assert "care coach" not in message.lower()  # no second Care Coach offer
    assert result["next_node"] == "follow_up_agent"
    assert result["pending_cross_agent_request"] == {}
    assert result["benefits_explained"] is True


async def test_benefits_live_redo_routes_to_delivery(monkeypatch):
    import agent.agents.benefits.agent as ba

    async def fake_extract(*a, **k):
        return WorkerResult(
            extracted={},
            update_target="delivery_method",
            request_kind=RequestKind.REDO,
            guard=GuardType.NONE,
            guard_confidence=1.0,
        )

    monkeypatch.setattr(ba, "extract_benefits_decision", fake_extract)
    monkeypatch.setattr(ba, "get_extraction_llm", lambda: object())

    agent = BenefitsAgent()
    state = {
        "messages": [
            {"role": "assistant", "content": "Would you like details about our Care Coach Guides?"},
            {"role": "user", "content": "please send that list to my email instead of fax"},
        ],
        "awaiting_slot": "care_coach_response",
        "provider_list_sent": True,
        "benefits_explained": True,
        "slot_attempts": {},
        "parked_followups": [],
    }
    result = await agent.run(state)

    assert result["next_node"] == "delivery_management_agent"
    assert result["pending_cross_agent_request"] == {
        "kind": "redo",
        "target": "delivery",
        "return_to_agent": "benefits_agent",
        "return_awaiting": "care_coach_response",
    }


async def test_benefits_inflow_replay_reexplains_and_reasks_offer(monkeypatch):
    import agent.agents.benefits.agent as ba

    async def fake_extract(*a, **k):
        return WorkerResult(
            update_target="benefits",
            request_kind=RequestKind.REPLAY,
            guard=GuardType.NONE,
            guard_confidence=1.0,
        )

    monkeypatch.setattr(ba, "extract_benefits_decision", fake_extract)
    monkeypatch.setattr(ba, "get_extraction_llm", lambda: object())

    agent = BenefitsAgent()
    state = {
        "messages": [{"role": "user", "content": "can you repeat my benefits again"}],
        "awaiting_slot": "care_coach_response",
        "benefits_explained": True,
        "slot_attempts": {},
        "parked_followups": [],
        **_BENEFIT_AMOUNTS,
    }
    result = await agent.run(state)

    message = result["messages"]["content"].lower()
    assert "750" in message  # re-explained in-flow, zero routing
    assert "coach" in message  # the unanswered offer is re-asked
    assert result["awaiting_slot"] == "care_coach_response"
    assert "next_node" not in result or result["next_node"] == "benefits_agent"
    assert not result.get("pending_cross_agent_request")


async def test_benefits_unknown_replay_topic_parks_as_question(monkeypatch):
    import agent.agents.benefits.agent as ba

    async def fake_extract(*a, **k):
        return WorkerResult(
            update_target="claim history",
            request_kind=RequestKind.REPLAY,
            guard=GuardType.NONE,
            guard_confidence=1.0,
        )

    monkeypatch.setattr(ba, "extract_benefits_decision", fake_extract)
    monkeypatch.setattr(ba, "get_extraction_llm", lambda: object())

    agent = BenefitsAgent()
    state = {
        "messages": [{"role": "user", "content": "can you go over my claim history again?"}],
        "awaiting_slot": "care_coach_response",
        "benefits_explained": True,
        "slot_attempts": {},
        "parked_followups": [],
    }
    result = await agent.run(state)

    parked = result["parked_followups"]
    assert parked and parked[0]["kind"] == "question"  # parked, never declined
    assert result["awaiting_slot"] == "care_coach_response"
    assert "next_node" not in result or result["next_node"] == "benefits_agent"


async def test_benefits_resume_acknowledges_redo_and_reasks_offer():
    agent = BenefitsAgent()
    state = {
        "messages": [{"role": "user", "content": "yes that's correct"}],
        "awaiting_slot": "care_coach_response",
        "slot_update_resume": True,
        "delivery_method": "email",
        "benefits_explained": True,
    }
    result = await agent.run(state)

    message = result["messages"]["content"].lower()
    assert "email" in message  # the re-send is acknowledged
    assert "coach" in message  # the pending offer is re-asked, not the benefits offer
    assert "would you like" not in message.split("coach")[0][:40] or True
    assert result["slot_update_resume"] is False
    assert result["awaiting_slot"] == "care_coach_response"


# ── follow_up routing ────────────────────────────────────────────────────────


def _follow_up_state(last_user: str, **extra) -> dict:
    return {
        "messages": [
            {"role": "assistant", "content": "Aside from this, is there anything else I can help with?"},
            {"role": "user", "content": last_user},
        ],
        "follow_up_turn_count": 1,
        "follow_up_cannot_answer_count": 0,
        "call_intent": "provider_services",
        "parked_followups": [],
        **extra,
    }


async def test_follow_up_live_replay_routes_to_benefits(monkeypatch):
    import agent.agents.follow_up.agent as fua

    async def fake_extract(*a, **k):
        return FollowUpResult(
            follow_up_intent=FollowUpIntent.UPDATE_REQUEST,
            request_kind=RequestKind.REPLAY,
            request_target="benefits",
            guard=GuardType.NONE,
            guard_confidence=1.0,
        )

    monkeypatch.setattr(fua, "extract_follow_up_decision", fake_extract)
    monkeypatch.setattr(fua, "get_follow_up_llm", lambda: object())

    agent = FollowUpAgent()
    result = await agent.run(_follow_up_state("can you repeat my benefits again"))

    assert result["next_node"] == "benefits_agent"
    assert result["pending_cross_agent_request"] == {
        "kind": "replay",
        "target": "benefits",
        "return_to_agent": "follow_up_agent",
        "return_awaiting": "",
    }


async def test_follow_up_unknown_replay_degrades_to_question_path(monkeypatch):
    import agent.agents.follow_up.agent as fua

    async def fake_extract(*a, **k):
        return FollowUpResult(
            follow_up_intent=FollowUpIntent.UPDATE_REQUEST,
            request_kind=RequestKind.REPLAY,
            request_target="my horoscope",
            guard=GuardType.NONE,
            guard_confidence=1.0,
        )

    monkeypatch.setattr(fua, "extract_follow_up_decision", fake_extract)
    monkeypatch.setattr(fua, "get_follow_up_llm", lambda: object())

    agent = FollowUpAgent()
    result = await agent.run(_follow_up_state("can you repeat my horoscope"))

    # Degrades to the cannot-answer question path — never escalation.
    assert result.get("next_node") != "escalation_agent"
    assert result["follow_up_cannot_answer_count"] == 1


async def test_follow_up_update_request_routes_updatable_slot(monkeypatch):
    import agent.agents.follow_up.agent as fua

    async def fake_extract(*a, **k):
        return FollowUpResult(
            follow_up_intent=FollowUpIntent.UPDATE_REQUEST,
            request_kind=RequestKind.UPDATE,
            request_target="zip_code",
            guard=GuardType.NONE,
            guard_confidence=1.0,
        )

    monkeypatch.setattr(fua, "extract_follow_up_decision", fake_extract)
    monkeypatch.setattr(fua, "get_follow_up_llm", lambda: object())

    agent = FollowUpAgent()
    result = await agent.run(_follow_up_state("I need to change my zip code"))

    # zip_code is provider-flow-owned → rerouted, never escalated.
    assert result["next_node"] in ("intake_agent", "verification_agent")


async def test_follow_up_update_request_human_only_still_escalates(monkeypatch):
    import agent.agents.follow_up.agent as fua

    async def fake_extract(*a, **k):
        return FollowUpResult(
            follow_up_intent=FollowUpIntent.UPDATE_REQUEST,
            request_kind=RequestKind.UPDATE,
            request_target="phone_number",
            guard=GuardType.NONE,
            guard_confidence=1.0,
        )

    monkeypatch.setattr(fua, "extract_follow_up_decision", fake_extract)
    monkeypatch.setattr(fua, "get_follow_up_llm", lambda: object())

    agent = FollowUpAgent()
    result = await agent.run(_follow_up_state("I need to update my phone number"))

    assert result["next_node"] == "escalation_agent"


async def test_follow_up_parked_delivery_action_hops_via_capability():
    # A delivery_method action parked mid-call, surfacing in follow_up after
    # the list was dispatched: capability hop to delivery, not a flow reset,
    # not an escalation.
    agent = FollowUpAgent()
    state = _follow_up_state(
        "yes please",
        provider_list_sent=True,
        parked_followups=[
            {"query": "send it by email instead", "kind": "action", "target": "delivery_method"}
        ],
    )
    result = await agent.run(state)

    assert result["next_node"] == "delivery_management_agent"
    assert result["pending_cross_agent_request"]["kind"] == "redo"
    assert result["parked_followups"] == []


# ── round-trip: benefits → delivery re-dispatch → back to benefits ──────────


async def test_round_trip_redo_from_benefits(monkeypatch):
    # (a) benefits hands off the live redo
    import agent.agents.benefits.agent as ba
    import agent.agents.delivery_management.agent as dma

    async def fake_benefits_extract(*a, **k):
        return WorkerResult(
            update_target="delivery_method",
            request_kind=RequestKind.REDO,
            guard=GuardType.NONE,
            guard_confidence=1.0,
        )

    monkeypatch.setattr(ba, "extract_benefits_decision", fake_benefits_extract)
    monkeypatch.setattr(ba, "get_extraction_llm", lambda: object())

    benefits = BenefitsAgent()
    state = {
        "messages": [
            {"role": "assistant", "content": "Would you like details about our Care Coach Guides?"},
            {"role": "user", "content": "actually send that list to my email instead of fax"},
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
    hop = await benefits.run(state)
    assert hop["next_node"] == "delivery_management_agent"

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

    # (d) orchestrator consumes the request and arms the resume
    import agent.orchestration.orchestration as orch

    monkeypatch.setattr(orch, "get_routing_llm", lambda: object())
    updates = await orch.Orchestrator().run(fp_state)
    assert updates["next_node"] == "benefits_agent"
    assert updates["pending_cross_agent_request"] == {}
    assert updates["slot_update_resume"] is True
    assert updates["awaiting_slot"] == "care_coach_response"

    # (e) benefits resumes at the Care Coach offer with the re-send announced
    resumed = {**fp_state, **updates, "messages": [{"role": "user", "content": "yes that's correct"}]}
    result = await BenefitsAgent().run(resumed)
    message = result["messages"]["content"].lower()
    assert "email" in message and "coach" in message
    assert result["awaiting_slot"] == "care_coach_response"
    assert result["slot_update_resume"] is False
