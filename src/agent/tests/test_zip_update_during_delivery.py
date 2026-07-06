"""
ZIP-update routing during delivery (Phase 4, fixes Bug C).

Covers:
  - SLOT_OWNERSHIP registry + resolve_update_target three-way resolution
  - _route_slot_update: pending_slot_update signal, owner hand-off,
    invalidation of provider_list_sent / zip_code_used, honest "now" message
  - delivery_management routes a mid-delivery ZIP update to provider_search
  - dispatch precondition: never dispatch while a zip-invalidating update is
    pending/parked
  - ignored-request guard: second identical declined request escalates
  - meta-questions about promised items answered via FOLLOWUP_ANSWER
  - round-trip: route → provider_search completes → fast-path returns to
    delivery_management at the preserved awaiting slot
"""

import logging
from types import SimpleNamespace

from agent.agents.delivery_management.agent import DeliveryManagementAgent
from agent.agents.provider_search.agent import ProviderSearchAgent
from agent.conversation.context import ConversationContext
from agent.core.guards import ConversationGuardsMixin
from agent.core.signals import SignalsMixin
from agent.core.slot_manager import SlotManagerMixin, _InternalSlotConfig
from agent.core.slot_ownership import get_ownership, invalidated_state_updates
from agent.orchestration.fast_path import get_fast_path_route
from agent.responses.static import MSG_REPEATED_REQUEST_ESCALATE

# ── fakes ────────────────────────────────────────────────────────────────────


class _FakeAgent(ConversationGuardsMixin, SlotManagerMixin, SignalsMixin):
    AGENT_NAME = "delivery_management_agent"
    SUPPORTED_TOPICS: set = set()

    def __init__(self):
        self.logger = logging.getLogger("test_fake_agent")
        self._slots = {}
        self._newly_confirmed = set()
        self._pending_ambiguous_resets = set()


def _ctx(confirmed=()):
    return ConversationContext(confirmed_slots=list(confirmed))


# ── registry + resolve_update_target ────────────────────────────────────────


def test_registry_modes():
    assert get_ownership("zip_code").updatable == "route_to_owner"
    assert get_ownership("zip_code").agent == "provider_search_agent"
    assert get_ownership("fax").updatable == "in_flow"
    assert get_ownership("email").agent == "delivery_management_agent"
    assert get_ownership("member_id").updatable == "in_flow"
    assert get_ownership("phone_number").updatable == "human_only"
    assert get_ownership("nonexistent_slot") is None


def test_zip_invalidates_provider_list():
    assert invalidated_state_updates("zip_code") == {"provider_list_sent": False, "zip_code_used": ""}
    assert invalidated_state_updates("fax") == {}


def test_resolve_allow_when_pipeline_collects_it():
    agent = _FakeAgent()
    out = agent.resolve_update_target("fax", _ctx(), {"fax": "5551234567"}, {"fax": None})
    assert out == "allow"


def test_resolve_route_for_owner_elsewhere():
    agent = _FakeAgent()
    out = agent.resolve_update_target("zip_code", _ctx(), {}, {"fax": None, "email": None})
    assert out == "route"


def test_resolve_decline_human_only_and_unknown():
    agent = _FakeAgent()
    assert agent.resolve_update_target("phone_number", _ctx(), {"phone_number": "x"}, None) == "decline"
    assert agent.resolve_update_target("made_up_slot", _ctx(), {}, {"fax": None}) == "decline"
    assert agent.resolve_update_target("", _ctx(), {}, None) == "decline"


def test_resolve_no_route_when_owner_is_self():
    # provider_search asking about zip_code with no value yet: own agent → not
    # a route; falls to decline (its pipeline path handles collection).
    class _PS(_FakeAgent):
        AGENT_NAME = "provider_search_agent"

    assert _PS().resolve_update_target("zip_code", _ctx(), {}, {"provider_type": None}) == "decline"


# ── _route_slot_update signal ────────────────────────────────────────────────


def test_route_slot_update_signal():
    agent = _FakeAgent()
    state = {"provider_list_sent": False, "zip_code_used": "90210", "parked_followups": []}
    route = agent._route_slot_update(state, "zip_code", _ctx(), return_awaiting="fax_confirmed")

    assert route["next_node"] == "provider_search_agent"
    assert route["awaiting_slot"] == "zip_code"
    assert route["pending_slot_update"] == {
        "target": "zip_code",
        "return_to_agent": "delivery_management_agent",
        "return_awaiting": "fax_confirmed",
    }
    # Invalidations: the stale list and the ZIP it was built from are cleared.
    assert route["provider_list_sent"] is False
    assert route["zip_code_used"] == ""

    message = route["messages"]["content"].lower()
    assert "zip code" in message
    assert "later" not in message  # routable targets are honored NOW


# ── delivery_management routes a mid-delivery ZIP update ────────────────────


async def test_delivery_routes_zip_update(monkeypatch):
    import agent.agents.delivery_management.agent as dma

    async def fake_extract(*a, **k):
        return SimpleNamespace(
            extracted={},
            corrections={},
            update_target="zip_code",
            guard="NONE",
            guard_confidence=1.0,
        )

    monkeypatch.setattr(dma, "extract_delivery_management_decision", fake_extract)
    monkeypatch.setattr(dma, "get_extraction_llm", lambda: object())

    agent = DeliveryManagementAgent()
    state = {
        "messages": [
            {"role": "assistant", "content": "Is 555-123-4567 still the best fax for you?"},
            {"role": "user", "content": "wait — my ZIP code changed, I moved"},
        ],
        "awaiting_slot": "fax_confirmed",
        "delivery_method": "fax",
        "fax": "5551234567",
        "zip_code": "90210",
        "zip_code_used": "90210",
        "parked_followups": [],
    }
    result = await agent.run(state)

    assert result["next_node"] == "provider_search_agent"
    assert result["pending_slot_update"]["return_to_agent"] == "delivery_management_agent"
    assert result["pending_slot_update"]["return_awaiting"] == "fax_confirmed"
    assert result["zip_code_used"] == ""


# ── dispatch precondition ────────────────────────────────────────────────────


async def test_dispatch_blocked_by_parked_zip_action(monkeypatch):
    import agent.agents.delivery_management.agent as dma

    async def boom(*a, **k):  # pragma: no cover - must not be called
        raise AssertionError("dispatch must not run while a ZIP update is pending")

    monkeypatch.setattr(dma, "dispatch_provider_list", boom)

    agent = DeliveryManagementAgent()
    state = {
        "awaiting_slot": "fax_confirmed",
        "delivery_method": "fax",
        "zip_code": "90210",
        "parked_followups": [{"query": "update zip code", "kind": "action", "target": "zip_code"}],
    }
    result = await agent._proceed_to_dispatch(state, "fax", "5551234567")

    assert result["next_node"] == "provider_search_agent"
    assert result["pending_slot_update"]["target"] == "zip_code"
    # The routed update consumed the parked action item.
    assert result["parked_followups"] == []


def test_blocking_detector_ignores_non_invalidating_actions():
    agent = DeliveryManagementAgent()
    state = {"parked_followups": [{"query": "update member id", "kind": "action", "target": "member_id"}]}
    assert agent._blocking_list_invalidator(state) == ""
    state = {"pending_slot_update": {"target": "zip_code", "return_to_agent": "x", "return_awaiting": "y"}}
    assert agent._blocking_list_invalidator(state) == "zip_code"


# ── ignored-request guard ────────────────────────────────────────────────────


def test_ignored_request_guard_escalates_on_second_repeat():
    agent = _FakeAgent()
    state = {}
    assert agent._ignored_request_guard(state, "phone_number") is None  # first: decline + re-ask
    escalation = agent._ignored_request_guard(state, "phone_number")  # second: escalate honestly
    assert escalation is not None
    assert escalation["next_node"] == "escalation_agent"
    assert escalation["escalation_pre_message"] in MSG_REPEATED_REQUEST_ESCALATE


async def test_offtopic_agent_second_identical_request_escalates(monkeypatch):
    import agent.llm.response_generator as rg

    async def fake_generate(**kwargs):
        return "Let's stay focused — could I get your fax confirmation?"

    monkeypatch.setattr(rg, "generate_recovery_message", fake_generate)

    agent = _FakeAgent()
    result = SimpleNamespace(guard="OFFTOPIC_AGENT", guard_confidence=0.9, extracted={})
    state = {"awaiting_slot": "fax_confirmed", "messages": [], "slot_attempts": {}}

    first = await agent.run_conversation_guards(state, user_text="cancel my gym membership", result=result)
    assert first["is_interrupt"] is True  # deflected once, re-asked

    second = await agent.run_conversation_guards(state, user_text="cancel my gym membership", result=result)
    assert second["next_node"] == "escalation_agent"
    assert second["escalation_pre_message"] in MSG_REPEATED_REQUEST_ESCALATE


# ── meta-questions about promised items ──────────────────────────────────────


def test_match_promised_item_pending_update():
    agent = _FakeAgent()
    state = {"pending_slot_update": {"target": "zip_code", "return_to_agent": "x", "return_awaiting": "y"}}
    assert "zip code" in agent._match_promised_item(state, "when will you update my zip code?")
    assert agent._match_promised_item(state, "what is my deductible?") == ""


def test_match_promised_item_parked_action():
    agent = _FakeAgent()
    state = {"parked_followups": [{"query": "update zip code", "kind": "action", "target": "zip_code"}]}
    assert "zip code update" in agent._match_promised_item(state, "when will you change my zip code?")


async def test_meta_question_forces_followup_answer(monkeypatch):
    captured = {}

    async def fake_generate(**kwargs):
        captured.update(kwargs)
        return "Your ZIP update is queued — I'll take care of it right after this."

    import agent.llm.response_generator as rg

    monkeypatch.setattr(rg, "generate_recovery_message", fake_generate)

    agent = _FakeAgent()
    ctx = _ctx(confirmed=["first_name"])
    decision = SimpleNamespace(
        extracted={},
        corrections={},
        update_target="",
        followup_query="timing of zip code update",
        followup_disposition="decline",
    )
    state = {"parked_followups": [{"query": "update zip code", "kind": "action", "target": "zip_code"}]}
    _, interrupt = await agent._handle_answered_followup(
        state,
        _InternalSlotConfig(
            slot_name="dob",
            prompt="dob?",
            normalizer=str,
            validator=lambda v: SimpleNamespace(valid=True),
        ),
        [{"role": "user", "content": "March 1 1990 — when will you update my zip code?"}],
        "03/01/1990",
        ctx,
        decision=decision,
        pending_slots=["dob"],
        slot_configs={},
        collected={},
    )
    assert captured["guard"] == "FOLLOWUP_ANSWER"
    assert "promised next step" in captured["confirmed_slots"]


# ── round-trip: delivery → provider_search → back to delivery ───────────────


async def test_round_trip_zip_update(monkeypatch):
    # (a) delivery hands off
    delivery = _FakeAgent()
    state = {
        "provider_list_sent": False,
        "zip_code": "90210",
        "zip_code_used": "90210",
        "member_status_verify": True,
    }
    route = delivery._route_slot_update(state, "zip_code", _ctx(), return_awaiting="fax_confirmed")
    assert route["next_node"] == "provider_search_agent"

    # (b) provider_search finishes the ZIP flow with the pending marker set —
    # COMPLETE signal (no delivery bridge ask), refreshed zip_code_used.
    ps = ProviderSearchAgent()
    ps_state = {**state, **{k: v for k, v in route.items() if k != "messages"}}
    done = ps._signal_done(ps_state, "dentist", "60660")
    assert done["is_interrupt"] is False
    assert done["next_node"] == "orchestrator"
    assert done["zip_code_used"] == "60660"

    # (c) fast-path returns control to delivery_management
    fp_state = {
        **ps_state,
        "last_agent_signal": done["last_agent_signal"],
        "active_agent": "provider_search_agent",
        "zip_code_used": "60660",
        "zip_code": "60660",
    }
    assert get_fast_path_route(fp_state) == "delivery_management_agent"

    # (d) orchestrator consumes pending_slot_update and arms the resume
    import agent.orchestration.orchestration as orch

    monkeypatch.setattr(orch, "get_routing_llm", lambda: object())
    updates = await orch.Orchestrator().run(fp_state)
    assert updates["next_node"] == "delivery_management_agent"
    assert updates["pending_slot_update"] == {}
    assert updates["slot_update_resume"] is True
    assert updates["awaiting_slot"] == "fax_confirmed"

    # (e) delivery resumes at fax_confirmed with the new ZIP acknowledged
    resumed_state = {
        **fp_state,
        **updates,
        "delivery_method": "fax",
        "fax": "5551234567",
        "zip_code_updated": True,
        "messages": [{"role": "user", "content": "60660"}],
    }
    result = await DeliveryManagementAgent().run(resumed_state)
    assert result["awaiting_slot"] == "fax_confirmed"
    assert result["slot_update_resume"] is False
    message = result["messages"]["content"]
    assert "60660" in message
    assert "555" in message  # fax readback re-asked


# ── verification-time corrections for provider-flow slots park, not drop ────


async def test_verification_time_zip_correction_routes_not_ghost_acks(monkeypatch):
    # During identity collection (verification pipeline: no zip tools), a
    # corrections{zip_code} turn must route to the owner — never acknowledge
    # an update that was not applied.
    captured = {}

    async def fake_generate(**kwargs):
        captured.update(kwargs)
        return "Got it."

    import agent.llm.response_generator as rg

    monkeypatch.setattr(rg, "generate_recovery_message", fake_generate)

    class _Verif(_FakeAgent):
        AGENT_NAME = "verification_agent"

    agent = _Verif()
    decision = SimpleNamespace(
        event_type=SimpleNamespace(value="corrected"),
        corrections={"zip_code": "60660"},
        update_target="",
        extracted={},
    )
    # zip_code IS populated in state (SF lookup) — the identity pipeline's
    # configs still don't collect it, so this must never resolve to "allow".
    state = {"awaiting_slot": "dob", "messages": [], "parked_followups": [], "zip_code": "90210"}
    value, interrupt = await agent._collect_slot(
        state,
        _InternalSlotConfig(
            slot_name="dob",
            prompt="What is your date of birth?",
            normalizer=str.strip,
            validator=lambda v: SimpleNamespace(valid=bool(v)),
        ),
        [{"role": "user", "content": "actually my zip is 60660"}],
        "",
        decision=decision,
        slot_configs={},
    )
    assert value is None
    # zip_code is route_to_owner → immediate hand-off, no ghost "I've updated".
    assert interrupt["next_node"] == "provider_search_agent"
    assert interrupt["pending_slot_update"]["target"] == "zip_code"
    assert interrupt["pending_slot_update"]["return_awaiting"] == "dob"
