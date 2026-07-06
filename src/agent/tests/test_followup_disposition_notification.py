"""
Follow-up disposition + structured parking tests (Phase 3, fixes Bug B).

Covers:
  - normalize_parked_followups: legacy plain strings coerced to structured
    dicts, junk dropped, unknown kinds coerced to "question"
  - header.md: park-wins rule and disposition examples present
  - slot_manager FOLLOWUP_PARK branch: structured parked items; an update
    request the pipeline cannot honor parks as kind="action" when the
    ownership registry names an owner (and declines when human-only)
  - follow_up agent: kind="action" parked items route via the registry —
    hand-off for owned slots, MSG_UPDATE_REQUEST_ESCALATE only for human-only
"""

from types import SimpleNamespace

import pytest

from agent.agents.follow_up.agent import FollowUpAgent
from agent.agents.follow_up.constants import MSG_UPDATE_REQUEST_ESCALATE
from agent.conversation.context import ConversationContext
from agent.core.slot_manager import SlotManagerMixin, _InternalSlotConfig
from agent.core.slot_ownership import OWNER_HUMAN, OWNER_VERIFICATION, slot_update_owner
from agent.state import normalize_parked_followups
from agent.utils import read_prompt

# ── normalize_parked_followups ───────────────────────────────────────────────


def test_legacy_strings_normalized():
    items = ["will I get a text?", "  how long does it take  ", ""]
    assert normalize_parked_followups(items) == [
        {"query": "will I get a text?", "kind": "question", "target": ""},
        {"query": "how long does it take", "kind": "question", "target": ""},
    ]


def test_structured_items_pass_through():
    items = [{"query": "update my member id", "kind": "action", "target": "member_id"}]
    assert normalize_parked_followups(items) == items


def test_mixed_legacy_and_structured():
    items = ["a question?", {"query": "update dob", "kind": "action", "target": "dob"}]
    out = normalize_parked_followups(items)
    assert out[0] == {"query": "a question?", "kind": "question", "target": ""}
    assert out[1] == {"query": "update dob", "kind": "action", "target": "dob"}


def test_junk_dropped_and_unknown_kind_coerced():
    items = [None, 42, {"kind": "action"}, {"query": "q", "kind": "banana", "target": None}]
    assert normalize_parked_followups(items) == [{"query": "q", "kind": "question", "target": ""}]


def test_none_and_empty():
    assert normalize_parked_followups(None) == []
    assert normalize_parked_followups([]) == []


# ── slot ownership registry ──────────────────────────────────────────────────


def test_identity_slots_owned_by_verification():
    for slot in ("first_name", "last_name", "member_id", "dob"):
        assert slot_update_owner(slot) == OWNER_VERIFICATION


def test_notification_slots_owned_by_claims_flow():
    assert slot_update_owner("notification_method") == "claim_services"


def test_provider_flow_slots_route_to_provider_services():
    # Phase 4: zip/fax/email are owned by the provider flow's agents, so a
    # post-flow parked action re-runs provider_services instead of escalating.
    for slot in ("zip_code", "fax", "email"):
        assert slot_update_owner(slot) == "provider_services"


def test_human_only_and_unknown_slots_stay_human():
    for slot in ("phone_number", "member_status_verify", "call_intent", "some_new_slot", ""):
        assert slot_update_owner(slot) == OWNER_HUMAN


# ── header.md prompt: park wins for later-stage questions ───────────────────


def test_header_prompt_park_guidance():
    header = read_prompt("extraction/header.md")
    assert "choose park — never decline" in header
    assert '"will I get a text/notification when it\'s sent?"' in header
    assert '"how long will delivery take?"' in header
    assert '"what\'s your favorite color?"' in header
    assert '"can you repeat my ZIP?"' in header
    # answer_now guidance: current-stage answerable questions are not parked.
    assert "Never park what the current flow can" in header
    assert "delivery is being arranged is answerable from the delivery" in header


# ── slot_manager park branch: structured items + action parking ─────────────


class _FakeAgent(SlotManagerMixin):
    AGENT_NAME = "test_agent"

    def __init__(self):
        self._slots = {}
        self._newly_confirmed = set()
        self._pending_ambiguous_resets = set()

    def ask_member(self, state, message):
        return {"response": message}


def _cfg(slot_name, prompt=""):
    return SimpleNamespace(
        slot_name=slot_name,
        prompt=prompt,
        normalizer=lambda v: str(v).strip(),
        validator=lambda v: SimpleNamespace(valid=True),
        slot_type=None,
    )


@pytest.fixture
def stub_generation(monkeypatch):
    captured = {}

    async def fake_generate(**kwargs):
        captured.update(kwargs)
        return "Got it — I'll get to that shortly."

    import agent.llm.response_generator as rg

    monkeypatch.setattr(rg, "generate_recovery_message", fake_generate)
    return captured


async def _run_followup(agent, *, disposition, followup_query, update_target="", state=None):
    ctx = ConversationContext(confirmed_slots=["first_name", "last_name", "member_id"])
    decision = SimpleNamespace(
        extracted={},
        corrections={},
        update_target=update_target,
        followup_query=followup_query,
        followup_disposition=disposition,
    )
    slot_configs = {
        "zip_code": _cfg("zip_code", prompt="Could I have your five-digit ZIP code?"),
    }
    return await agent._handle_answered_followup(
        state if state is not None else {},
        _InternalSlotConfig(
            slot_name="dob",
            prompt="What is your date of birth?",
            normalizer=str,
            validator=lambda v: SimpleNamespace(valid=True),
        ),
        [{"role": "user", "content": "March first 1990"}],
        "03/01/1990",
        ctx,
        decision=decision,
        pending_slots=["dob", "zip_code"],
        slot_configs=slot_configs,
        collected={},
    )


async def test_park_produces_structured_question_item(stub_generation):
    _, interrupt = await _run_followup(
        _FakeAgent(), disposition="park", followup_query="will I get a text when it's sent?"
    )
    assert stub_generation["guard"] == "FOLLOWUP_PARK"
    assert interrupt["parked_followups"] == [
        {"query": "will I get a text when it's sent?", "kind": "question", "target": ""}
    ]


async def test_park_normalizes_preexisting_legacy_strings(stub_generation):
    _, interrupt = await _run_followup(
        _FakeAgent(),
        disposition="park",
        followup_query="how long will delivery take?",
        state={"parked_followups": ["an old legacy question?"]},
    )
    assert interrupt["parked_followups"] == [
        {"query": "an old legacy question?", "kind": "question", "target": ""},
        {"query": "how long will delivery take?", "kind": "question", "target": ""},
    ]


async def test_unhonorable_update_with_owner_parks_as_action(stub_generation):
    # member_id is not in this pipeline's slot_configs → not updatable here,
    # but the registry owns it (verification) → park as action, not decline.
    _, interrupt = await _run_followup(
        _FakeAgent(),
        disposition="none",
        followup_query="",
        update_target="member_id",
    )
    assert stub_generation["guard"] == "FOLLOWUP_PARK"
    assert interrupt["parked_followups"] == [
        {"query": "update member id", "kind": "action", "target": "member_id"}
    ]


async def test_unhonorable_update_human_only_still_declines(stub_generation):
    # phone_number is human-only in the registry → decline, nothing parked.
    _, interrupt = await _run_followup(
        _FakeAgent(),
        disposition="none",
        followup_query="",
        update_target="phone_number",
    )
    assert stub_generation["guard"] == "FOLLOWUP_DECLINE"
    assert "parked_followups" not in interrupt


# ── follow_up agent: action-vs-question routing ─────────────────────────────


def _followup_agent():
    return FollowUpAgent()


def test_parked_action_human_only_escalates():
    agent = _followup_agent()
    result = agent._route_parked_action(
        {"call_intent": "claim_services"},
        {"query": "change my phone number", "kind": "action", "target": "phone_number"},
    )
    assert result["next_node"] == "escalation_agent"
    assert result["escalation_pre_message"] == MSG_UPDATE_REQUEST_ESCALATE
    assert result["parked_followups"] == []


def test_parked_action_identity_slot_reroutes_to_verification():
    agent = _followup_agent()
    result = agent._route_parked_action(
        {"call_intent": "claim_services"},
        {"query": "update my member id", "kind": "action", "target": "member_id"},
    )
    assert result["next_node"] == "verification_agent"
    assert result["call_intent"] == "claim_services"


def test_parked_action_flow_owned_slot_reroutes_to_owning_intent():
    agent = _followup_agent()
    result = agent._route_parked_action(
        {"call_intent": "provider_services"},
        {"query": "change my notification method", "kind": "action", "target": "notification_method"},
    )
    assert result["next_node"] == "verification_agent"
    assert result["call_intent"] == "claim_services"


def test_parked_action_identity_without_intent_escalates():
    agent = _followup_agent()
    result = agent._route_parked_action(
        {},
        {"query": "update my dob", "kind": "action", "target": "dob"},
    )
    assert result["next_node"] == "escalation_agent"
    assert result["escalation_pre_message"] == MSG_UPDATE_REQUEST_ESCALATE


async def test_run_routes_parked_action_before_llm(monkeypatch):
    # An action item present on entry must route immediately — no opener, no
    # LLM call (extract_follow_up_decision would raise if reached).
    import agent.agents.follow_up.agent as fua

    async def boom(*a, **k):  # pragma: no cover - must not be called
        raise AssertionError("LLM path must not run for parked actions")

    monkeypatch.setattr(fua, "extract_follow_up_decision", boom)
    agent = _followup_agent()
    state = {
        "messages": [],
        "call_intent": "claim_services",
        "parked_followups": [{"query": "update my member id", "kind": "action", "target": "member_id"}],
    }
    result = await agent.run(state)
    assert result["next_node"] == "verification_agent"
