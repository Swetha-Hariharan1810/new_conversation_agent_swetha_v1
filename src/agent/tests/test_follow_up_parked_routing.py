"""
follow_up: parked questions route to the owning agent; closure ordering
(Phase 5 — BUG-1).

BUG-1: a parked notification/delivery question surfaced in follow_up was
answered by the generation LLM, which hallucinated the channel/address it
claimed something was sent to. Parked questions that map to a registered
replay capability now hop to the owning agent BEFORE any LLM answer attempt
— the owner answers from real state (_replay_provider_list) and can never
invent a destination. A member's explicit DONE outranks stale parked items:
close immediately, drop the list loudly, never answer after closure.
"""

import logging

import pytest

from agent.agents.delivery_management.agent import DeliveryManagementAgent
from agent.agents.follow_up.agent import FollowUpAgent
from agent.core.signal import AgentStatus
from agent.llm.schema import FollowUpIntent, FollowUpResult

# ── harness ──────────────────────────────────────────────────────────────────


def _state(last_user: str, parked: list | None = None, **extra) -> dict:
    return {
        "messages": [
            {"role": "assistant", "content": "Aside from this, is there anything else I can help with?"},
            {"role": "user", "content": last_user},
        ],
        "follow_up_turn_count": 1,
        "follow_up_cannot_answer_count": 0,
        "call_intent": "provider_services",
        "parked_followups": parked or [],
        **extra,
    }


def _mk_agent(monkeypatch, result: FollowUpResult | None = None, captured: dict | None = None):
    import agent.agents.follow_up.agent as fua

    async def fake_extract(*a, **k):
        if captured is not None:
            captured.update(k)
        if result is None:  # pragma: no cover - guard for routing-only tests
            raise AssertionError("LLM path must not run when a parked question routes")
        return result

    monkeypatch.setattr(fua, "extract_follow_up_decision", fake_extract)
    monkeypatch.setattr(fua, "get_follow_up_llm", lambda: object())
    return FollowUpAgent()


_NOTIF_QUESTION = {
    "query": "will I get a notification when the list is sent?",
    "kind": "question",
    "target": "",
}


# ── BUG-1: parked questions with a data owner route, never generate ──────────


class TestBug1ParkedQuestionRouting:
    @pytest.mark.parametrize(
        "query",
        [
            "will I get a notification when the list is sent?",
            "where did you send the provider list?",
            "when will the list be delivered?",
            "what exactly did you send me?",  # detect_request replay hit
        ],
    )
    async def test_list_question_hops_to_delivery_replay(self, monkeypatch, query):
        agent = _mk_agent(monkeypatch, result=None)  # LLM must not run
        state = _state(
            "yes one more thing",
            parked=[{"query": query, "kind": "question", "target": ""}],
            provider_list_sent=True,
        )
        hop = await agent.run(state)
        assert hop["next_node"] == "delivery_management_agent"
        assert hop["pending_cross_agent_request"] == {
            "kind": "replay",
            "target": "provider_list",
            "return_to_agent": "follow_up_agent",
            "return_awaiting": "",
        }
        assert hop["parked_followups"] == []  # the parked item was consumed

    async def test_delivery_answers_from_real_state(self, monkeypatch):
        # Second leg: the owner recaps from state — real channel, real
        # destination, real ZIP. Nothing generated, nothing invented.
        agent = _mk_agent(monkeypatch, result=None)
        state = _state(
            "yes one more thing",
            parked=[dict(_NOTIF_QUESTION)],
            provider_list_sent=True,
            delivery_method="fax",
            fax="5551234567",
            zip_code_used="90210",
            provider_type="dentist",
        )
        hop = await agent.run(state)

        delivery_state = {**state, **{k: v for k, v in hop.items() if k != "messages"}}
        result = await DeliveryManagementAgent().run(delivery_state)
        message = result["messages"]["content"].lower()
        assert "fax" in message and "5551234567" in message and "90210" in message
        assert result["next_node"] == "follow_up_agent"
        assert result["pending_cross_agent_request"] == {}

    async def test_benefits_question_hops_to_benefits_replay(self, monkeypatch):
        agent = _mk_agent(monkeypatch, result=None)
        state = _state(
            "sure",
            parked=[{"query": "what was my deductible again?", "kind": "question", "target": ""}],
            benefits_explained=True,
        )
        hop = await agent.run(state)
        assert hop["next_node"] == "benefits_agent"
        assert hop["pending_cross_agent_request"]["kind"] == "replay"
        assert hop["pending_cross_agent_request"]["target"] == "benefits"
        assert hop["parked_followups"] == []

    async def test_unowned_question_stays_for_llm(self, monkeypatch):
        # No owning capability → the LLM path answers it (grounded), exactly
        # as before.
        captured: dict = {}
        agent = _mk_agent(
            monkeypatch,
            FollowUpResult(follow_up_intent=FollowUpIntent.QUESTION, answer="Our hours are 9 to 5."),
            captured,
        )
        state = _state(
            "hi again",
            parked=[{"query": "what are your office hours?", "kind": "question", "target": ""}],
            provider_list_sent=True,
        )
        out = await agent.run(state)
        assert captured["parked_followups"] == ["what are your office hours?"]
        assert out["messages"]["content"] == "Our hours are 9 to 5."
        assert out["parked_followups"] == []

    async def test_data_missing_stays_for_llm(self, monkeypatch):
        # The capability exists but the data does not (no list ever sent):
        # never route a replay of something never produced — the grounded
        # LLM path (and its cannot-answer machinery) handles it.
        captured: dict = {}
        agent = _mk_agent(
            monkeypatch,
            FollowUpResult(follow_up_intent=FollowUpIntent.QUESTION, answer=None),
            captured,
        )
        state = _state("hello", parked=[dict(_NOTIF_QUESTION)], provider_list_sent=False)
        out = await agent.run(state)
        assert captured["parked_followups"] == [_NOTIF_QUESTION["query"]]
        assert out["follow_up_cannot_answer_count"] == 1  # cannot-answer path
        assert out["parked_followups"] == []

    async def test_routes_on_first_entry_before_opener(self, monkeypatch):
        # Entry turn (no follow_up_turn_count yet): the parked question routes
        # immediately — the member is never asked "anything else?" first.
        agent = _mk_agent(monkeypatch, result=None)
        state = _state("email please", parked=[dict(_NOTIF_QUESTION)], provider_list_sent=True)
        del state["follow_up_turn_count"]
        del state["follow_up_cannot_answer_count"]
        hop = await agent.run(state)
        assert hop["next_node"] == "delivery_management_agent"
        assert hop["pending_cross_agent_request"]["kind"] == "replay"


# ── BUG-1 closure ordering: DONE outranks stale parked items ─────────────────


class TestBug1ClosureOrdering:
    async def test_bare_closure_skips_routing_and_closes(self, monkeypatch, caplog):
        # "no" with a routable parked item: the closure keyword gate skips
        # routing; the classifier's DONE closes immediately and the parked
        # list is dropped with a warning — no trailing answer, no hop.
        agent = _mk_agent(monkeypatch, FollowUpResult(follow_up_intent=FollowUpIntent.DONE))
        monkeypatch.setattr(logging.getLogger("agent"), "propagate", True)
        state = _state("no", parked=[dict(_NOTIF_QUESTION)], provider_list_sent=True)
        with caplog.at_level(logging.WARNING):
            out = await agent.run(state)
        assert out["last_agent_signal"]["status"] == AgentStatus.COMPLETE
        assert out["last_agent_signal"]["closure_requested"] is True
        assert out["parked_followups"] == []
        assert "next_node" not in out or out["next_node"] != "delivery_management_agent"
        dropped = next(r for r in caplog.records if "closing with unresolved parked items" in r.message)
        assert dropped.dropped_parked == [_NOTIF_QUESTION["query"]]

    async def test_done_after_llm_drops_parked_items(self, monkeypatch):
        # A fuzzier closure ("that should be everything, thanks so much") is
        # not in the keyword set — but an unroutable parked item falls to the
        # LLM, DONE comes back, and closure still wins with the list cleared.
        agent = _mk_agent(monkeypatch, FollowUpResult(follow_up_intent=FollowUpIntent.DONE))
        state = _state(
            "that should be everything, thanks so much",
            parked=[{"query": "what are your office hours?", "kind": "question", "target": ""}],
        )
        out = await agent.run(state)
        assert out["last_agent_signal"]["status"] == AgentStatus.COMPLETE
        assert out["last_agent_signal"]["closure_requested"] is True
        assert out["parked_followups"] == []
        # No member-facing answer rides along with the closure.
        assert not out.get("messages")

    async def test_done_without_parked_items_logs_no_warning(self, monkeypatch, caplog):
        agent = _mk_agent(monkeypatch, FollowUpResult(follow_up_intent=FollowUpIntent.DONE))
        monkeypatch.setattr(logging.getLogger("agent"), "propagate", True)
        with caplog.at_level(logging.WARNING):
            out = await agent.run(_state("no thanks"))
        assert out["last_agent_signal"]["closure_requested"] is True
        assert not any("unresolved parked" in r.message for r in caplog.records)
