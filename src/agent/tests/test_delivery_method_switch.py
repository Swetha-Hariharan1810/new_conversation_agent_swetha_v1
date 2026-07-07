"""
Delivery method switch + update routing inside the manual confirmation
branches (Phase 2 — BUG-3, BUG-5).

BUG-3: a channel switch during fax/email confirmation ("actually email is
better", "can you email it instead") was treated as a failed confirmation
answer and re-asked verbatim. Now _maybe_switch_method honors it across all
extraction variants: an explicit delivery_method, an LLM redo flag, the raw
phrase only (regex fallback), a WAIT mislabel, and the other channel's value
answering this channel's question.

BUG-5: an update statement ("my ZIP code changed", "I moved") during a
confirmation read-back burned retries instead of routing to the owning
agent. Now it routes across all extraction variants — including when the
LLM returned nothing, mislabeled the turn WAIT, or misread it as a decline.
"""

import pytest

from agent.agents.delivery_management.agent import DeliveryManagementAgent
from agent.llm.schema import EventType, RequestKind, WorkerResult

# ── harness ──────────────────────────────────────────────────────────────────


def _mk_agent(monkeypatch, result: WorkerResult) -> DeliveryManagementAgent:
    import agent.agents.delivery_management.agent as dma

    async def fake_extract(*a, **k):
        return result

    monkeypatch.setattr(dma, "extract_delivery_management_decision", fake_extract)
    monkeypatch.setattr(dma, "get_extraction_llm", lambda: object())
    return DeliveryManagementAgent()


def _state(awaiting="fax_confirmed", user="", **over):
    state = {
        "messages": [
            {
                "role": "assistant",
                "content": "The fax number we have on file is 555-123-4567. Is this correct?",
            },
            {"role": "user", "content": user},
        ],
        "awaiting_slot": awaiting,
        "delivery_method": "fax",
        "fax": "5551234567",
        "email": "jane.doe@example.com",
        "zip_code": "90210",
        "zip_code_used": "90210",
        "parked_followups": [],
    }
    state.update(over)
    return state


# ── BUG-3: channel switch during contact confirmation ────────────────────────


class TestBug3MethodSwitch:
    @pytest.mark.parametrize(
        "result,utterance",
        [
            # LLM extracted the new method (prompt-level fix)
            (WorkerResult(extracted={"delivery_method": "email"}), "actually email is better"),
            # LLM flagged the redo but produced no method
            (
                WorkerResult(update_target="delivery_method", request_kind=RequestKind.REDO),
                "can you email it instead",
            ),
            # LLM missed everything — regex fallback carries the turn
            (WorkerResult(), "send it to my email instead of fax"),
            (WorkerResult(), "just use the other method"),
            # LLM mislabeled the turn WAIT — veto + fill
            (WorkerResult(event_type=EventType.WAIT), "can you email it instead"),
        ],
    )
    async def test_switch_fax_to_email_pre_dispatch(self, monkeypatch, result, utterance):
        agent = _mk_agent(monkeypatch, result)
        out = await agent.run(_state(user=utterance))
        assert out["awaiting_slot"] == "email_confirmed"
        assert out["delivery_method"] == "email"
        assert out["pending_fax"] == ""
        # The on-file email is read back for confirmation, spoken form.
        assert "jane dot doe at example dot com" in out["messages"]["content"]

    async def test_switch_email_to_fax(self, monkeypatch):
        agent = _mk_agent(monkeypatch, WorkerResult())
        state = _state(
            awaiting="email_confirmed",
            user="actually fax works better",
            delivery_method="email",
        )
        state["messages"][0]["content"] = (
            "I'll send it to jane dot doe at example dot com. Is that the right email address?"
        )
        out = await agent.run(state)
        assert out["awaiting_slot"] == "fax_confirmed"
        assert out["delivery_method"] == "fax"
        assert out["pending_email"] == ""
        assert "555" in out["messages"]["content"]

    async def test_switch_carries_new_email_value(self, monkeypatch):
        # Method + the new contact in one utterance → straight to the
        # pending-value read-back, no extra "what email should we use?" turn.
        result = WorkerResult(extracted={"delivery_method": "email", "email": "new.addr@example.com"})
        agent = _mk_agent(monkeypatch, result)
        out = await agent.run(_state(user="just email it to new.addr@example.com"))
        assert out["awaiting_slot"] == "email_confirmed"
        assert out["pending_email"] == "new.addr@example.com"
        assert out["delivery_method"] == "email"
        assert out["pending_fax"] == ""
        assert "new dot addr at example dot com" in out["messages"]["content"]

    async def test_other_channel_value_alone_implies_switch(self, monkeypatch):
        # Answering the fax question with an email address IS the switch.
        result = WorkerResult(extracted={"email": "new.addr@example.com"})
        agent = _mk_agent(monkeypatch, result)
        out = await agent.run(_state(user="send it to new.addr@example.com"))
        assert out["awaiting_slot"] == "email_confirmed"
        assert out["delivery_method"] == "email"
        assert out["pending_email"] == "new.addr@example.com"

    async def test_same_channel_redirect_is_not_a_switch(self, monkeypatch):
        # "use a different fax" redirects on the SAME channel — the decline
        # path asks for the new fax; the method must not flip.
        result = WorkerResult(extracted={"fax_confirmed": "no"})
        agent = _mk_agent(monkeypatch, result)
        out = await agent.run(_state(user="use a different fax number"))
        assert out["awaiting_slot"] == "fax"
        assert out.get("delivery_method", "fax") == "fax"

    async def test_switch_from_fax_update_branch(self, monkeypatch):
        # Even after declining the on-file fax (awaiting the new fax value),
        # the caller can still switch channels instead of giving a number.
        agent = _mk_agent(monkeypatch, WorkerResult())
        out = await agent.run(_state(awaiting="fax", user="actually email is better"))
        assert out["awaiting_slot"] == "email_confirmed"
        assert out["delivery_method"] == "email"

    async def test_switch_resets_abandoned_channel_counters(self, monkeypatch):
        agent = _mk_agent(monkeypatch, WorkerResult(extracted={"delivery_method": "email"}))
        state = _state(
            user="actually email is better",
            slot_attempts={
                "fax_change_cycles": {"attempt_count": 2, "confirmed": False, "last_value": None},
                "fax_confirmed": {"attempt_count": 2, "confirmed": False, "last_value": None},
            },
        )
        out = await agent.run(state)
        assert out["slot_attempts"]["fax_change_cycles"]["attempt_count"] == 0
        assert out["slot_attempts"]["fax_confirmed"]["attempt_count"] == 0

    async def test_post_dispatch_switch_begins_redispatch(self, monkeypatch):
        # After the list went out, a switch is a re-send (Phase 6 redo), not
        # an in-place method flip.
        agent = _mk_agent(monkeypatch, WorkerResult())
        state = _state(
            awaiting="benefits_response",
            user="can you send it to my email instead",
            provider_list_sent=True,
            benefits_offer_made=True,
        )
        out = await agent.run(state)
        assert out["awaiting_slot"] == "delivery_method"
        assert out["pending_cross_agent_request"]["kind"] == "redo"
        assert out["pending_cross_agent_request"]["return_awaiting"] == "benefits_response"


# ── BUG-5: slot updates during confirmation route, never retry over them ─────


class TestBug5UpdateRouting:
    @pytest.mark.parametrize(
        "result,utterance",
        [
            # LLM produced the target (pre-Phase-2 behavior, still works)
            (
                WorkerResult(update_target="zip_code", request_kind=RequestKind.UPDATE),
                "my zip code changed",
            ),
            # LLM put it in corrections{} — the shim promotes it
            (WorkerResult(corrections={"zip_code": "60660"}), "actually my zip is 60660"),
            # LLM produced nothing — regex fallback
            (WorkerResult(), "wait — my zip code changed, i moved"),
            (WorkerResult(), "i just moved"),
            # LLM mislabeled the correction turn WAIT — veto + fill
            (WorkerResult(event_type=EventType.WAIT), "hold on, my zip code changed"),
            # LLM misread the change statement as a decline — the routed
            # update wins before the "no" path can fire
            (WorkerResult(extracted={"fax_confirmed": "no"}), "no wait, my zip code is wrong"),
        ],
    )
    async def test_zip_update_routes_from_fax_confirmed(self, monkeypatch, result, utterance):
        agent = _mk_agent(monkeypatch, result)
        out = await agent.run(_state(user=utterance))
        assert out["next_node"] == "provider_search_agent"
        assert out["pending_cross_agent_request"]["kind"] == "update"
        assert out["pending_cross_agent_request"]["target"] == "zip_code"
        assert out["pending_cross_agent_request"]["return_awaiting"] == "fax_confirmed"
        # The stale list and the ZIP it was built from are invalidated.
        assert out["zip_code_used"] == ""

    async def test_zip_update_routes_from_email_confirmed(self, monkeypatch):
        agent = _mk_agent(monkeypatch, WorkerResult())
        state = _state(awaiting="email_confirmed", user="my zip is wrong", delivery_method="email")
        out = await agent.run(state)
        assert out["next_node"] == "provider_search_agent"
        assert out["pending_cross_agent_request"]["return_awaiting"] == "email_confirmed"

    async def test_fallback_same_channel_update_asks_new_value(self, monkeypatch):
        # LLM produced nothing usable; regex says "change my fax number" —
        # decline-equivalent: ask for the new fax, never a verbatim retry.
        agent = _mk_agent(monkeypatch, WorkerResult())
        out = await agent.run(_state(user="i want to change my fax number"))
        assert out["awaiting_slot"] == "fax"
        assert out["pending_fax"] == ""

    async def test_unclassifiable_turn_still_retries(self, monkeypatch):
        # Genuinely unclear turns keep the existing retry behavior.
        agent = _mk_agent(monkeypatch, WorkerResult())
        out = await agent.run(_state(user="hmm what do you mean exactly"))
        assert out["awaiting_slot"] == "fax_confirmed"
        assert "555" in out["messages"]["content"]  # read-back retried once
        assert "pending_cross_agent_request" not in out
