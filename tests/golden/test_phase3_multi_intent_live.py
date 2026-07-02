"""
test_phase3_multi_intent_live.py — Phase 3: multi-intent live, narrated by the
generator (behind MULTI_INTENT_LIVE + TURNPLAN_DECODE=live).

Hermetic: the understanding decode (get_understanding_llm) and the generator
(get_generation_llm) are faked, so we can assert the CONTRACT — one decode + one
generate call, one composed sentence, grounding — without a network.

Covered:
  * The gate case — "my DOB is 01/01/1990, is my card active, and can you refund
    me?" → one sentence that accepts the DOB, answers the card question inline
    from the snapshot, declines the refund, and asks the next slot; exactly one
    decode + one generate call; no fabricated value.
  * In-scope answerable follow-up answered INLINE by default; PARK_ANSWERABLE=true
    parks it instead (enqueued for draining, not answered this turn).
  * Out-of-scope aside → declined, folded into the same sentence (never dropped).
  * Invalidating correction → ack + mark dirty + rewind to owner (still live).
  * Grounding guardrail — an ungrounded generated value falls back to the safe
    template; unresolved-owner secondaries never route.
  * Single-intent clean answer stays byte-identical (fast return, no compose call).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

import tests.golden  # noqa: F401 — ensures src/ is on sys.path
from agent.llm.schema import (
    Correction,
    EventType,
    GuardType,
    SecondaryIntent,
    SecondaryIntentType,
    TurnPlan,
)
from agent.orchestration.registry import queue_owners

pytestmark = pytest.mark.regression


# ── fakes: one understanding decode + one generation call, both counted ───────


class _Counter:
    def __init__(self):
        self.decode = 0
        self.generate = 0
        self.gen_context = ""


class _FakeStructured:
    def __init__(self, plan, counter):
        self._plan = plan
        self._c = counter

    async def ainvoke(self, _messages, **_kw):
        self._c.decode += 1
        return self._plan


class _FakeUnderstandingLLM:
    def __init__(self, plan, counter):
        self._plan = plan
        self._c = counter

    def with_structured_output(self, _m):
        return _FakeStructured(self._plan, self._c)


class _FakeGenLLM:
    """Echoes a deterministic composed sentence, recording the structured context
    it was handed so we can assert clause ordering / grounding inputs."""

    def __init__(self, counter, content):
        self._c = counter
        self._content = content

    async def ainvoke(self, messages, **_kw):
        self._c.generate += 1
        self._c.gen_context = messages[-1].content
        return SimpleNamespace(content=self._content)


def _member_config(slot="dob"):
    from agent.core.slot_manager import _InternalSlotConfig
    from agent.slots import normalizers as N
    from agent.slots import validators as V
    from agent.slots.types import SlotType

    specs = {
        "dob": (N.normalize_dob, V.validate_dob, SlotType.DOB),
        "provider_type": (N.normalize_provider_type, V.validate_provider_type, SlotType.PROVIDER_TYPE),
        "delivery_method": (
            N.normalize_delivery_method,
            V.validate_delivery_method,
            SlotType.DELIVERY_METHOD,
        ),
    }
    norm, val, st = specs[slot]
    return _InternalSlotConfig(slot_name=slot, prompt="", normalizer=norm, validator=val, slot_type=st)


def _probe_agent(state, name="verification_agent"):
    from agent.core.agent import BaseAgent

    class _ProbeAgent(BaseAgent):
        AGENT_NAME = name

        async def run(self, _s):  # pragma: no cover
            return {}

    return _ProbeAgent.from_state(state)


def _decision(event_type=EventType.ANSWERED_WITH_FOLLOWUP, extracted=None):
    return SimpleNamespace(
        event_type=event_type,
        extracted=extracted or {},
        corrections={},
        guard=GuardType.NONE,
        guard_confidence=0.0,
    )


def _last_ai(interrupt):
    m = interrupt.get("messages")
    return m.get("content") if isinstance(m, dict) else ""


@pytest.fixture(autouse=True)
def _live_flags(monkeypatch):
    monkeypatch.setenv("MULTI_INTENT_LIVE", "true")
    monkeypatch.setenv("TURNPLAN_DECODE", "live")
    # Install the async LLM decoder as the acting decoder for these tests.
    from agent.llm.turnplan_decoder import llm_turnplan_decoder
    from agent.orchestration import shadow

    prev = shadow.get_shadow_decoder()
    shadow.set_shadow_decoder(llm_turnplan_decoder)
    try:
        yield
    finally:
        shadow.set_shadow_decoder(prev)


# ── the gate case ─────────────────────────────────────────────────────────────


async def test_gate_case_one_sentence_one_decode_one_generate(monkeypatch):
    utterance = "my DOB is 01/01/1990, is my card active, and can you refund me?"
    state = {
        "awaiting_slot": "dob",
        "dob": "",
        "slot_attempts": {},
        "first_name": "Emily",
        "member_id": "M123456",  # snapshot fact enabling the card answer
        "call_intent": "provider_services",
        "active_agent": "verification_agent",
        "messages": [
            {"role": "assistant", "content": "And your date of birth?"},
            {"role": "user", "content": utterance},
        ],
    }
    plan = TurnPlan(
        slot_answer="01/01/1990",
        secondary_intents=[
            SecondaryIntent(
                type=SecondaryIntentType.IN_SCOPE_INDEPENDENT,
                owner="follow_up_agent",
                verbatim_span="is my card active",
                answerable_from_snapshot=True,
                answer="Yes, your member card is active.",
            ),
            SecondaryIntent(
                type=SecondaryIntentType.OUT_OF_SCOPE,
                owner=None,
                verbatim_span="can you refund me",
            ),
        ],
        confidence=0.95,
    )
    counter = _Counter()
    composed = (
        "Got your date of birth — yes, your member card is active, "
        "I can't help with a refund here, and now, are you the plan holder?"
    )

    agent = _probe_agent(state)
    with patch("agent.llm.config.get_understanding_llm", lambda: _FakeUnderstandingLLM(plan, counter)):
        with patch("agent.llm.response_generator.get_generation_llm", lambda: _FakeGenLLM(counter, composed)):
            value, interrupt = await agent._collect_slot(
                dict(state),
                _member_config("dob"),
                state["messages"],
                pre_extracted="01/01/1990",
                decision=_decision(extracted={"dob": "01/01/1990"}),
                pending_slots=["relationship"],
            )

    # (a) DOB accepted.
    assert value == "01/01/1990"
    # exactly one decode + one generate call.
    assert counter.decode == 1, f"expected 1 decode, got {counter.decode}"
    assert counter.generate == 1, f"expected 1 generate, got {counter.generate}"
    # ONE composed sentence reached the caller.
    assert _last_ai(interrupt) == composed
    # The structured context handed to the generator carried every clause…
    ctx_text = counter.gen_context
    assert "Speech act: multi_intent" in ctx_text
    assert "Validated answer this turn: 01/01/1990" in ctx_text  # (a) accept
    assert "Answer to include" in ctx_text and "card is active" in ctx_text  # (b) inline
    assert "Declined" in ctx_text  # (c) refund declined
    assert "Next, ask for:" in ctx_text  # (d) next ask
    # …and the turn advanced to the next slot; nothing fabricated in state.
    assert interrupt.get("awaiting_slot") == "relationship"
    # The answered-inline follow-up was NOT parked (handled this turn).
    assert not interrupt.get("intent_queue")


# ── inline vs park dial ───────────────────────────────────────────────────────


async def _run_side_question(monkeypatch, *, park_answerable: bool):
    utterance = "pediatrician, and also what's my deductible?"
    state = {
        "awaiting_slot": "provider_type",
        "provider_type": "",
        "slot_attempts": {},
        "individual_deductible": "1500",
        "call_intent": "provider_services",
        "active_agent": "provider_search_agent",
        "messages": [
            {"role": "assistant", "content": "What type of provider?"},
            {"role": "user", "content": utterance},
        ],
    }
    plan = TurnPlan(
        slot_answer="pediatrician",
        secondary_intents=[
            SecondaryIntent(
                type=SecondaryIntentType.IN_SCOPE_INDEPENDENT,
                owner="benefits_agent",
                verbatim_span="what's my deductible",
                answerable_from_snapshot=True,
                answer="Your individual deductible is $1,500 per year.",
            )
        ],
        confidence=0.95,
    )
    counter = _Counter()
    if park_answerable:
        monkeypatch.setenv("PARK_ANSWERABLE", "true")
    else:
        monkeypatch.delenv("PARK_ANSWERABLE", raising=False)

    agent = _probe_agent(state, name="provider_search_agent")
    with patch("agent.llm.config.get_understanding_llm", lambda: _FakeUnderstandingLLM(plan, counter)):
        with patch(
            "agent.llm.response_generator.get_generation_llm",
            lambda: _FakeGenLLM(counter, "Great, a pediatrician — noted; what ZIP should I use?"),
        ):
            value, interrupt = await agent._collect_slot(
                dict(state),
                _member_config("provider_type"),
                state["messages"],
                pre_extracted="pediatrician",
                decision=_decision(extracted={"provider_type": "pediatrician"}),
                pending_slots=["zip_code"],
            )
    return counter, interrupt, value


async def test_answerable_followup_answered_inline_by_default(monkeypatch):
    counter, interrupt, value = await _run_side_question(monkeypatch, park_answerable=False)
    assert value == "Pediatrician"
    # Answered inline → context carries the grounded answer, and nothing is parked.
    assert "Answer to include" in counter.gen_context
    assert "deductible" in counter.gen_context
    assert not interrupt.get("intent_queue")


async def test_answerable_followup_parked_when_flag_set(monkeypatch):
    counter, interrupt, _v = await _run_side_question(monkeypatch, park_answerable=True)
    # Parked → enqueued for draining, and NOT answered inline this turn.
    assert "benefits_agent" in queue_owners(interrupt.get("intent_queue"))
    assert "Answer to include" not in counter.gen_context
    assert "Parked" in counter.gen_context


# ── invalidating correction still live (ack + dirty + rewind) ─────────────────


async def test_invalidating_correction_marks_dirty_and_rewinds(monkeypatch):
    utterance = "fax, but I need to update my ZIP code to 94110"
    state = {
        "awaiting_slot": "delivery_method",
        "delivery_method": "",
        "slot_attempts": {},
        "zip_code": "94107",
        "provider_list_sent": False,
        "dirty_artifacts": {},
        "call_intent": "provider_services",
        "active_agent": "delivery_management_agent",
        "messages": [
            {"role": "assistant", "content": "Fax or email?"},
            {"role": "user", "content": utterance},
        ],
    }
    plan = TurnPlan(
        slot_answer="fax",
        secondary_intents=[],
        correction=Correction(field="zip_code", owner="provider_search_agent", new_value="94110"),
        confidence=0.95,
    )
    counter = _Counter()
    agent = _probe_agent(state, name="delivery_management_agent")
    with patch("agent.llm.config.get_understanding_llm", lambda: _FakeUnderstandingLLM(plan, counter)):
        with patch(
            "agent.llm.response_generator.get_generation_llm",
            lambda: _FakeGenLLM(counter, "Sure, fax it is — let's update your ZIP first; what's the ZIP?"),
        ):
            value, interrupt = await agent._collect_slot(
                dict(state),
                _member_config("delivery_method"),
                state["messages"],
                pre_extracted="fax",
                decision=_decision(extracted={"delivery_method": "fax"}),
                pending_slots=[],
            )

    assert value == "fax"
    assert interrupt.get("dirty_artifacts", {}).get("provider_list") is True
    assert interrupt.get("next_node") == "provider_search_agent"
    assert interrupt.get("awaiting_slot") == "zip_code"
    assert interrupt.get("zip_code_used") == ""
    assert "Correction acknowledged: ZIP code" in counter.gen_context


# ── grounding guardrail: ungrounded generated value → safe template ───────────


async def test_ungrounded_generated_value_falls_back_to_template(monkeypatch):
    utterance = "pediatrician, and also what's my deductible?"
    state = {
        "awaiting_slot": "provider_type",
        "provider_type": "",
        "slot_attempts": {},
        "individual_deductible": "1500",
        "active_agent": "provider_search_agent",
        "messages": [
            {"role": "assistant", "content": "What type of provider?"},
            {"role": "user", "content": utterance},
        ],
    }
    plan = TurnPlan(
        slot_answer="pediatrician",
        secondary_intents=[
            SecondaryIntent(
                type=SecondaryIntentType.IN_SCOPE_INDEPENDENT,
                owner="benefits_agent",
                verbatim_span="what's my deductible",
                answerable_from_snapshot=True,
                answer="Your deductible is met.",
            )
        ],
        confidence=0.95,
    )
    counter = _Counter()
    # The generator hallucinates a Member ID that was never grounded this turn.
    hallucinated = "Sure — your Member ID M999999 is all set; what ZIP should I use?"
    agent = _probe_agent(state, name="provider_search_agent")
    with patch("agent.llm.config.get_understanding_llm", lambda: _FakeUnderstandingLLM(plan, counter)):
        with patch(
            "agent.llm.response_generator.get_generation_llm", lambda: _FakeGenLLM(counter, hallucinated)
        ):
            _v, interrupt = await agent._collect_slot(
                dict(state),
                _member_config("provider_type"),
                state["messages"],
                pre_extracted="pediatrician",
                decision=_decision(extracted={"provider_type": "pediatrician"}),
                pending_slots=["zip_code"],
            )

    spoken = _last_ai(interrupt)
    assert "M999999" not in spoken, "ungrounded value leaked into the spoken turn"


# ── single-intent stays byte-identical (fast return, no compose) ──────────────


async def test_single_intent_clean_answer_no_compose_call(monkeypatch):
    """A truly single-intent clean answer keeps the fast return: no secondary, so
    the LLM decode fast-paths (no decode call) and no generate call is made."""
    state = {
        "awaiting_slot": "provider_type",
        "provider_type": "",
        "slot_attempts": {},
        "active_agent": "provider_search_agent",
        "messages": [
            {"role": "assistant", "content": "What type of provider?"},
            {"role": "user", "content": "pediatrician"},
        ],
    }
    counter = _Counter()
    agent = _probe_agent(state, name="provider_search_agent")
    with patch("agent.llm.config.get_understanding_llm", lambda: _FakeUnderstandingLLM(None, counter)):
        with patch("agent.llm.response_generator.get_generation_llm", lambda: _FakeGenLLM(counter, "x")):
            value, interrupt = await agent._collect_slot(
                dict(state),
                _member_config("provider_type"),
                state["messages"],
                pre_extracted="pediatrician",
                decision=_decision(
                    event_type=EventType.ANSWERED, extracted={"provider_type": "pediatrician"}
                ),
                pending_slots=["zip_code"],
            )

    assert value == "Pediatrician"
    assert interrupt is None  # clean fast return — pipeline advances normally
    assert counter.decode == 0  # fast-path skipped the LLM decode
    assert counter.generate == 0  # no composition


# ── Phase-0 follow-up case now flips to answered-or-parked ─────────────────────


async def test_phase0_followup_case_now_parked_not_dropped(monkeypatch):
    """The Phase-0 characterization case ("member id is M123456, and is my card
    still active?") — acknowledged-only today — is now parked (or answered), never
    silently dropped, when MULTI_INTENT_LIVE is on."""
    from agent.core.slot_manager import _InternalSlotConfig
    from agent.slots.normalizers import normalize_member_id
    from agent.slots.types import SlotType
    from agent.slots.validators import validate_member_id

    utterance = "member id is M123456, and is my card still active?"
    state = {
        "awaiting_slot": "member_id",
        "member_id": "",
        "slot_attempts": {},
        "active_agent": "verification_agent",
        "messages": [
            {"role": "assistant", "content": "Can I get your Member ID?"},
            {"role": "user", "content": utterance},
        ],
    }
    plan = TurnPlan(
        slot_answer="M123456",
        secondary_intents=[
            SecondaryIntent(
                type=SecondaryIntentType.IN_SCOPE_INDEPENDENT,
                owner="follow_up_agent",
                verbatim_span="is my card still active",
                answerable_from_snapshot=False,  # snapshot has no card-status → park it
            )
        ],
        confidence=0.9,
    )
    counter = _Counter()
    cfg = _InternalSlotConfig(
        slot_name="member_id",
        prompt="",
        normalizer=normalize_member_id,
        validator=validate_member_id,
        slot_type=SlotType.MEMBER_ID,
    )
    agent = _probe_agent(state)
    with patch("agent.llm.config.get_understanding_llm", lambda: _FakeUnderstandingLLM(plan, counter)):
        _msg = "Thanks — I'll come back to your card question; and your date of birth?"
        with patch(
            "agent.llm.response_generator.get_generation_llm",
            lambda: _FakeGenLLM(counter, _msg),
        ):
            value, interrupt = await agent._collect_slot(
                dict(state),
                cfg,
                state["messages"],
                pre_extracted="M123456",
                decision=_decision(extracted={"member_id": "M123456"}),
                pending_slots=["dob"],
            )

    assert value == "M123456"
    # The follow-up is now PARKED for draining — not acknowledged-only/dropped.
    assert "follow_up_agent" in queue_owners(interrupt.get("intent_queue"))


# ── never guess: unresolved owner isn't routed; low confidence asks ───────────


async def test_unresolved_owner_secondary_not_routed(monkeypatch):
    """A secondary whose owner doesn't resolve is dropped by the resolver before it
    can reach the member — never parked, never routed."""
    utterance = "pediatrician, and also escalate to the moon department"
    state = {
        "awaiting_slot": "provider_type",
        "provider_type": "",
        "slot_attempts": {},
        "active_agent": "provider_search_agent",
        "messages": [
            {"role": "assistant", "content": "What type of provider?"},
            {"role": "user", "content": utterance},
        ],
    }
    plan = TurnPlan(
        slot_answer="pediatrician",
        secondary_intents=[
            SecondaryIntent(
                type=SecondaryIntentType.IN_SCOPE_INDEPENDENT,
                owner="moon_department_agent",  # not in the registry → dropped
                verbatim_span="escalate to the moon department",
            )
        ],
        confidence=0.9,
    )
    counter = _Counter()
    agent = _probe_agent(state, name="provider_search_agent")
    with patch("agent.llm.config.get_understanding_llm", lambda: _FakeUnderstandingLLM(plan, counter)):
        with patch(
            "agent.llm.response_generator.get_generation_llm",
            lambda: _FakeGenLLM(counter, "Let's keep going — what ZIP should I use?"),
        ):
            value, interrupt = await agent._collect_slot(
                dict(state),
                _member_config("provider_type"),
                state["messages"],
                pre_extracted="pediatrician",
                decision=_decision(extracted={"provider_type": "pediatrician"}),
                pending_slots=["zip_code"],
            )

    assert value == "Pediatrician"
    # The unresolved owner was dropped — never parked, never routed.
    assert "moon_department_agent" not in queue_owners(interrupt.get("intent_queue"))
    assert interrupt.get("next_node") != "moon_department_agent"


async def test_low_confidence_secondary_asks_never_acts(monkeypatch):
    """A low-confidence decode routes to clarify — the live path asks, never acts
    (no park, no route)."""
    utterance = "pediatrician, mrrphmm something"
    state = {
        "awaiting_slot": "provider_type",
        "provider_type": "",
        "slot_attempts": {},
        "active_agent": "provider_search_agent",
        "messages": [
            {"role": "assistant", "content": "What type of provider?"},
            {"role": "user", "content": utterance},
        ],
    }
    plan = TurnPlan(
        slot_answer="pediatrician",
        secondary_intents=[
            SecondaryIntent(
                type=SecondaryIntentType.UNKNOWN,
                owner=None,
                verbatim_span="mrrphmm something",
            )
        ],
        confidence=0.2,  # below the resolver threshold → clarify
    )
    counter = _Counter()
    agent = _probe_agent(state, name="provider_search_agent")
    with patch("agent.llm.config.get_understanding_llm", lambda: _FakeUnderstandingLLM(plan, counter)):
        with patch(
            "agent.llm.response_generator.get_generation_llm",
            lambda: _FakeGenLLM(counter, "Just to be sure — could you say that again?"),
        ):
            _v, interrupt = await agent._collect_slot(
                dict(state),
                _member_config("provider_type"),
                state["messages"],
                pre_extracted="pediatrician",
                decision=_decision(extracted={"provider_type": "pediatrician"}),
                pending_slots=["zip_code"],
            )

    # Nothing was parked or routed on a low-confidence turn.
    assert not (interrupt.get("intent_queue") or [])
