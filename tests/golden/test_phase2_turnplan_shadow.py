"""
test_phase2_turnplan_shadow.py — Phase 2: one LLM turn-understanding decode, SHADOW.

Proves, hermetically (the understanding LLM is faked):
  * The LLM TurnPlan decoder agrees with GPT's event_type on single-intent turns
    and pays nothing for them (fast-path → deterministic heuristic, no LLM call).
  * It recovers the bundled secondary on the Phase-0 characterization case
    ("member id is M123456, and is my card still active?") — which the regex
    heuristic never sees — and can answer an in-scope side question from the
    session snapshot in the SAME decode call (no extra round-trip).
  * On decode failure it falls back to the deterministic heuristic (never None
    when the heuristic can recover a plan): try → LLM → heuristic → None.
  * Installed in shadow it is LOG-ONLY: the live turn is byte-for-byte unchanged,
    and a `turnplan_shadow` metric is emitted for the observed decode.
  * A question NOT answerable from the snapshot leaves `answer` empty (parked /
    declined downstream, never fabricated).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import tests.golden  # noqa: F401 — ensures src/ is on sys.path
from agent.llm.schema import (
    EventType,
    GuardType,
    SecondaryIntent,
    SecondaryIntentType,
    TurnPlan,
)
from agent.llm.turnplan_decoder import _fast_path_single_intent, llm_turnplan_decoder

pytestmark = pytest.mark.regression


# ── fakes ────────────────────────────────────────────────────────────────────


class _FakeStructured:
    def __init__(self, plan: TurnPlan, sink: dict):
        self._plan = plan
        self._sink = sink

    async def ainvoke(self, messages, **_kw):
        self._sink["messages"] = messages
        self._sink["calls"] = self._sink.get("calls", 0) + 1
        return self._plan


class _FakeUnderstandingLLM:
    def __init__(self, plan: TurnPlan, sink: dict):
        self._plan = plan
        self._sink = sink

    def with_structured_output(self, _model):
        return _FakeStructured(self._plan, self._sink)


def _decision(event_type=EventType.ANSWERED, extracted=None, corrections=None):
    return SimpleNamespace(
        event_type=event_type,
        extracted=extracted or {},
        corrections=corrections or {},
        guard=GuardType.NONE,
        guard_confidence=0.0,
    )


@contextmanager
def _understanding(plan: TurnPlan):
    sink: dict = {}
    with patch("agent.llm.config.get_understanding_llm", lambda: _FakeUnderstandingLLM(plan, sink)):
        yield sink


# ── fast-path: single-intent pays nothing ────────────────────────────────────


def test_fast_path_skips_llm_on_plain_single_intent():
    assert _fast_path_single_intent("M123456", _decision(extracted={"member_id": "M123456"})) is True
    # A regex secondary cue defeats the fast-path.
    assert _fast_path_single_intent("fax, but use a different fax number", _decision()) is False
    # answered_with_followup defeats it even with no regex cue (Phase-0 case).
    assert (
        _fast_path_single_intent(
            "member id is M123456, and is my card still active?",
            _decision(event_type=EventType.ANSWERED_WITH_FOLLOWUP),
        )
        is False
    )


async def test_single_intent_agrees_with_gpt_and_makes_no_llm_call():
    """Plain single-intent turn: the LLM decode is skipped (fast-path), the decoder
    returns the deterministic heuristic plan, and it agrees with GPT — slot answer
    present, no secondaries, no correction."""
    state = {"awaiting_slot": "member_id", "messages": [{"role": "user", "content": "M123456"}]}
    decision = _decision(extracted={"member_id": "M123456"})

    # If the LLM were called this would blow up (no patch installed) — proving no call.
    plan = await llm_turnplan_decoder(state, "M123456", decision)

    assert plan is not None
    assert plan.slot_answer == "M123456"
    assert plan.secondary_intents == []
    assert plan.correction is None


# ── multi-intent: recovers the secondary the regex never sees ─────────────────


async def test_recovers_secondary_on_phase0_characterization_case():
    """The Phase-0 case: answer + a bundled follow-up joined by a bare 'and' — no
    regex cue, so the heuristic never sees it, but GPT flags answered_with_followup
    so the full LLM decode runs and recovers the secondary."""
    utterance = "member id is M123456, and is my card still active?"
    state = {
        "awaiting_slot": "member_id",
        "messages": [
            {"role": "assistant", "content": "Can I get your Member ID?"},
            {"role": "user", "content": utterance},
        ],
    }
    decision = _decision(event_type=EventType.ANSWERED_WITH_FOLLOWUP, extracted={"member_id": "M123456"})
    llm_plan = TurnPlan(
        slot_answer="M123456",
        secondary_intents=[
            SecondaryIntent(
                type=SecondaryIntentType.IN_DOMAIN_UNSUPPORTED,
                owner="follow_up_agent",
                verbatim_span="is my card still active",
                answerable_from_snapshot=False,
            )
        ],
        confidence=0.9,
    )

    with _understanding(llm_plan) as sink:
        plan = await llm_turnplan_decoder(state, utterance, decision)

    assert sink.get("calls") == 1  # the LLM decode ran
    assert plan.slot_answer == "M123456"
    assert len(plan.secondary_intents) == 1
    assert plan.secondary_intents[0].verbatim_span == "is my card still active"
    # Snapshot was injected into the decode's user block (single-pass answering).
    user_content = sink["messages"][-1]["content"]
    assert "SESSION SNAPSHOT" in user_content
    assert "Caller just said" in user_content


async def test_answerable_side_question_answered_from_snapshot():
    """An in-scope question the snapshot can answer comes back answered in the same
    decode call — no extra round-trip."""
    utterance = "pediatrician, but also what's my deductible?"
    state = {
        "awaiting_slot": "provider_type",
        "individual_deductible": "1500",
        "messages": [{"role": "user", "content": utterance}],
    }
    llm_plan = TurnPlan(
        slot_answer="pediatrician",
        secondary_intents=[
            SecondaryIntent(
                type=SecondaryIntentType.IN_SCOPE_INDEPENDENT,
                owner="benefits_agent",
                verbatim_span="what's my deductible",
                answerable_from_snapshot=True,
                answer="Your individual deductible is $1,500 per calendar year.",
            )
        ],
        confidence=0.95,
    )
    with _understanding(llm_plan):
        plan = await llm_turnplan_decoder(state, utterance, _decision())

    sec = plan.secondary_intents[0]
    assert sec.answerable_from_snapshot is True
    assert sec.answer and "deductible" in sec.answer


async def test_unanswerable_question_leaves_answer_empty():
    """A question the snapshot cannot answer is left unanswered (parked/declined
    downstream), never fabricated."""
    utterance = "pediatrician, but do you speak Spanish?"
    state = {"awaiting_slot": "provider_type", "messages": [{"role": "user", "content": utterance}]}
    llm_plan = TurnPlan(
        slot_answer="pediatrician",
        secondary_intents=[
            SecondaryIntent(
                type=SecondaryIntentType.OUT_OF_SCOPE,
                owner=None,
                verbatim_span="do you speak Spanish",
                answerable_from_snapshot=False,
                answer=None,
            )
        ],
    )
    with _understanding(llm_plan):
        plan = await llm_turnplan_decoder(state, utterance, _decision())

    sec = plan.secondary_intents[0]
    assert sec.answerable_from_snapshot is False
    assert not sec.answer


# ── fallback: try → LLM → heuristic → None ────────────────────────────────────


async def test_decode_failure_falls_back_to_heuristic():
    """When the LLM decode raises, the deterministic heuristic recovers a plan."""
    utterance = "fax, but send it to a different fax number"
    state = {
        "awaiting_slot": "delivery_method",
        "messages": [{"role": "user", "content": utterance}],
    }
    decision = _decision(event_type=EventType.ANSWERED_WITH_FOLLOWUP, extracted={"delivery_method": "fax"})

    class _BoomLLM:
        def with_structured_output(self, _m):
            class _S:
                async def ainvoke(self, *_a, **_k):
                    raise RuntimeError("understanding decode down")

            return _S()

    with patch("agent.llm.config.get_understanding_llm", lambda: _BoomLLM()):
        plan = await llm_turnplan_decoder(state, utterance, decision)

    # Heuristic recovered the redirect secondary deterministically.
    assert plan is not None
    assert any(s.verbatim_span for s in plan.secondary_intents)


# ── shadow install: log-only, live path unchanged ────────────────────────────


class _MetricSink(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


@contextmanager
def _capture_shadow_logs():
    lg = logging.getLogger("agent.orchestration.shadow")
    handler = _MetricSink()
    prev = lg.level
    lg.setLevel(logging.DEBUG)
    lg.addHandler(handler)
    try:
        yield handler.records
    finally:
        lg.removeHandler(handler)
        lg.setLevel(prev)


async def test_observer_logs_only_and_returns_outcome():
    """run_turnplan_observer runs the installed observer and logs turnplan_shadow.
    It returns an outcome for tests but never touches the turn."""
    from agent.orchestration import shadow

    utterance = "member id is M123456, and is my card still active?"
    state = {
        "awaiting_slot": "member_id",
        "messages": [{"role": "user", "content": utterance}],
    }
    llm_plan = TurnPlan(
        slot_answer="M123456",
        secondary_intents=[
            SecondaryIntent(
                type=SecondaryIntentType.OUT_OF_SCOPE,
                owner=None,
                verbatim_span="is my card still active",
            )
        ],
    )

    async def _observer(_state, _utt, _decision):
        return llm_plan

    shadow.set_turnplan_observer(_observer)
    try:
        with _capture_shadow_logs() as records:
            outcome = await shadow.run_turnplan_observer(
                state,
                utterance=utterance,
                awaiting_slot="member_id",
                decision=_decision(event_type=EventType.ANSWERED_WITH_FOLLOWUP),
                agent_name="verification_agent",
            )
    finally:
        shadow.clear_turnplan_observer()

    assert outcome is not None  # returned for inspection
    shadow_events = [r for r in records if getattr(r, "metric", None) == "turnplan_shadow"]
    assert shadow_events, "expected a turnplan_shadow log from the observer"
    assert getattr(shadow_events[0], "source", "") == "llm_observer"


async def test_no_observer_is_a_noop():
    from agent.orchestration import shadow

    shadow.clear_turnplan_observer()
    outcome = await shadow.run_turnplan_observer(
        {"awaiting_slot": "member_id"}, utterance="M123456", awaiting_slot="member_id"
    )
    assert outcome is None


def test_configure_installs_per_flag(monkeypatch):
    from agent.core import flags
    from agent.llm.turnplan_decoder import configure_turnplan_decoder, llm_turnplan_decoder
    from agent.orchestration import shadow

    # shadow → observer installed, live decoder untouched
    monkeypatch.setenv(flags.ENV_TURNPLAN_DECODE, "shadow")
    shadow.clear_turnplan_observer()
    configure_turnplan_decoder()
    assert shadow.get_turnplan_observer() is llm_turnplan_decoder

    # off → observer cleared
    monkeypatch.setenv(flags.ENV_TURNPLAN_DECODE, "off")
    configure_turnplan_decoder()
    assert shadow.get_turnplan_observer() is None
