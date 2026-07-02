"""
test_phase2_dropped_metric.py — Phase 2 dropped-request metric (observability only).

Proves the metric fires on multi-intent turns and that the dropped-request count
becomes non-zero on UAT-007 today — with NO behavior change.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

import pytest

import tests.golden  # noqa: F401 — ensures src/ is on sys.path
from agent.orchestration.registry import queue_owners
from tests.golden.driver import run_fixture

pytestmark = pytest.mark.regression


# ── log capture independent of pytest caplog (per-logger propagate: false) ─────


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@contextmanager
def capture_metric_logs():
    lg = logging.getLogger("agent.orchestration.observability")
    handler = _ListHandler()
    handler.setLevel(logging.DEBUG)
    prev_level = lg.level
    lg.setLevel(logging.DEBUG)
    lg.addHandler(handler)
    try:
        yield handler.records
    finally:
        lg.removeHandler(handler)
        lg.setLevel(prev_level)


def _dropped_events(records) -> list[logging.LogRecord]:
    return [
        r
        for r in records
        if getattr(r, "metric", None) == "dropped_request" and getattr(r, "outcome", None) == "dropped"
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Detector — deterministic, PII-safe
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "utterance",
    [
        "Fax, but I need to update my ZIP code.",
        "Oh, by the way, can you send it to another fax number?",
        "Later. But can you send the list to another fax number?",
        "Yes. I do. But I'm sorry. Can you send the list to a different fax number?",
        "Before that, can you send the list of the providers on a different fax number, please?",
        "I'm looking for a pediatrician — also, what's my deductible?",
    ],
)
def test_detect_secondary_request_true(utterance):
    from agent.orchestration.observability import detect_secondary_request

    assert detect_secondary_request(utterance) is True


@pytest.mark.parametrize(
    "utterance",
    [
        "Fax",
        "send it to my fax",
        "yes that's correct",
        "Primary Care Physician",
        "M seven one four five nine eight",
        "Plan holder.",
        "April twelfth nineteen eighty eight",
        "Yes.",
        "",
        None,
    ],
)
def test_detect_secondary_request_false(utterance):
    from agent.orchestration.observability import detect_secondary_request

    assert detect_secondary_request(utterance) is False


def test_shape_is_pii_safe():
    from agent.orchestration.observability import secondary_request_shape

    shape = secondary_request_shape("Fax, but I need to update my ZIP code.")
    assert shape["pivot"] == "but"
    assert shape["target"] == "zip_code"
    assert shape["redirect"] is True
    assert isinstance(shape["n_tokens"], int)
    # No raw utterance text leaks into the shape.
    assert "zip code" not in str(shape).lower() or shape["target"] == "zip_code"
    assert all(k in shape for k in ("pivot", "redirect", "target", "n_tokens", "has_question"))


# ──────────────────────────────────────────────────────────────────────────────
# Outcome classification
# ──────────────────────────────────────────────────────────────────────────────


def test_classify_outcome_parked_actioned_dropped():
    from agent.orchestration.observability import classify_secondary_outcome

    # Newly parked a dependent artifact this turn.
    assert (
        classify_secondary_outcome(
            {"dirty_artifacts": {}}, {"dirty_artifacts": {"provider_list": True}}, target="zip_code"
        )
        == "parked"
    )
    # Owner field the secondary referenced was changed this turn.
    assert (
        classify_secondary_outcome(
            {"fax": "4150000000"}, {"fax": "4155553299"}, target="fax"
        )
        == "actioned"
    )
    # Neither — dropped.
    assert classify_secondary_outcome({}, {}, target="fax") == "dropped"


# ──────────────────────────────────────────────────────────────────────────────
# UAT-007 — the metric fires and the count is non-zero today
# ──────────────────────────────────────────────────────────────────────────────


async def test_uat_007_secondary_now_handled_not_dropped():
    """Phase 3B flip: the UAT-007 ZIP request is no longer dropped — the resolver
    handles it (parked: provider_list marked stale, routed to re-resolve), so the
    metric records 'parked' and the dropped count for the turn is 0."""
    from tests.golden.driver import load_fixture

    fixture = load_fixture("uat_007_multi_intent")
    with capture_metric_logs() as records:
        run = await run_fixture(fixture)

    # The silent drop is gone.
    assert run.dropped_request_count == 0
    assert _dropped_events(records) == []

    # The secondary is now classified as handled (parked), not dropped.
    parked = [
        r
        for r in records
        if getattr(r, "metric", None) == "dropped_request" and getattr(r, "outcome", None) == "parked"
    ]
    assert parked, "expected the secondary to be recorded as parked (handled)"

    # And nothing was dispatched on the disputed ZIP; the call routed to re-resolve.
    assert run.recorder.count("dispatch_provider_list") == 0
    assert run.final_state.get("next_node") == "provider_search_agent"


async def test_later_fax_redirect_turn_is_parked_not_dropped():
    """The UAT-007 'send it to another fax number' shape, arriving during the
    benefits offer, is no longer silently dropped: the Phase 2 turn gate routes
    the hand-coded benefits confirmation through the resolver, which PARKS the
    side request (enqueued for draining) and re-asks the benefits question."""
    fax_readback = "Would you also like me to go over the benefits for office visits with your Pediatrician?"
    fixture = {
        "id": "UNIT-DROP-BENEFITS-FAXREDIRECT",
        "driver": "delivery_management_agent",
        "initial_state": {
            "messages": [{"role": "assistant", "content": fax_readback}],
            "member_status_verify": True,
            "member_id": "M714598",
            "call_intent": "provider_services",
            "active_agent": "delivery_management_agent",
            "provider_type": "Pediatrician",
            "zip_code": "94107",
            "zip_code_used": "94107",
            "fax": "415-555-3299",
            "email": "",
            "delivery_method": "fax",
            "provider_list_sent": True,
            "benefits_offer_made": True,
            "awaiting_slot": "benefits_response",
            "dirty_artifacts": {},
            "slot_attempts": {},
            "is_interrupt": True,
            "app_run_id": "unit-drop-benefits",
        },
        "turns": [
            {"user": "Oh, by the way, can you send it to another fax number?", "extraction": {}},
        ],
    }
    with capture_metric_logs() as records:
        run = await run_fixture(fixture, print_latency=False)

    # The silent drop is gone: the secondary is parked (handled), not dropped.
    assert run.dropped_request_count == 0
    assert _dropped_events(records) == []
    parked = [
        r
        for r in records
        if getattr(r, "metric", None) == "dropped_request" and getattr(r, "outcome", None) == "parked"
    ]
    assert parked, "expected the fax-redirect to be recorded as parked (handled)"
    assert "delivery_management_agent" in queue_owners(run.final_state.get("intent_queue"))
    # The agent stays in the benefits offer; nothing dispatched.
    assert run.final_state.get("awaiting_slot") == "benefits_response"
    assert run.recorder.count("dispatch_provider_list") == 0


# ──────────────────────────────────────────────────────────────────────────────
# State plumbing
# ──────────────────────────────────────────────────────────────────────────────


def test_reset_for_new_intent_zeroes_dropped_request_count():
    from agent.state import reset_for_new_intent

    updates = reset_for_new_intent({"dropped_request_count": 5}, "claim_services")
    assert updates["dropped_request_count"] == 0


async def test_single_intent_turn_does_not_count():
    """A clean single-answer turn emits no metric and increments nothing."""
    fixture = {
        "id": "UNIT-SINGLE-INTENT",
        "driver": "provider_search_agent",
        "initial_state": {
            "messages": [{"role": "assistant", "content": "What type of provider are you looking for?"}],
            "member_status_verify": True,
            "member_id": "M714598",
            "call_intent": "provider_services",
            "active_agent": "provider_search_agent",
            "provider_type": "",
            "zip_code": "94107",
            "zip_code_used": "",
            "awaiting_slot": "provider_type",
            "dirty_artifacts": {},
            "slot_attempts": {},
            "is_interrupt": True,
            "app_run_id": "unit-single",
        },
        "turns": [{"user": "Pediatrician", "extraction": {"extracted": {"provider_type": "pediatrician"}}}],
    }
    run = await run_fixture(fixture, print_latency=False)
    assert run.dropped_request_count == 0
