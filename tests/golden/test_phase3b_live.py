"""
test_phase3b_live.py — Phase 3B: closed-set templates + the invalidating-correction
path promoted live, with single-intent regression coverage.
"""

from __future__ import annotations

import pytest

import tests.golden  # noqa: F401 — ensures src/ is on sys.path
from agent.responses import turn_acts
from tests.golden.driver import run_fixture

pytestmark = pytest.mark.regression


# ── closed-set templates (zero generative surface) ───────────────────────────


def test_correction_ack_with_slot_mentions_both_and_asks():
    msg = turn_acts.render_correction_ack(field="zip_code", slot_value="fax", attempt=0)
    assert "fax" in msg  # the slot answer is acknowledged
    assert "ZIP code" in msg  # the corrected field is acknowledged
    assert "current 5-digit ZIP code" in msg  # and we ask for the new value
    assert "{" not in msg and "}" not in msg  # no unfilled placeholders


def test_correction_ack_fix_only_when_no_slot_value():
    msg = turn_acts.render_correction_ack(field="zip_code", attempt=0)
    assert "ZIP code" in msg
    assert "current 5-digit ZIP code" in msg


def test_template_rotation_is_deterministic_by_attempt():
    pool_len = len(turn_acts._CORRECTION_ACK_WITH_SLOT)
    a0 = turn_acts.render_correction_ack(field="zip_code", slot_value="fax", attempt=0)
    a1 = turn_acts.render_correction_ack(field="zip_code", slot_value="fax", attempt=1)
    a_wrap = turn_acts.render_correction_ack(field="zip_code", slot_value="fax", attempt=pool_len)
    assert a0 != a1  # rotates with attempt
    assert a0 == a_wrap  # wraps modulo pool length (deterministic)


def test_re_ask_and_clarify_use_slot_label():
    assert "date of birth" in turn_acts.render_re_ask(slot_label="date of birth")
    assert "date of birth" in turn_acts.render_clarify(slot_label="date of birth")
    assert "{" not in turn_acts.render_re_ask(slot_label="x")


def test_unsupported_decline_renders_non_empty():
    assert turn_acts.render_unsupported_decline(attempt=0).strip()
    assert turn_acts.render_unsupported_decline(attempt=1).strip()


# ── live invalidating-correction (UAT-007) end to end ────────────────────────


async def test_uat_007_invalidating_correction_live():
    from tests.golden.driver import load_fixture

    run = await run_fixture(load_fixture("uat_007_multi_intent"))
    turn0 = run.turns[0]

    # Both intents acknowledged; routed to re-resolve the ZIP; nothing dispatched.
    assert "fax" in turn0.ai.lower() and "zip" in turn0.ai.lower()
    assert run.final_state.get("delivery_method") == "fax"
    assert run.final_state.get("next_node") == "provider_search_agent"
    assert turn0.awaiting_slot == "zip_code"
    assert run.final_state.get("dirty_artifacts", {}).get("provider_list") is True
    assert run.recorder.count("dispatch_provider_list") == 0
    assert run.dropped_request_count == 0
    # One understanding decode → per-turn latency well within a deterministic budget.
    assert run.latencies_ms[0] < 250


# ── single-intent regression: ANSWERED_WITH_FOLLOWUP still works ─────────────


async def test_answered_with_followup_single_intent_preserved():
    """A benign 'answer + thanks' turn (no secondary, no invalidating correction)
    must keep the existing ANSWERED_WITH_FOLLOWUP behavior: slot confirmed,
    awaiting cleared, NOT rerouted by the resolver."""
    fixture = {
        "id": "UNIT-AWF-SINGLE",
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
            "app_run_id": "unit-awf",
        },
        "turns": [
            {
                "user": "Pediatrician, thanks so much for the help.",
                "extraction": {
                    "extracted": {"provider_type": "pediatrician"},
                    "event_type": "answered_with_followup",
                },
            }
        ],
    }
    run = await run_fixture(fixture, print_latency=False)

    # Slot captured; existing follow-up path ran (awaiting cleared, stays in agent).
    assert run.final_state.get("provider_type") == "Pediatrician"
    assert run.turns[0].awaiting_slot == ""
    assert run.final_state.get("next_node") == "provider_search_agent"
    # Not rerouted as an invalidating correction, nothing marked dirty.
    assert run.final_state.get("dirty_artifacts", {}).get("provider_list") in (None, False)


async def test_clean_single_intent_unaffected():
    """A plain single-answer turn confirms and proceeds exactly as before."""
    fixture = {
        "id": "UNIT-CLEAN",
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
            "app_run_id": "unit-clean2",
        },
        "turns": [{"user": "Pediatrician", "extraction": {"extracted": {"provider_type": "pediatrician"}}}],
    }
    run = await run_fixture(fixture, print_latency=False)
    assert run.final_state.get("provider_type") == "Pediatrician"
    assert run.turns[0].awaiting_slot == "zip_confirmed"  # advanced normally
    assert run.recorder.count("dispatch_provider_list") == 0
