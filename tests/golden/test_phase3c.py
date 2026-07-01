"""
test_phase3c.py — Phase 3C: multi-intent acknowledgement + open-redirect.

Two things are proven here:
  1. The acknowledgement is TEMPLATE-FIRST and reliably covers the UAT-007
     combinatorics at the resolver→render level — so no model call is needed on
     the common path.
  2. The full UAT-007 ZIP detour runs end-to-end across agents: every distinct
     request gets a spoken outcome, nothing is silently dropped, the provider
     list is never delivered on the disputed ZIP (it is delivered on the
     re-resolved ZIP), and per-turn latency stays within a deterministic budget.
"""

from __future__ import annotations

import pytest

import tests.golden  # noqa: F401 — ensures src/ is on sys.path
from agent.llm.schema import (
    Correction,
    SecondaryIntent,
    SecondaryIntentType,
    TurnPlan,
)
from agent.orchestration.resolver import resolve_turn
from agent.orchestration.shadow import heuristic_decoder
from agent.responses import turn_acts
from tests.golden.driver import run_conversation

pytestmark = pytest.mark.regression


# ──────────────────────────────────────────────────────────────────────────────
# Template-first: render the resolver's multi-intent outcome (no model call)
# ──────────────────────────────────────────────────────────────────────────────


def _render(outcome) -> str:
    """Render a resolver outcome through the closed-set templates (what the live
    path does). Returns the spoken text for assertion."""
    if outcome.speech_act == "multi_intent_ack":
        return turn_acts.render_multi_intent_ack(outcome.parked)
    if outcome.speech_act == "correction_ack":
        return turn_acts.render_correction_ack(field="zip_code", slot_value="fax")
    if outcome.speech_act == "unsupported_decline":
        return turn_acts.render_unsupported_decline()
    if outcome.speech_act == "open_redirect":
        return turn_acts.render_open_redirect()
    return ""


def test_multi_intent_ack_template_covers_fax_redirect_plus_benefits_later():
    """UAT-007 combinatoric: 'send it to another fax number' + 'benefits later'."""
    state = {"awaiting_slot": "benefits_response", "dirty_artifacts": {}, "intent_queue": []}
    plan = TurnPlan(
        secondary_intents=[
            SecondaryIntent(
                type=SecondaryIntentType.IN_SCOPE_INDEPENDENT,
                owner="delivery_management_agent",
                verbatim_span="another fax number",
            ),
            SecondaryIntent(
                type=SecondaryIntentType.IN_SCOPE_INDEPENDENT,
                owner="benefits_agent",
                verbatim_span="benefits later",
            ),
        ]
    )
    out = resolve_turn(plan, state, utterance="send it to another fax number, benefits later")
    assert out.speech_act == "multi_intent_ack"
    assert out.parked == ["delivery_management_agent", "benefits_agent"]
    msg = _render(out)
    # Both parked intents named; no unfilled placeholders.
    assert "delivery details" in msg
    assert "benefits question" in msg
    assert "{" not in msg and "}" not in msg


def test_multi_intent_ack_template_covers_all_redirect_phrasings():
    """Every distinct fax-redirect phrasing in the UAT-007 log decodes + renders
    to a spoken multi-intent acknowledgement (reliable → keep templates)."""
    redirects = [
        "Oh, by the way, can you send it to another fax number?",
        "Later. But can you send the list to another fax number?",
        "Yes. I do. But I'm sorry. Can you send the list to a different fax number?",
        "Before that, can you send the list of the providers on a different fax number, please?",
    ]
    state = {"awaiting_slot": "benefits_response", "dirty_artifacts": {}, "intent_queue": []}
    for utterance in redirects:
        plan = heuristic_decoder(state, utterance, None)
        assert plan is not None, f"no plan for {utterance!r}"
        out = resolve_turn(plan, state, utterance=utterance)
        assert out.speech_act == "multi_intent_ack", f"{utterance!r} → {out.speech_act}"
        msg = _render(out)
        assert msg.strip() and "{" not in msg, f"bad render for {utterance!r}: {msg!r}"


def test_correction_ack_template_covers_invalidating_zip():
    state = {"awaiting_slot": "delivery_method", "dirty_artifacts": {}, "intent_queue": []}
    plan = TurnPlan(
        slot_answer="fax",
        correction=Correction(field="zip_code", owner="provider_search_agent"),
    )
    out = resolve_turn(plan, state, utterance="Fax, but I need to update my ZIP code.")
    assert out.speech_act == "correction_ack"
    msg = _render(out)
    assert "fax" in msg and "ZIP code" in msg


def test_open_redirect_template_is_ask_only():
    msg = turn_acts.render_open_redirect()
    assert msg.strip()
    # Ask/redirect only — never asserts an action was taken.
    assert "?" in msg


def test_out_of_scope_resolves_to_spoken_decline_or_redirect():
    """An unanswerable side-question gets a spoken outcome (decline / redirect),
    never a silent drop, and the resolver never acts on it."""
    # Slot answered + out-of-scope side question → decline (spoken), slot kept.
    state = {"awaiting_slot": "fax_confirmed", "dirty_artifacts": {}, "intent_queue": []}
    plan = TurnPlan(
        slot_answer="yes",
        secondary_intents=[
            SecondaryIntent(type=SecondaryIntentType.OUT_OF_SCOPE, owner=None, verbatim_span="weather")
        ],
    )
    out = resolve_turn(plan, state, utterance="yes and what's the weather")
    assert out.speech_act == "unsupported_decline"
    assert _render(out).strip()

    # No slot answer, only an out-of-scope question → a spoken, non-acting
    # outcome (decline or ask-only redirect). Never acts.
    state2 = {"awaiting_slot": "", "dirty_artifacts": {}, "intent_queue": []}
    plan2 = TurnPlan(
        secondary_intents=[
            SecondaryIntent(type=SecondaryIntentType.OUT_OF_SCOPE, owner=None, verbatim_span="weather")
        ]
    )
    out2 = resolve_turn(plan2, state2, utterance="what's the weather")
    assert out2.speech_act in ("unsupported_decline", "open_redirect")
    assert out2.state_updates == {}  # never act
    assert _render(out2).strip()


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end UAT-007 ZIP detour across agents
# ──────────────────────────────────────────────────────────────────────────────


def _uat007_initial_state() -> dict:
    return {
        "messages": [
            {"role": "assistant", "content": "Your in-network provider list is ready. Fax or email?"}
        ],
        "member_status_verify": True,
        "member_id": "M714598",
        "first_name": "Daniel",
        "call_intent": "provider_services",
        "active_agent": "delivery_management_agent",
        "next_node": "delivery_management_agent",
        "provider_type": "Pediatrician",
        "zip_code": "94107",
        "zip_code_used": "94107",
        "fax": "415-555-3299",
        "email": "",
        "delivery_method": "",
        "awaiting_slot": "",
        "dirty_artifacts": {},
        "intent_queue": [],
        "slot_attempts": {},
        "is_interrupt": True,
        "app_run_id": "e2e-uat-007",
    }


async def test_uat_007_zip_detour_end_to_end():
    turns = [
        # 1) member answers fax AND disputes the ZIP → ack both, route to ZIP owner
        {
            "agent": "delivery_management_agent",
            "user": "Fax, but I need to update my ZIP code.",
            "extraction": {"extracted": {"delivery_method": "fax"}},
        },
        # 2) provider_search collects the corrected ZIP → SF update, dirty cleared
        {
            "agent": "provider_search_agent",
            "user": "It's 94110.",
            "extraction": {"extracted": {"zip_code": "94110"}},
        },
        # 3) back at delivery (bridge) — fax already chosen; confirm contact
        {
            "agent": "delivery_management_agent",
            "user": "fax",
            "extraction": {"extracted": {"delivery_method": "fax"}},
        },
        # 4) confirm the fax → dispatch on the RE-RESOLVED ZIP
        {
            "agent": "delivery_management_agent",
            "user": "yes",
            "extraction": {"extracted": {"fax_confirmed": "yes"}},
        },
        # 5) decline the benefits offer → done
        {
            "agent": "delivery_management_agent",
            "user": "no thanks",
            "extraction": {"extracted": {"benefits_response": "no"}},
        },
    ]
    run = await run_conversation(_uat007_initial_state(), turns, fixture_id="UAT-007-E2E")

    # Every distinct member turn produced a spoken outcome — nothing silently dropped.
    assert all(t.ai.strip() for t in run.turns), [t.ai for t in run.turns]

    # Turn 0 acknowledged BOTH the fax and the ZIP-update.
    assert "fax" in run.turns[0].ai.lower() and "zip" in run.turns[0].ai.lower()

    # The ZIP was re-resolved before any delivery.
    zip_updates = run.recorder.for_tool("update_zip_code")
    assert [c["zip_code"] for c in zip_updates] == ["94110"]

    # The list was delivered exactly once, on the RE-RESOLVED ZIP — never the disputed one.
    dispatches = run.recorder.for_tool("dispatch_provider_list")
    assert len(dispatches) == 1
    assert dispatches[0]["zip_code"] == "94110"
    assert all(d["zip_code"] != "94107" for d in dispatches)

    # No silent drop recorded, and per-turn latency within a deterministic budget.
    assert run.dropped_request_count == 0
    assert all(ms < 250 for ms in run.latencies_ms)
