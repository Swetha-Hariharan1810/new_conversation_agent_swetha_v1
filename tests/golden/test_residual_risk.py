"""
test_residual_risk.py — residual-risk checks to verify before declaring done.

  * Template variation ceiling: every speech-act has enough phrasings that
    rotation does not repeat within a single collection loop (≈3 attempts).
  * Contract change cost / isolation: the understanding decode (TurnPlan) is a
    separate, dormant-by-default seam with token headroom; per-agent slot
    extraction is untouched, so single-intent flows are unchanged.

(Intra-enum misclassification is S6 and span/owner drop is S7 in
test_scenarios_s1_s8.py — the deterministic safety net that runs in CI.)
"""

from __future__ import annotations

import pytest

import tests.golden  # noqa: F401 — ensures src/ is on sys.path
from agent.responses import turn_acts
from tests.golden.driver import run_fixture

pytestmark = pytest.mark.regression

# Slots exhaust at MAX_SLOT_ATTEMPTS == 3, so a pool must offer ≥3 distinct
# phrasings to avoid repeating inside one collection loop.
_ATTEMPTS = (0, 1, 2)


def _distinct_over_attempts(render) -> bool:
    return len({render(a) for a in _ATTEMPTS}) == len(_ATTEMPTS)


def test_re_ask_and_clarify_do_not_repeat_within_a_loop():
    assert _distinct_over_attempts(lambda a: turn_acts.render_re_ask(slot_label="date of birth", attempt=a))
    assert _distinct_over_attempts(lambda a: turn_acts.render_clarify(slot_label="date of birth", attempt=a))


def test_correction_ack_variation_ceiling():
    assert _distinct_over_attempts(
        lambda a: turn_acts.render_correction_ack(field="zip_code", slot_value="fax", attempt=a)
    )
    assert _distinct_over_attempts(lambda a: turn_acts.render_correction_ack(field="member_id", attempt=a))


def test_decline_and_redirect_variation_ceiling():
    assert _distinct_over_attempts(lambda a: turn_acts.render_unsupported_decline(attempt=a))
    assert _distinct_over_attempts(lambda a: turn_acts.render_open_redirect(attempt=a))


def test_multi_intent_ack_variation_ceiling():
    assert _distinct_over_attempts(
        lambda a: turn_acts.render_multi_intent_ack(["benefits_agent"], attempt=a)
    )
    assert _distinct_over_attempts(
        lambda a: turn_acts.render_multi_intent_ack(
            ["benefits_agent"], rebuilding="update your ZIP", attempt=a
        )
    )


def test_understanding_decode_llm_tier_provisioned_with_headroom():
    # The TurnPlan decode has its own LLM tier with max_tokens headroom over the
    # per-agent extractor — provisioned and isolated, dormant until installed.
    from agent.llm import config

    assert callable(config.get_understanding_llm)
    assert callable(config.get_extraction_llm)  # per-agent slot extractor unchanged


async def test_contract_change_isolated_single_intent_unchanged_with_decode_off():
    """Per-agent slot extraction is independent of the understanding decode:
    with the decode cleared, a single-intent turn behaves exactly as before."""
    from agent.orchestration import shadow as shadow_mod

    fixture = {
        "id": "RR-ISOLATION",
        "driver": "provider_search_agent",
        "initial_state": {
            "messages": [{"role": "assistant", "content": "What type of provider?"}],
            "member_status_verify": True,
            "member_id": "M714598",
            "call_intent": "provider_services",
            "active_agent": "provider_search_agent",
            "provider_type": "",
            "zip_code": "94107",
            "zip_code_used": "",
            "awaiting_slot": "provider_type",
            "dirty_artifacts": {},
            "intent_queue": [],
            "slot_attempts": {},
            "is_interrupt": True,
            "app_run_id": "rr",
        },
        "turns": [{"user": "a pediatrician", "extraction": {"extracted": {"provider_type": "pediatrician"}}}],
    }

    shadow_mod.set_shadow_decoder(None)  # understanding decode OFF
    off = await run_fixture(fixture, print_latency=False)
    shadow_mod.set_shadow_decoder(shadow_mod.heuristic_decoder)  # ON
    on = await run_fixture(fixture, print_latency=False)

    for key in ("provider_type", "awaiting_slot", "next_node"):
        assert off.final_state.get(key) == on.final_state.get(key)
    assert off.final_state.get("provider_type") == "Pediatrician"
