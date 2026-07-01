"""
test_phase3a_shadow.py — shadow-mode integration (Phase 3A).

Drives the real agents through the golden fixtures with the TurnPlan understanding
decode + resolver installed at the shared _collect_slot chokepoint. The shadow
only LOGS — these tests prove (a) the single resolver catches the multi-intent
turns regardless of which agent is active, and (b) the live path is unchanged.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

import pytest

import tests.golden  # noqa: F401 — ensures src/ is on sys.path
from agent.orchestration.shadow import clear_shadow_decoder, get_shadow_decoder, heuristic_decoder
from tests.golden.driver import load_fixture, run_fixture

pytestmark = pytest.mark.regression


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@contextmanager
def shadow_logs(decoder=heuristic_decoder):
    """Install the decoder and capture its log records; restore prior decoder."""
    from agent.orchestration import shadow as shadow_mod

    lg = logging.getLogger("agent.orchestration.shadow")
    handler = _ListHandler()
    handler.setLevel(logging.DEBUG)
    prev_level = lg.level
    prev_decoder = shadow_mod.get_shadow_decoder()
    lg.setLevel(logging.DEBUG)
    lg.addHandler(handler)
    shadow_mod.set_shadow_decoder(decoder)
    try:
        yield handler.records
    finally:
        shadow_mod.set_shadow_decoder(prev_decoder)
        lg.removeHandler(handler)
        lg.setLevel(prev_level)


def _shadow_events(records):
    return [r for r in records if getattr(r, "metric", None) == "turnplan_shadow"]


# ── promoted by default + kill switch ────────────────────────────────────────


async def test_understanding_decode_installed_by_default():
    # Phase 3B promotes the deterministic decode: installed by default (the
    # conftest autouse fixture pins it, mirroring import-time default).
    assert get_shadow_decoder() is heuristic_decoder


async def test_decoder_kill_switch_disables_live_and_shadow():
    # Clearing the decoder reverts to pre-3B behavior: no shadow log AND no live
    # invalidating-correction handling (the slot is collected and the agent asks
    # to confirm the fax, never routing to the ZIP owner).
    clear_shadow_decoder()
    assert get_shadow_decoder() is None

    handler = _ListHandler()
    lg = logging.getLogger("agent.orchestration.shadow")
    lg.addHandler(handler)
    try:
        run = await run_fixture(load_fixture("uat_007_multi_intent"), print_latency=False)
    finally:
        lg.removeHandler(handler)

    assert _shadow_events(handler.records) == []
    # Live invalidating handling is OFF → old single-intent collection path.
    assert run.turns[0].awaiting_slot == "fax_confirmed"
    assert run.final_state.get("next_node") != "provider_search_agent"


# ── catches the UAT-007 ZIP request at the delivery chokepoint ───────────────


async def test_shadow_catches_zip_request_on_uat007():
    fixture = load_fixture("uat_007_multi_intent")
    with shadow_logs() as records:
        run = await run_fixture(fixture, print_latency=False)

    events = _shadow_events(records)
    assert events, "expected a shadow resolver decision on the delivery_method turn"
    # The resolver recovered the dropped ZIP request: correction_ack + provider_list dirty.
    zip_catch = [
        e for e in events if getattr(e, "speech_act", None) == "correction_ack"
    ]
    assert zip_catch, f"resolver did not catch the ZIP request: {[vars(e) for e in events]}"
    assert any("provider_list" in (getattr(e, "dirty", []) or []) for e in zip_catch)
    assert any(getattr(e, "rewind_target", None) == "provider_search_agent" for e in zip_catch)

    # LIVE behavior unchanged vs Phase 1/2: still no dispatch on the disputed ZIP.
    assert run.recorder.count("dispatch_provider_list") == 0
    assert run.final_state.get("next_node") == "provider_search_agent"


# ── catches an in-scope independent at the provider_search chokepoint ─────────


async def test_shadow_catches_independent_on_provider_search():
    fixture = load_fixture("slot_interrupt_fresh_request")
    with shadow_logs() as records:
        run = await run_fixture(fixture, print_latency=False)

    events = _shadow_events(records)
    assert events, "expected a shadow resolver decision on the provider_type turn"
    acks = [e for e in events if getattr(e, "speech_act", None) == "multi_intent_ack"]
    assert acks, f"resolver did not multi-intent-ack: {[vars(e) for e in events]}"
    assert any("benefits_agent" in (getattr(e, "parked", []) or []) for e in acks)

    # Phase 3C: the independent is now acted live — acknowledged and parked.
    assert run.final_state.get("provider_type") == "Pediatrician"
    assert "benefits_agent" in (run.final_state.get("intent_queue") or [])


# ── shadow does not change behavior anywhere ─────────────────────────────────


def _clean_single_intent_fixture():
    return {
        "id": "UNIT-CLEAN-SINGLE",
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
            "app_run_id": "unit-clean",
        },
        "turns": [{"user": "Pediatrician", "extraction": {"extracted": {"provider_type": "pediatrician"}}}],
    }


async def test_non_invalidating_turn_unchanged_by_decode():
    """Only the invalidating-correction case is promoted live in 3B. A clean
    single-intent turn must behave identically with the decode ON vs OFF."""
    from agent.orchestration import shadow as shadow_mod

    shadow_mod.set_shadow_decoder(None)
    off = await run_fixture(_clean_single_intent_fixture(), print_latency=False)
    shadow_mod.set_shadow_decoder(heuristic_decoder)
    on = await run_fixture(_clean_single_intent_fixture(), print_latency=False)

    for key in ("next_node", "awaiting_slot", "provider_type", "zip_code"):
        assert off.final_state.get(key) == on.final_state.get(key), f"decode changed {key}"


# ── resolver would catch the later 'send it to another fax' redirects ────────


def test_resolver_catches_redirect_requests():
    """The UAT-007 fax-redirect utterances resolve to an actionable plan
    (not a silent drop) — demonstrating coverage of the later turns even though
    they arrive on inline (non-_collect_slot) branches today."""
    from agent.orchestration.resolver import resolve_turn

    redirects = [
        "Oh, by the way, can you send it to another fax number?",
        "Later. But can you send the list to another fax number?",
        "Before that, can you send the list of the providers on a different fax number, please?",
    ]
    state = {"awaiting_slot": "benefits_response", "dirty_artifacts": {}, "intent_queue": []}
    for utterance in redirects:
        plan = heuristic_decoder(state, utterance, None)
        assert plan is not None, f"decoder produced no plan for {utterance!r}"
        out = resolve_turn(plan, state, utterance=utterance)
        # Caught as an actionable multi-intent ack routed to the contact owner.
        assert out.speech_act == "multi_intent_ack", f"{utterance!r} → {out.speech_act}"
        assert "delivery_management_agent" in out.parked
