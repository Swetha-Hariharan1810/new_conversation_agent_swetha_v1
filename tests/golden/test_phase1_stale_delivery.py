"""
test_phase1_stale_delivery.py — focused tests for the Phase 1 deterministic
stale-delivery guard: the invalidation registry, the delivery gate, and the
provider_search mark/clear wiring. Hermetic (no LLM, no network); the agent
tests reuse the golden driver's deterministic fakes.
"""

from __future__ import annotations

import re

import pytest

import tests.golden  # noqa: F401 — ensures src/ is on sys.path
from tests.golden.driver import run_fixture

pytestmark = pytest.mark.regression


# ──────────────────────────────────────────────────────────────────────────────
# Pure registry (zero model cost, no I/O)
# ──────────────────────────────────────────────────────────────────────────────


def test_artifacts_invalidated_by():
    from agent.orchestration.invalidation import artifacts_invalidated_by

    assert artifacts_invalidated_by("zip_code") == ["provider_list"]
    assert artifacts_invalidated_by("unknown_field") == []
    # Returns a fresh list — mutating it must not corrupt the registry.
    got = artifacts_invalidated_by("zip_code")
    got.append("garbage")
    assert artifacts_invalidated_by("zip_code") == ["provider_list"]


def test_owner_registry():
    from agent.orchestration.invalidation import owner_of

    assert owner_of("zip_code") == "provider_search_agent"
    assert owner_of("provider_list") == "delivery_management_agent"
    assert owner_of("nope") is None


def test_mark_clear_is_dirty_are_pure():
    from agent.orchestration.invalidation import clear_dirty, is_dirty, mark_dirty

    base: dict = {}
    marked = mark_dirty(base, "zip_code")
    assert marked == {"provider_list": True}
    assert base == {}  # input not mutated
    assert is_dirty(marked, "provider_list") is True
    assert is_dirty(None, "provider_list") is False
    assert is_dirty({}, "provider_list") is False

    cleared = clear_dirty(marked, "provider_list")
    assert cleared == {"provider_list": False}
    assert is_dirty(cleared, "provider_list") is False
    assert marked == {"provider_list": True}  # input not mutated

    # Marking an upstream field with no downstream artifacts is a no-op copy.
    assert mark_dirty({"x": True}, "unknown") == {"x": True}


# ──────────────────────────────────────────────────────────────────────────────
# Delivery gate — reads ONLY dirty_artifacts, unconditional w.r.t. classification
# ──────────────────────────────────────────────────────────────────────────────


def _delivery_state(*, dirty: bool) -> dict:
    fax_readback = "The fax number we have on file is 415-555-3299. Is this correct?"
    return {
        "messages": [{"role": "assistant", "content": fax_readback}],
        "member_status_verify": True,
        "member_id": "M714598",
        "first_name": "Daniel",
        "call_intent": "provider_services",
        "active_agent": "delivery_management_agent",
        "provider_type": "Pediatrician",
        "zip_code": "94107",
        "zip_code_used": "94107",
        "fax": "415-555-3299",
        "email": "",
        "delivery_method": "fax",
        "awaiting_slot": "fax_confirmed",
        "dirty_artifacts": {"provider_list": True} if dirty else {},
        "slot_attempts": {},
        "is_interrupt": True,
        "app_run_id": "unit-gate",
    }


async def test_gate_blocks_dispatch_when_provider_list_dirty():
    fixture = {
        "id": "UNIT-GATE-BLOCK",
        "driver": "delivery_management_agent",
        "initial_state": _delivery_state(dirty=True),
        "turns": [{"user": "Yes.", "extraction": {"extracted": {"fax_confirmed": "yes"}}}],
    }
    run = await run_fixture(fixture, print_latency=False)

    assert run.recorder.count("dispatch_provider_list") == 0
    assert run.final_state.get("next_node") == "provider_search_agent"
    assert run.final_state.get("awaiting_slot") == "zip_code"
    assert run.final_state.get("zip_code_used") == ""  # forces re-resolution
    assert re.search(r"zip", run.last_ai(), re.IGNORECASE)
    assert not run.final_state.get("provider_list_sent")


async def test_gate_allows_dispatch_when_clean():
    fixture = {
        "id": "UNIT-GATE-ALLOW",
        "driver": "delivery_management_agent",
        "initial_state": _delivery_state(dirty=False),
        "turns": [{"user": "Yes.", "extraction": {"extracted": {"fax_confirmed": "yes"}}}],
    }
    run = await run_fixture(fixture, print_latency=False)

    dispatches = run.recorder.for_tool("dispatch_provider_list")
    assert len(dispatches) == 1
    assert dispatches[0]["zip_code"] == "94107"
    assert dispatches[0]["delivery_method"] == "fax"
    assert run.final_state.get("provider_list_sent") is True
    assert run.final_state.get("awaiting_slot") == "benefits_response"


# ──────────────────────────────────────────────────────────────────────────────
# provider_search marking / clearing
# ──────────────────────────────────────────────────────────────────────────────


def _provider_zip_confirm_state(*, dirty: bool = False) -> dict:
    return {
        "messages": [{"role": "assistant", "content": "I have your ZIP code as 94107. Is that right?"}],
        "member_status_verify": True,
        "member_id": "M714598",
        "first_name": "Daniel",
        "call_intent": "provider_services",
        "active_agent": "provider_search_agent",
        "provider_type": "Pediatrician",
        "zip_code": "94107",
        "zip_code_used": "",
        "awaiting_slot": "zip_confirmed",
        "dirty_artifacts": {"provider_list": True} if dirty else {},
        "slot_attempts": {},
        "is_interrupt": True,
        "app_run_id": "unit-provider",
    }


async def test_provider_search_marks_dirty_on_zip_decline():
    fixture = {
        "id": "UNIT-PROVIDER-MARK",
        "driver": "provider_search_agent",
        "initial_state": _provider_zip_confirm_state(),
        "turns": [{"user": "No, that's not right.", "extraction": {"extracted": {"zip_confirmed": "no"}}}],
    }
    run = await run_fixture(fixture, print_latency=False)

    assert run.final_state.get("dirty_artifacts", {}).get("provider_list") is True
    assert run.final_state.get("awaiting_slot") == "zip_code"


async def test_provider_search_clears_dirty_on_zip_confirm():
    fixture = {
        "id": "UNIT-PROVIDER-CLEAR",
        "driver": "provider_search_agent",
        "initial_state": _provider_zip_confirm_state(dirty=True),
        "turns": [{"user": "Yes, that's correct.", "extraction": {"extracted": {"zip_confirmed": "yes"}}}],
    }
    run = await run_fixture(fixture, print_latency=False)

    assert run.final_state.get("dirty_artifacts", {}).get("provider_list") is False
    assert run.final_state.get("next_node") == "delivery_management_agent"


# ──────────────────────────────────────────────────────────────────────────────
# reset_for_new_intent zeroes the registry
# ──────────────────────────────────────────────────────────────────────────────


def test_reset_for_new_intent_zeroes_dirty_artifacts():
    from agent.state import reset_for_new_intent

    updates = reset_for_new_intent({"dirty_artifacts": {"provider_list": True}}, "claim_services")
    assert updates["dirty_artifacts"] == {}
