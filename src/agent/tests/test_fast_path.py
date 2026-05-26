"""
test_fast_path.py — Unit tests for get_fast_path_route() Phase 1 fast-path routing.

No credentials required. Tests are purely deterministic.

Run all:  pytest src/agent/tests/test_fast_path.py -v
"""

from __future__ import annotations

import pytest

from agent.orchestration.fast_path import get_fast_path_route


def _state(**overrides) -> dict:
    base: dict = {
        "last_agent_signal": {},
        "active_agent": "",
        "member_status_verify": False,
        "call_intent": "",
        "closure_requested": False,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Phase 1 — post-verification routing
# ---------------------------------------------------------------------------


def test_provider_services_routes_to_provider_search() -> None:
    state = _state(
        member_status_verify=True,
        active_agent="verification_agent",
        call_intent="provider_services",
    )
    assert get_fast_path_route(state) == "provider_search_agent"


def test_claim_services_does_not_route_to_provider_search() -> None:
    state = _state(
        member_status_verify=True,
        active_agent="verification_agent",
        call_intent="claim_services",
    )
    result = get_fast_path_route(state)
    assert result != "provider_search_agent"
    assert result == "closure_agent"


def test_unknown_intent_does_not_route_to_provider_search() -> None:
    state = _state(
        member_status_verify=True,
        active_agent="verification_agent",
        call_intent="",
    )
    result = get_fast_path_route(state)
    assert result != "provider_search_agent"


# ---------------------------------------------------------------------------
# Existing fast-path contracts must still hold
# ---------------------------------------------------------------------------


def test_escalation_signal_overrides_everything() -> None:
    state = _state(
        member_status_verify=True,
        active_agent="verification_agent",
        call_intent="provider_services",
        last_agent_signal={"status": "escalate", "closure_requested": False},
    )
    assert get_fast_path_route(state) == "escalation_agent"


def test_blocked_signal_overrides_everything() -> None:
    state = _state(
        member_status_verify=True,
        active_agent="provider_search_agent",
        call_intent="provider_services",
        last_agent_signal={"status": "blocked", "closure_requested": False},
    )
    assert get_fast_path_route(state) == "escalation_agent"


def test_unverified_member_forces_verification() -> None:
    state = _state(
        member_status_verify=False,
        active_agent="orchestrator",
        call_intent="provider_services",
    )
    assert get_fast_path_route(state) == "verification_agent"


def test_no_fast_path_when_already_in_domain_agent() -> None:
    state = _state(
        member_status_verify=True,
        active_agent="provider_search_agent",
        call_intent="provider_services",
        last_agent_signal={"status": "complete", "closure_requested": False},
    )
    # provider_search_agent is not verification_agent, so post-verification
    # branch does NOT fire; no closure_requested so closure branch doesn't
    # fire either → fast path returns None (LLM orchestrator decides).
    assert get_fast_path_route(state) is None


# ---------------------------------------------------------------------------
# End-to-end routing simulation (marker: regression)
# ---------------------------------------------------------------------------


def test_full_provider_flow_routing() -> None:
    """
    Simulate next_node chains through intake → verification → provider_search
    → delivery_management using state manipulation only (no LLM calls).

    Asserts that each transition produces the expected next_node at each step:
      1. After verification completes → fast_path routes to provider_search_agent
      2. While provider_search is running → fast_path returns None
         (graph routes provider_search → delivery_management via next_node override)
      3. While delivery_management is running → fast_path returns None
         (orchestrator LLM handles delivery_management complete → benefits_agent)
    """
    # Step 1: verification just completed for provider_services intent
    state_post_verify = _state(
        member_status_verify=True,
        active_agent="verification_agent",
        call_intent="provider_services",
        last_agent_signal={"status": "complete", "closure_requested": False},
    )
    assert get_fast_path_route(state_post_verify) == "provider_search_agent", (
        "After verification, provider_services must route to provider_search_agent"
    )

    # Step 2: provider_search_agent running (complete, next_node already set to
    # delivery_management_agent by _signal_done's context_updates override)
    # The graph's conditional_routing reads next_node directly — fast_path is None.
    state_ps_complete = _state(
        member_status_verify=True,
        active_agent="provider_search_agent",
        call_intent="provider_services",
        last_agent_signal={"status": "complete", "closure_requested": False},
    )
    assert get_fast_path_route(state_ps_complete) is None, (
        "provider_search complete is graph-routed to delivery_management_agent; "
        "fast_path must not intercept"
    )

    # Step 3: delivery_management_agent completed — orchestrator LLM decides
    # (proactive_offer_available=True → benefits_agent via Priority 5 orchestrator rule)
    state_dm_complete = _state(
        member_status_verify=True,
        active_agent="delivery_management_agent",
        call_intent="provider_services",
        last_agent_signal={"status": "complete", "closure_requested": False},
        proactive_offer_available=True,
    )
    assert get_fast_path_route(state_dm_complete) is None, (
        "delivery_management complete must fall through to LLM orchestrator (Priority 5)"
    )
