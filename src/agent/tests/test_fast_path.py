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
