"""
fast_path.py — Deterministic routing (core agents only).
"""

from __future__ import annotations

from agent.core.signal import AgentSignal, AgentStatus
from agent.state import State


def get_fast_path_route(state: State) -> str | None:
    signal = AgentSignal.from_state_dict(state.get("last_agent_signal", {}))
    active_agent = state.get("active_agent")
    member_verified = state.get("member_status_verify", False)

    # Hard escalation
    if signal.status in {AgentStatus.ESCALATE, AgentStatus.BLOCKED}:
        return "escalation_agent"

    # Verification gate
    if not member_verified and active_agent != "verification_agent":
        return "verification_agent"

    # delivery_management complete → benefits (always)
    if (
        member_verified
        and active_agent == "delivery_management_agent"
        and signal.status == AgentStatus.COMPLETE
    ):
        return "benefits_agent"

    # benefits complete → care_wellness (always)
    if member_verified and active_agent == "benefits_agent" and signal.status == AgentStatus.COMPLETE:
        return "care_wellness_agent"

    # care_wellness complete → follow_up (always)
    if member_verified and active_agent == "care_wellness_agent" and signal.status == AgentStatus.COMPLETE:
        return "follow_up_agent"

    # Follow-up agent complete + closure requested → closure_agent
    if (
        member_verified
        and active_agent == "follow_up_agent"
        and signal.status == AgentStatus.COMPLETE
        and signal.closure_requested
    ):
        return "closure_agent"

    # Routing fix: follow_up_agent fast-path
    # follow_up_agent complete without closure → stay in follow_up_agent.
    # This should not normally occur (ask_member bypasses orchestrator), but
    # if signal_complete(closure_requested=False) fires for any reason,
    # routing to the LLM orchestrator risks an intake-style re-routing decision.
    if (
        member_verified
        and active_agent == "follow_up_agent"
        and signal.status == AgentStatus.COMPLETE
        and not signal.closure_requested
    ):
        return "follow_up_agent"

    # General closure signal — never re-route if already inside closure_agent
    if signal.status == AgentStatus.COMPLETE and signal.closure_requested:
        if active_agent != "closure_agent":
            return "closure_agent"
        return None  # closure_agent set next_node=END; let conditional_routing handle it

    # After verification — route to the correct domain agent
    if member_verified and active_agent == "verification_agent":
        intent = state.get("call_intent", "")
        if intent == "provider_services":
            return "provider_search_agent"
        # No other domain agents implemented yet → closure
        return "closure_agent"

    return None
