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

    # Closure
    if signal.status == AgentStatus.COMPLETE and signal.closure_requested:
        return "closure_agent"

    # After verification — route to the correct domain agent
    if member_verified and active_agent == "verification_agent":
        intent = state.get("call_intent", "")
        if intent == "provider_services":
            return "provider_search_agent"
        # No other domain agents implemented yet → closure
        return "closure_agent"

    return None
