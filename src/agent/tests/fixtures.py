"""
fixtures.py — Shared state builders and assertion helpers for verification tests.
Do NOT import from test_verification_agent_mock.py or test_verification_agent.py.
"""

from __future__ import annotations

import uuid

VERIFIED_MEMBER = {
    "verified": True,
    "phone_number": "6175554101",
    "zip_code": "12139",
    "relationship": "plan_holder, subscriber",
}


def make_state(**overrides) -> dict:
    """Full base state with all fields needed by VerificationAgent."""
    state: dict = {
        "messages": [],
        "metadata_events": [],
        "is_interrupt": False,
        "next_node": "",
        "app_run_id": str(uuid.uuid4()),
        "slot_attempts": {},
        "call_intent": "provider_services",
        "awaiting_slot": "",
        "active_agent": "",
        "first_name": "",
        "last_name": "",
        "member_id": "",
        "dob": "",
        "relationship": "",
        "member_status_verify": False,
        "previous_agents": [],
        "conversation_context": None,
        "correction_return_to": "",
        "ambiguous_counts": {},
        "last_agent_signal": {},
        "new_intent_detected": "",
        "offtopic_global_count": 0,
        "closure_requested": False,
        "intent_queue": [],
        "orchestrator_reasoning": "",
        "router_loop_count": 0,
        "call_intent": "provider_services",  # noqa: F601
        "ref_no": "",
        "conversation_summary": None,
        "caller_role": "",
        "phone_number": "",
        "zip_code": "",
        "fax": "",
        "email": "",
        "phone_confirmed": False,
        "phone_update_requested": False,
        "escalation_reference_number": "",
        "escalation_reason": "",
        "member_status_verify": False,  # noqa: F601
    }
    state.update(overrides)
    return state


def make_verified_state(**overrides) -> dict:
    """State with member fully verified (all 4 identity slots + member_status_verify=True)."""
    return make_state(
        first_name="Emily",
        last_name="Carter",
        member_id="M907503",
        dob="04/12/1988",
        member_status_verify=True,
        **overrides,
    )


def advance(state: dict, result: dict, user_text: str | None = None) -> dict:
    """Merge result into state, append assistant message, optionally append user message."""
    new_state = {**state}
    for key, val in result.items():
        if key == "messages":
            continue
        new_state[key] = val
    messages = list(state.get("messages") or [])
    if isinstance(result.get("messages"), dict):
        messages.append(result["messages"])
    if user_text is not None:
        messages.append({"role": "user", "content": user_text})
    new_state["messages"] = messages
    return new_state


def is_ask(result: dict) -> bool:
    """Result is waiting for caller input."""
    return result.get("is_interrupt") is True and result.get("next_node") == "verification_agent"


def is_escalation(result: dict) -> bool:
    """Result signals escalation."""
    return result.get("next_node") == "escalation_agent" and result.get("is_interrupt") is False


def is_complete(result: dict) -> bool:
    """Result signals verification complete."""
    return (
        result.get("member_status_verify") is True
        and result.get("is_interrupt") is False
        and result.get("next_node") == "orchestrator"
    )


def get_response(result: dict) -> str:
    """Extract AI response text from result['messages']."""
    msg = result.get("messages", {})
    if isinstance(msg, dict):
        return msg.get("content", "")
    if isinstance(msg, list) and msg:
        last = msg[-1]
        return last.get("content", "") if isinstance(last, dict) else str(last)
    return ""


def get_awaiting(result: dict) -> str:
    return result.get("awaiting_slot", "")


def get_attempt(result: dict, slot: str) -> int:
    """Return attempt_count for slot from result['slot_attempts']."""
    attempts = result.get("slot_attempts") or {}
    slot_state = attempts.get(slot, {})
    return slot_state.get("attempt_count", 0) if isinstance(slot_state, dict) else 0


def get_ambiguous(result: dict, slot: str) -> int:
    """Return ambiguous_counts[slot] from result, or 0."""
    counts = result.get("ambiguous_counts") or {}
    return counts.get(slot, 0)
