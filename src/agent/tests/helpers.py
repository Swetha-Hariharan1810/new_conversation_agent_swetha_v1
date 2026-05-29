"""
tests/helpers.py — Shared test utilities for live integration tests.

Consolidates helpers previously copy-pasted between test_verification_agent.py
and test_intake_agent.py.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path


def load_test_env() -> None:
    """Walk up from this file and from cwd looking for .env, load it once."""
    try:
        candidates = [
            Path(__file__).parents[3] / ".env",  # project root: src/agent/tests/ → 3 up
            Path(__file__).parents[2] / ".env",  # one level up fallback
            Path.cwd() / ".env",
        ]
        for dot in candidates:
            if not dot.exists():
                continue
            for line in dot.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
            break
    except Exception:
        pass


def make_base_state(**overrides) -> dict:
    """Return a minimal LangGraph state dict suitable for both agents."""
    state: dict = {
        "messages": [],
        "metadata_events": [],
        "is_interrupt": False,
        "next_node": "",
        "app_run_id": str(uuid.uuid4()),
        "slot_attempts": {},
        "call_intent": "",
        "conversation_summary": None,
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
    }
    state.update(overrides)
    return state


_ESCALATION_NODE = "escalation_agent"
_VERIFICATION_NODE = "verification_agent"
_ORCHESTRATOR_NODE = "orchestrator"


def is_escalation(result: dict) -> bool:
    """True when the agent signalled escalation (non-interrupt transition to escalation_agent)."""
    return result.get("next_node") == _ESCALATION_NODE and result.get("is_interrupt") is False


def is_ask_member(result: dict) -> bool:
    """True when the agent is waiting for caller input (interrupt on the same node)."""
    return result.get("next_node") == _VERIFICATION_NODE and result.get("is_interrupt") is True


def is_complete(result: dict) -> bool:
    """True when verification completed successfully and member is verified."""
    return (
        result.get("next_node") == _ORCHESTRATOR_NODE
        and result.get("is_interrupt") is False
        and result.get("member_status_verify") is True
    )
