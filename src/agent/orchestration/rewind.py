"""
rewind.py — deterministic routing for pending intents.

get_rewind_route sends control back to the agent that owns a disputed value so
it can be corrected before anything is delivered. get_drain_route sends control
to the agent that handles a deferred independent request. Both are pure.
"""

from __future__ import annotations

from typing import Optional

from agent.core.pending_intents import (
    CORRECTION_OWNER,
    INTENT_AGENT,
    next_open_correction,
    next_open_independent,
)
from agent.state import State


def get_rewind_route(state: State) -> Optional[str]:
    pending = list(state.get("pending_intents") or [])
    item = next_open_correction(pending)
    if not item:
        return None
    return CORRECTION_OWNER.get(item.get("target") or "")


def get_drain_route(state: State) -> Optional[str]:
    pending = list(state.get("pending_intents") or [])
    item = next_open_independent(pending)
    if not item:
        return None
    return INTENT_AGENT.get((item.get("raw_text") or "").strip())
