"""
telemetry.py — visibility for context retention.

count_open_in_scope counts requests that still need an outcome.
log_unaddressed_request emits a structured line a dashboard can aggregate into
the unaddressed request rate.
"""

from __future__ import annotations

from agent.core.pending_intents import IntentKind, IntentStatus
from agent.logger import get_logger
from agent.state import State

_logger = get_logger(__name__)

_IN_SCOPE = {IntentKind.IN_SCOPE_INDEPENDENT.value, IntentKind.IN_SCOPE_INVALIDATING.value}


def count_open_in_scope(pending: list[dict]) -> int:
    return sum(
        1
        for d in (pending or [])
        if d.get("kind") in _IN_SCOPE and d.get("status") == IntentStatus.OPEN.value
    )


def log_unaddressed_request(state: State, *, count: int) -> None:
    if count <= 0:
        return
    _logger.info(
        "unaddressed_request",
        extra={
            "event": "unaddressed_request",
            "open_count": count,
            "app_run_id": state.get("app_run_id", ""),
        },
    )
