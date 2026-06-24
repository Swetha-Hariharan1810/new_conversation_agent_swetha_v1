"""handlers.py — Claim Adjustment workflow handlers."""

from __future__ import annotations

from agent.logger import get_logger
from agent.state import State
from agent.utils import pick

logger = get_logger(__name__)

_MSG_LOOKUP_FAIL = [
    "I'm sorry, I wasn't able to retrieve your adjustment request. "
    "Let me connect you with a representative who can help.",
    "I'm having trouble pulling up that adjustment request. Connecting you with a representative.",
]


async def lookup_adjustment(agent, state: State) -> tuple[dict | None, dict | None]:
    """
    Look up the adjustment request in Salesforce by reference_number + member_id.

    Returns (adjustment_record, interrupt_or_none).
    If interrupt is not None, caller must return it immediately.
    """
    from agent.storage.queries.adjustments import find_adjustment

    reference_number = state.get("reference_number", "")
    member_id = state.get("member_id", "")

    if not reference_number:
        logger.warning("lookup_adjustment: no reference_number in state")
        return None, agent.signal_escalate(
            state, pick(_MSG_LOOKUP_FAIL), reason="no_reference_number_for_lookup"
        )

    try:
        record = await find_adjustment(reference_number, member_id)
        if not record:
            return None, None  # not found — caller handles with MSG_REF_NOT_FOUND
        return record, None
    except Exception:
        logger.exception("lookup_adjustment: Salesforce call failed")
        return None, agent.signal_escalate(state, pick(_MSG_LOOKUP_FAIL), reason="adjustment_lookup_error")
