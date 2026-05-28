"""
handlers.py — Unclear intent handling for IntakeAgent.

Uses LLM 2 (Gemini, generate_recovery_message) to produce a natural
recovery sentence. Falls back to static strings on exception.
"""

from __future__ import annotations

from agent.agents.intake.constants import (
    INTENT_SLOT,
    LOG_INTENT_UNCLEAR,
    LOG_MAX_RETRY,
    MAX_CLARIFICATION_ATTEMPTS,
    UNCLEAR_ESCALATION,
    UNCLEAR_INTENT_REASON,
)
from agent.logger import get_logger
from agent.state import State
from agent.utils import _last_user_msg

logger = get_logger(__name__)


def _get_clarification_attempts(state: State) -> int:
    slot = (state.get("slot_attempts") or {}).get(INTENT_SLOT, {})
    return slot if isinstance(slot, int) else slot.get("attempt_count", 0)


async def handle_unclear_intent(agent, state: State, result=None) -> dict:
    attempts = _get_clarification_attempts(state)

    if attempts >= MAX_CLARIFICATION_ATTEMPTS:
        logger.warning(LOG_MAX_RETRY)
        return agent.signal_escalate(
            state=state,
            message=UNCLEAR_ESCALATION,
            reason=UNCLEAR_INTENT_REASON,
            initiator="Agent",
        )

    agent.slot_fail(INTENT_SLOT, is_asr=False)
    logger.info(LOG_INTENT_UNCLEAR, extra={"attempt": attempts + 1})
    messages = list(state.get("messages") or [])

    # if attempts == 0:
    #     from agent.agents.intake.constants import UNCLEAR_FIRST_ATTEMPT_MSGS
    #     from agent.utils import pick

    #     msg = pick(UNCLEAR_FIRST_ATTEMPT_MSGS)
    #     return agent.ask_member(state, msg)

    from agent.llm.response_generator import generate_recovery_message

    label_override = (
        "their reason for calling — just say 'provider' to find "
        "a doctor, or 'claim' for billing questions. "
        "One word is fine."
    )
    msg = await generate_recovery_message(
        slot_name="intent",
        attempt=attempts + 1,
        guard="RETRY",
        last_messages=messages[-4:],
        slot_label_override=label_override,
        caller_name=None,
        confirmed_slots={},
        user_utterance=_last_user_msg(messages),
    )
    return agent.ask_member(state, msg)
