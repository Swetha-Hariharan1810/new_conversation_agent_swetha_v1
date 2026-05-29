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
    LOG_OUT_OF_SCOPE,
    MAX_CLARIFICATION_ATTEMPTS,
    OUT_OF_SCOPE_REASON,
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


def _match_out_of_scope_routing(utterance: str) -> tuple[str, str, str]:
    """
    Match the caller's utterance against the keyword routing table.
    Returns (matched_phrase, team, number).
    Falls back to defaults if no keyword matches.
    Runs in <0.1ms — no LLM call, no token cost.
    """
    from agent.agents.intake.constants import (
        OUT_OF_SCOPE_FALLBACK_NUMBER,
        OUT_OF_SCOPE_FALLBACK_TEAM,
        OUT_OF_SCOPE_KEYWORD_ROUTING,
    )

    t = (utterance or "").lower()
    for keyword, team, number in OUT_OF_SCOPE_KEYWORD_ROUTING:
        if keyword in t:
            return keyword, team, number
    return "", OUT_OF_SCOPE_FALLBACK_TEAM, OUT_OF_SCOPE_FALLBACK_NUMBER


async def handle_out_of_scope_intent(agent, state: State, result=None) -> dict:
    """
    Called when the extraction LLM classifies intent as out_of_scope.

    Team and phone number are resolved by keyword-matching the caller's
    last utterance — no LLM extraction needed, no silent fallback risk.
    Routes directly to END (not escalation_agent) so the caller hears
    exactly one message, not the out-of-scope message followed by the
    escalation agent's ref number message.
    """
    import random

    from langgraph.graph import END

    from agent.agents.intake.constants import (
        OUT_OF_SCOPE_MSG_TEMPLATES,
    )
    from agent.utils import _last_user_msg

    logger.warning(LOG_OUT_OF_SCOPE)

    # Get the caller's last utterance from state messages
    messages = list(state.get("messages") or [])
    last_user = _last_user_msg(messages)

    # Keyword match — O(n) over a small fixed list, <0.1ms
    keyword, team, number = _match_out_of_scope_routing(last_user)

    # Build topic description from matched keyword or generic fallback
    if keyword:
        # Make it sound natural: "billing" → "your billing question"
        topic_description = f"your {keyword} question"
    else:
        topic_description = "your request"

    msg = random.choice(OUT_OF_SCOPE_MSG_TEMPLATES).format(
        topic_description=topic_description,
        team=team,
        number=number,
    )

    # Route directly to END — do not go through escalation_agent.
    # The message already says "I'll connect you now" and gives the number.
    # escalation_agent would add a second message with a reference number.
    result = agent.ask_member(state, msg)
    result["next_node"] = END
    result["escalation_reason"] = OUT_OF_SCOPE_REASON
    result["is_interrupt"] = False
    return result
