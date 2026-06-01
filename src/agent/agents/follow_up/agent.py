"""
agent.py — FollowUpAgent

Two escalation rules, both owned entirely by Python:

  UPDATE_REQUEST
    Signal escalate immediately on every occurrence.
    Fixed message + reference number. No counting, no threshold.

  MSG_CANNOT_ANSWER
    Count consecutive cannot-answer turns (resets when any real answer
    is given, or on done/unsure/update_request).
    On the 3rd consecutive cannot-answer, escalate using that same
    cannot-answer message as the prefix.
"""

from __future__ import annotations

from agent.agents.follow_up.constants import (
    AGENT_NAME,
    BARE_AFFIRMATIONS,
    LOG_ANSWERED,
    LOG_CANNOT_ANSWER,
    LOG_CLOSURE,
    LOG_ENTERED,
    MAX_CANNOT_ANSWER_BEFORE_ESCALATE,
    MAX_FOLLOW_UP_TURNS,
    MSG_CANNOT_ANSWER,
    MSG_CONTINUATION,
    MSG_NUDGE,
    MSG_UPDATE_REQUEST_ESCALATE,
)
from agent.agents.follow_up.llm import extract_follow_up_decision
from agent.core.agent import BaseAgent
from agent.llm.config import get_follow_up_llm
from agent.llm.schema import FollowUpIntent
from agent.logger import get_logger
from agent.state import State
from agent.utils import (
    _last_assistant_msg,
    _last_user_msg,
    build_extraction_prompt_extraction,
    pick,
)

logger = get_logger(__name__)

_FORBIDDEN_ANSWER_PHRASES = (
    "i can help you find",
    "i can help you with",
    "i can help with",
)


class FollowUpAgent(BaseAgent):
    AGENT_NAME = AGENT_NAME

    async def run(self, state: State) -> dict:  # noqa: C901
        messages = list(state.get("messages") or [])
        last_user = _last_user_msg(messages)
        last_agent = _last_assistant_msg(messages)
        turn_count = (state.get("follow_up_turn_count") or 0) + 1
        consecutive_cannot_answer = state.get("follow_up_cannot_answer_count") or 0

        # ── Hard turn cap ────────────────────────────────────────────────────
        if turn_count > MAX_FOLLOW_UP_TURNS:
            logger.info("follow_up_agent: max turns reached — routing to closure")
            return self.signal_complete(
                state,
                message="",
                resolved_intents=["follow_up"],
                closure_requested=True,
            )

        # ── FAST PATH: bare affirmations — zero LLM calls ───────────────────
        if last_user and last_user.lower().strip() in BARE_AFFIRMATIONS:
            logger.info("follow_up_agent: bare affirmation — nudging")
            result = self.ask_member(state, pick(MSG_NUDGE))
            result["follow_up_turn_count"] = turn_count
            result["follow_up_cannot_answer_count"] = consecutive_cannot_answer
            return result

        # ── LLM call: classify intent + generate answer ──────────────────────
        extraction_result = await extract_follow_up_decision(
            get_follow_up_llm(),
            build_extraction_prompt_extraction("extraction/follow_up.md"),
            last_agent_message=last_agent,
            last_user_message=last_user,
            recent_messages=messages[-6:],
            state=state,
        )

        # ── Conversation guards ──────────────────────────────────────────────
        if interrupt := await self.run_conversation_guards(
            state, user_text=last_user, result=extraction_result
        ):
            return interrupt

        follow_up_intent = extraction_result.follow_up_intent if extraction_result else FollowUpIntent.UNSURE
        answer = (extraction_result.answer or "").strip() if extraction_result else ""

        # ── DONE ─────────────────────────────────────────────────────────────
        if follow_up_intent == FollowUpIntent.DONE:
            logger.info(LOG_CLOSURE)
            return self.signal_complete(
                state,
                message="",
                resolved_intents=["follow_up"],
                closure_requested=True,
            )

        # ── UPDATE_REQUEST — immediate escalation, every time ────────────────
        if follow_up_intent == FollowUpIntent.UPDATE_REQUEST:
            logger.info("follow_up_agent: update_request — escalating immediately")
            return self.signal_escalate(
                state,
                MSG_UPDATE_REQUEST_ESCALATE,
                reason="update_request_in_follow_up",
                initiator="Agent",
            )

        # ── UNSURE ────────────────────────────────────────────────────────────
        if follow_up_intent == FollowUpIntent.UNSURE:
            logger.info("follow_up_agent: unsure — nudging")
            result = self.ask_member(state, pick(MSG_NUDGE))
            result["follow_up_turn_count"] = turn_count
            result["follow_up_cannot_answer_count"] = 0  # reset streak
            return result

        # ── QUESTION ─────────────────────────────────────────────────────────
        # Strip forbidden redirect phrases
        if answer and any(p in answer.lower() for p in _FORBIDDEN_ANSWER_PHRASES):
            logger.info(LOG_CANNOT_ANSWER)
            answer = ""

        if not answer:
            # Cannot answer this question
            new_cannot_answer_count = consecutive_cannot_answer + 1
            logger.info(LOG_CANNOT_ANSWER + " (consecutive=%d)", new_cannot_answer_count)
            cannot_answer_msg = pick(MSG_CANNOT_ANSWER)

            # ── Escalate on Nth consecutive cannot-answer ────────────────────
            if new_cannot_answer_count >= MAX_CANNOT_ANSWER_BEFORE_ESCALATE:
                logger.info(
                    "follow_up_agent: %d consecutive cannot-answer turns — escalating",
                    new_cannot_answer_count,
                )
                return self.signal_escalate(
                    state,
                    cannot_answer_msg,
                    reason="repeated_cannot_answer_in_follow_up",
                    initiator="Agent",
                )

            # Still under threshold — reply and count
            message = f"{cannot_answer_msg} {pick(MSG_CONTINUATION)}"
            result = self.ask_member(state, message)
            result["follow_up_turn_count"] = turn_count
            result["follow_up_cannot_answer_count"] = new_cannot_answer_count
            return result

        # Real answer — reset cannot-answer streak
        logger.info(LOG_ANSWERED)
        result = self.ask_member(state, answer)
        result["follow_up_turn_count"] = turn_count
        result["follow_up_last_question"] = last_user
        result["follow_up_cannot_answer_count"] = 0
        return result


async def follow_up_agent(state: State) -> dict:
    logger.info(LOG_ENTERED, extra={"call_intent": state.get("call_intent", "")})
    return await FollowUpAgent.from_state(state).execute(state)
