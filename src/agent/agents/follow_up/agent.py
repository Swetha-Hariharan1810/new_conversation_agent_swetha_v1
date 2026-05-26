"""
agent.py — FollowUpAgent: answers member follow-up questions from session
context after the main provider-services flow completes.

Design:
  - ONE LLM call (generation only) per answering turn.
  - No extraction, no guard pipeline, no new-intent detection.
  - Closure is detected with keyword matching before the LLM is called.
  - All session state is serialised and passed directly to the LLM.
"""
from __future__ import annotations

from agent.agents.follow_up.constants import (
    AGENT_NAME,
    CLOSURE_KEYWORDS,
    LOG_ANSWERED,
    LOG_CANNOT_ANSWER,
    LOG_CLOSURE,
    LOG_ENTERED,
    MAX_FOLLOW_UP_TURNS,
    MSG_CANNOT_ANSWER,
    MSG_CONTINUATION,
)
from agent.agents.follow_up.llm import generate_follow_up_answer
from agent.core.agent import BaseAgent
from agent.logger import get_logger
from agent.state import State
from agent.utils import _last_user_msg, pick

logger = get_logger(__name__)


class FollowUpAgent(BaseAgent):
    AGENT_NAME = AGENT_NAME

    async def run(self, state: State) -> dict:
        messages = list(state.get("messages") or [])
        last_user = _last_user_msg(messages)
        turn_count = (state.get("follow_up_turn_count") or 0) + 1

        # ── Guard: max turns ──────────────────────────────────────────────
        if turn_count > MAX_FOLLOW_UP_TURNS:
            logger.info("follow_up_agent: max turns reached — routing to closure")
            return self.signal_complete(
                state,
                message="",
                resolved_intents=["follow_up"],
                closure_requested=True,
            )

        # ── Step 1: closure detection ─────────────────────────────────────
        if self._is_closure(last_user):
            logger.info(LOG_CLOSURE)
            return self.signal_complete(
                state,
                message="",
                resolved_intents=["follow_up"],
                closure_requested=True,
            )

        # ── Step 2: answer from session context via single LLM call ───────
        answer = await generate_follow_up_answer(state, last_user)

        if not answer:
            # LLM returned nothing or [CANNOT_ANSWER] — give a brief apology
            # but still ask the continuation question so the member can redirect
            logger.info(LOG_CANNOT_ANSWER)
            sorry = pick(MSG_CANNOT_ANSWER)
            continuation = pick(MSG_CONTINUATION)
            message = f"{sorry} {continuation}"
        else:
            logger.info(LOG_ANSWERED)
            continuation = pick(MSG_CONTINUATION)
            message = f"{answer}\n\n{continuation}"

        result = self.ask_member(state, message)
        result["follow_up_turn_count"] = turn_count
        result["follow_up_last_question"] = last_user
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _is_closure(self, text: str) -> bool:
        """True when the member's utterance signals they are done."""
        if not text:
            return False
        lower = text.lower().strip()
        if lower in CLOSURE_KEYWORDS:
            return True
        for kw in CLOSURE_KEYWORDS:
            if " " in kw and lower.startswith(kw):
                return True
        return False


async def follow_up_agent(state: State) -> dict:
    logger.info(LOG_ENTERED, extra={"call_intent": state.get("call_intent", "")})
    return await FollowUpAgent.from_state(state).execute(state)
