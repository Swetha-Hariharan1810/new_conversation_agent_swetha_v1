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
    BARE_AFFIRMATIONS,
    CLOSURE_KEYWORDS,
    LOG_ANSWERED,
    LOG_CANNOT_ANSWER,
    LOG_CLOSURE,
    LOG_ENTERED,
    MAX_FOLLOW_UP_TURNS,
    MSG_CANNOT_ANSWER,
    MSG_CONTINUATION,
    MSG_NUDGE,
)
from agent.agents.follow_up.llm import extract_follow_up_decision, generate_follow_up_answer
from agent.core.agent import BaseAgent
from agent.llm.config import get_extraction_llm
from agent.logger import get_logger
from agent.state import State
from agent.utils import _last_assistant_msg, _last_user_msg, build_extraction_prompt, pick

logger = get_logger(__name__)


class FollowUpAgent(BaseAgent):
    AGENT_NAME = AGENT_NAME

    async def run(self, state: State) -> dict:
        messages = list(state.get("messages") or [])
        last_user = _last_user_msg(messages)
        last_agent = _last_assistant_msg(messages)
        turn_count = (state.get("follow_up_turn_count") or 0) + 1

        # ── structural guard — unchanged ─────────────────────────────────────
        if turn_count > MAX_FOLLOW_UP_TURNS:
            logger.info("follow_up_agent: max turns reached — routing to closure")
            return self.signal_complete(
                state,
                message="",
                resolved_intents=["follow_up"],
                closure_requested=True,
            )

        # ── LLM 1: single call for guards + intent routing ───────────────────
        extraction_result = await extract_follow_up_decision(
            get_extraction_llm(),
            build_extraction_prompt("extraction/follow_up.md"),
            last_agent_message=last_agent,
            last_user_message=last_user,
            recent_messages=messages[-6:],
        )

        # ── guard check — short circuits if anything fires ───────────────────
        if interrupt := await self.run_conversation_guards(
            state, user_text=last_user, result=extraction_result
        ):
            return interrupt

        # ── intent routing from same extraction result ────────────────────────
        extracted = (extraction_result.extracted or {}) if extraction_result else {}
        follow_up_intent = extracted.get("follow_up_intent", "")

        if follow_up_intent == "done":
            logger.info(LOG_CLOSURE)
            return self.signal_complete(
                state,
                message="",
                resolved_intents=["follow_up"],
                closure_requested=True,
            )

        if follow_up_intent == "unsure":
            logger.info("follow_up_agent: bare affirmation — nudging")
            result = self.ask_member(state, pick(MSG_NUDGE))
            result["follow_up_turn_count"] = turn_count
            return result

        # ── keyword fallback — only when LLM extraction returned empty ────────
        if not follow_up_intent:
            if self._is_closure(last_user):
                logger.info(LOG_CLOSURE)
                return self.signal_complete(
                    state,
                    message="",
                    resolved_intents=["follow_up"],
                    closure_requested=True,
                )
            if last_user.lower().strip() in BARE_AFFIRMATIONS:
                logger.info("follow_up_agent: bare affirmation — nudging")
                result = self.ask_member(state, pick(MSG_NUDGE))
                result["follow_up_turn_count"] = turn_count
                return result

        # ── LLM 2: answer generation — follow_up_intent == "question" ────────
        answer = await generate_follow_up_answer(state, last_user)

        if not answer:
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
