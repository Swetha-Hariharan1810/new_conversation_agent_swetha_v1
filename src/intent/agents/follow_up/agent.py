"""
agent.py — FollowUpAgent: answers member follow-up questions from session
context after the main provider-services flow completes.

Design:
  - Keyword fast-path for closure and bare affirmations (zero LLM calls).
  - ONE LLM call per answering turn that handles guards + intent routing
    + answer generation from session context in a single forward pass.
  - No separate generation call; answer comes back on WorkerResult.answer.
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
from agent.agents.follow_up.llm import extract_follow_up_decision
from agent.core.agent import BaseAgent
from agent.llm.config import get_follow_up_llm
from agent.logger import get_logger
from agent.state import State
from agent.utils import _last_assistant_msg, _last_user_msg, build_extraction_prompt_extraction, pick

logger = get_logger(__name__)


class FollowUpAgent(BaseAgent):
    AGENT_NAME = AGENT_NAME

    async def run(self, state: State) -> dict:
        messages = list(state.get("messages") or [])
        last_user = _last_user_msg(messages)
        last_agent = _last_assistant_msg(messages)
        turn_count = (state.get("follow_up_turn_count") or 0) + 1

        # ── structural guard ─────────────────────────────────────────────────
        if turn_count > MAX_FOLLOW_UP_TURNS:
            logger.info("follow_up_agent: max turns reached — routing to closure")
            return self.signal_complete(
                state,
                message="",
                resolved_intents=["follow_up"],
                closure_requested=True,
            )

        # ── FAST PATH: keyword checks — zero LLM calls ───────────────────────
        if last_user:
            lower = last_user.lower().strip()

            # Closure keywords — signal done immediately
            if self._is_closure(last_user):
                logger.info(LOG_CLOSURE)
                return self.signal_complete(
                    state,
                    message="",
                    resolved_intents=["follow_up"],
                    closure_requested=True,
                )

            # Bare affirmations — nudge without calling LLM
            if lower in BARE_AFFIRMATIONS:
                logger.info("follow_up_agent: bare affirmation — nudging")
                result = self.ask_member(state, pick(MSG_NUDGE))
                result["follow_up_turn_count"] = turn_count
                return result

        # ── SINGLE LLM CALL: guards + intent + answer generation ────────────
        extraction_result = await extract_follow_up_decision(
            get_follow_up_llm(),
            build_extraction_prompt_extraction("extraction/follow_up.md"),
            last_agent_message=last_agent,
            last_user_message=last_user,
            recent_messages=messages[-6:],
            state=state,
        )

        # ── guard check ──────────────────────────────────────────────────────
        if interrupt := await self.run_conversation_guards(
            state, user_text=last_user, result=extraction_result
        ):
            return interrupt

        # ── intent routing ───────────────────────────────────────────────────
        extracted = (extraction_result.extracted or {}) if extraction_result else {}
        follow_up_intent = extracted.get("follow_up_intent", "")
        answer = (extraction_result.answer or "").strip() if extraction_result else ""

        if follow_up_intent == "done":
            logger.info(LOG_CLOSURE)
            return self.signal_complete(
                state,
                message="",
                resolved_intents=["follow_up"],
                closure_requested=True,
            )

        if follow_up_intent == "unsure":
            logger.info("follow_up_agent: unsure — nudging")
            nudge_message = answer if answer else pick(MSG_NUDGE)
            result = self.ask_member(state, nudge_message)
            result["follow_up_turn_count"] = turn_count
            return result

        # ── keyword fallback — only when LLM extraction returned empty ───────
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

        # ── use answer from the single LLM call ─────────────────────────────
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
