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
    CLOSURE_KEYWORDS,
    FLOW_COMPLETE_FLAGS,
    INTAKE_INTENTS,
    INTAKE_RESCREEN_INTENTS,
    LOG_ANSWERED,
    LOG_CANNOT_ANSWER,
    LOG_CLOSURE,
    LOG_ENTERED,
    LOG_NEW_INTENT,
    MAX_CANNOT_ANSWER_BEFORE_ESCALATE,
    MAX_FOLLOW_UP_TURNS,
    MSG_CANNOT_ANSWER,
    MSG_CONTINUATION,
    MSG_FOLLOW_UP_ASK,
    MSG_NUDGE,
    MSG_UPDATE_REQUEST_ESCALATE,
)
from agent.agents.follow_up.llm import extract_follow_up_decision
from agent.core.agent import BaseAgent
from agent.llm.config import get_follow_up_llm
from agent.llm.schema import FollowUpIntent
from agent.logger import get_logger
from agent.orchestration.orchestration import AgentNode
from agent.state import State, reset_for_new_intent
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


def _last_user_is_question(last_user: str) -> bool:
    """
    Return True only when the last user message is a genuine question or
    statement that follow_up_agent should classify and answer via the LLM,
    rather than overriding it with the canned opening prompt.

    The opener should be suppressed when:
      - The message is non-empty AND
      - It is not a bare affirmation ("yes", "ok", …) AND
      - It is not a closure signal ("no", "bye", "that's all", …)

    Bare affirmations and closure signals should still receive the opener
    (or be handled by the bare-affirmation fast path / LLM DONE detection
    on the *next* turn after the opener is sent).
    """
    if not last_user:
        return False
    lowered = last_user.lower().strip()
    # Bare affirmations: let the opener fire, then the member can respond.
    if lowered in BARE_AFFIRMATIONS:
        return False
    # Closure signals (e.g. "no", "bye"): the opener must NOT fire because
    # it would echo their "no" back as a new question. Instead fall through
    # to the LLM which will classify it as DONE and close the call.
    if lowered in CLOSURE_KEYWORDS:
        return True  # treat as substantive so LLM classifies DONE
    return True


def _prior_flow_complete(state: State, intent: str) -> bool:
    """True when the flow for `intent` has already finished this call.

    Lets a same-intent request (e.g. a second, distinct claim) qualify as a
    fresh intake while a same-intent clarification — where the flow is still
    open — does not.
    """
    flag = FLOW_COMPLETE_FLAGS.get(intent)
    return bool(state.get(flag)) if flag else False


def is_new_intake_intent(detected_intent: str | None, state: State) -> bool:
    """True only when `detected_intent` is a fresh, routable intake intent that
    warrants restarting the call from verification.

    Returns False for:
      * empty / missing detected_intent — covers "no"/goodbye/end-call, survey
        responses, and same-intent clarifications, which the classifier returns
        without a detected_intent (as done/unsure/question, not new_intent);
      * any intent outside the routable intake set (INTAKE_INTENTS);
      * the same intent just served, unless that prior flow already completed
        (otherwise it is a same-intent clarification, not a new request).
    """
    if not detected_intent:
        return False
    intent = detected_intent.strip().lower()
    if intent not in INTAKE_INTENTS:
        return False
    served = (state.get("call_intent") or "").strip().lower()
    if intent != served:
        return True
    return _prior_flow_complete(state, intent)


class FollowUpAgent(BaseAgent):
    AGENT_NAME = AGENT_NAME

    async def run(self, state: State) -> dict:  # noqa: C901
        messages = list(state.get("messages") or [])
        last_user = _last_user_msg(messages)
        last_agent = _last_assistant_msg(messages)

        # ── PHASE 0: First entry — ask opening question ───────────────────────
        if not state.get("follow_up_turn_count"):
            if not state.get("claim_timeline_notification_channel"):
                # Normal entry: opening question not yet delivered.
                #
                # Two cases where we must NOT send the canned opener and should
                # instead fall through to LLM classification:
                #
                # 1. The member already asked a substantive question before
                #    follow_up_agent ran for the first time — e.g. care_wellness
                #    asked "Is there anything else?" and the member replied with
                #    "What is my deductible?" or "What is my OOP max?".
                #    Suppressing the opener lets the LLM answer the question.
                #
                # 2. The member said a closure/DONE word (e.g. "no") in reply
                #    to care_wellness's closing question.  Sending the opener
                #    would discard their "no" and loop them back into a
                #    redundant "Aside from this…?" prompt.  Instead fall through
                #    so the LLM classifies it as DONE and closes the call.
                #
                # Only send the opener when last_user is truly empty (no prior
                # human turn) or is a bare affirmation that carries no intent.
                if _last_user_is_question(last_user):
                    # Fall through to LLM classification.
                    # Seed counters so the rest of the function runs correctly.
                    state = {
                        **state,
                        "follow_up_turn_count": 1,
                        "follow_up_cannot_answer_count": 0,
                    }
                else:
                    result = self.ask_member(state, pick(MSG_FOLLOW_UP_ASK))
                    result["follow_up_turn_count"] = 1
                    result["follow_up_cannot_answer_count"] = 0
                    return result

            else:
                # Handoff from notification_setup already delivered the combined
                # confirm + "Aside from this..." message. The last human message
                # in history (e.g. "email") is the N2 preference answer — it must
                # NOT be treated as the member's follow-up response.
                # Guard: if the last conversation message is an AI message, the
                # member has not yet replied to "Aside from this..." — yield
                # silently with is_interrupt=True (no new AI turn) and wait.
                last_msg = messages[-1] if messages else {}
                last_role = (
                    last_msg.get("role", "") if isinstance(last_msg, dict) else getattr(last_msg, "role", "")
                )
                if last_role == "assistant":
                    return {
                        "next_node": self.AGENT_NAME,
                        "is_interrupt": True,
                        "active_agent": self.AGENT_NAME,
                        "slot_attempts": self.slots_dict(),
                        "metadata_events": [],
                        "app_run_id": state.get("app_run_id", ""),
                        "follow_up_turn_count": 1,
                        "follow_up_cannot_answer_count": 0,
                    }
                # Last message is human — member already responded to the
                # "Aside from this..." question; fall through to LLM classification.

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
        call_intent = state.get("call_intent", "")
        if call_intent == "claim_services":
            prompt_file = "extraction/follow_up_claims.md"
        else:
            prompt_file = "extraction/follow_up.md"

        extraction_result = await extract_follow_up_decision(
            get_follow_up_llm(),
            build_extraction_prompt_extraction(prompt_file),
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

        # ── NEW_INTENT — member asked about a different service ──────────────────
        if follow_up_intent == FollowUpIntent.NEW_INTENT:
            detected = (extraction_result.detected_intent or "").strip()
            if is_new_intake_intent(detected, state):
                # Fresh intake intent. Most go straight to verification, but a few
                # must pass back through intake first so its front-door screening
                # (e.g. the unsupported-provider-type gate) re-runs before identity
                # is re-collected.
                if detected.lower() in INTAKE_RESCREEN_INTENTS:
                    return self._reroute_through_intake(state, detected)
                # Fresh intake intent → full reset + re-verify before the new flow.
                return self._reroute_through_verification(state, detected)
            # Not a fresh intake intent (missing/unrecognised detected_intent, or a
            # same-intent clarification). Fall through to the QUESTION/cannot-answer
            # handling below so we stay in follow_up rather than restarting the call.
            logger.info(
                "follow_up_agent: new_intent did not qualify as fresh intake — staying in follow_up",
                extra={"detected_intent": detected, "call_intent": state.get("call_intent", "")},
            )

        # ── UNSURE ────────────────────────────────────────────────────────────
        if follow_up_intent == FollowUpIntent.UNSURE:
            logger.info("follow_up_agent: unsure — nudging")
            result = self.ask_member(state, pick(MSG_NUDGE))
            result["follow_up_turn_count"] = turn_count
            result["follow_up_cannot_answer_count"] = 0  # reset streak
            return result

        # ── QUESTION ─────────────────────────────────────────────────────────
        # Strip forbidden redirect phrases — only block if the answer STARTS with them
        if answer:
            answer_lower = answer.lower().strip()
            if any(answer_lower.startswith(p) for p in _FORBIDDEN_ANSWER_PHRASES):
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

    def _reroute_through_verification(self, state: State, detected_intent: str) -> dict:
        """
        A fresh intake intent was detected mid-follow-up. Fully reset the
        conversation (identity + verification flags + every domain field) and
        route to the verification node so the caller re-verifies before the new
        flow begins.

        Routing uses the same mechanism every other hand-off uses: set
        ``next_node`` and let ``conditional_routing`` (the follow_up node's
        conditional edge) dispatch — no Command(goto=...) needed. ``is_interrupt``
        is False so verification runs in the same super-step and owns its own
        first prompt.

        ``reset_for_new_intent`` stages the new intent in BOTH ``call_intent``
        (which verification reads to choose its claims/provider pipeline, and
        which the post-verification fast-path reads to pick the domain agent) and
        ``pending_intent`` (a durable marker that this is a mid-call switch).
        """
        logger.info(
            LOG_NEW_INTENT,
            extra={"detected_intent": detected_intent, "previous_intent": state.get("call_intent", "")},
        )
        updates = reset_for_new_intent(state, detected_intent)
        updates["next_node"] = AgentNode.VERIFICATION.value
        updates["is_interrupt"] = False
        updates["active_agent"] = self.AGENT_NAME
        updates["metadata_events"] = []
        updates["app_run_id"] = state.get("app_run_id", "")
        return updates

    def _reroute_through_intake(self, state: State, detected_intent: str) -> dict:
        """
        A fresh intake intent that must be re-screened was detected mid-follow-up.
        Like ``_reroute_through_verification`` this fully resets the conversation,
        but routes to the *intake* node instead of verification so intake re-applies
        its front-door screening (e.g. the unsupported-provider-type gate) before
        identity is re-collected.

        ``call_intent`` is deliberately cleared here. ``reset_for_new_intent`` stages
        the intent in ``call_intent``, but intake's entry guard
        (``if state.get("call_intent")``) skips classification — and therefore
        screening — whenever ``call_intent`` is set, routing straight to
        verification. Leaving it populated would defeat the re-screen. With
        ``call_intent`` empty, intake re-classifies the triggering utterance, runs
        its screening, then sets ``call_intent`` itself and hands off to verification
        exactly as on a first-time call.

        ``pending_intent`` and ``reverify_bridge_pending`` — both meant for the
        direct-to-verification path — are cleared for the same reason: intake owns
        the bridge message, and the subsequent verification should behave as a
        first-time verification routed by ``call_intent``.

        The three overrides on top of ``reset_for_new_intent``:
          * ``call_intent = ""``             — force intake to re-classify (the
            entry-guard fix; a populated call_intent skips classify + screen).
          * ``pending_intent = ""``          — take the normal first-time
            intake → verification → orchestrator path; no mid-call-switch dispatch.
          * ``reverify_bridge_pending = False`` — intake emits its own first-name
            bridge after classifying, so don't have verification double-bridge.

        Routing reuses the intake NODE rather than a bespoke edge: setting
        ``next_node = intake_agent`` with ``is_interrupt = False`` runs intake in the
        same super-step (no extra round-trip), and we inherit ``intake_routing``'s
        existing escalate / verification-bridge / END dispatch for free. Intake's
        greeting is skipped because ``messages`` is non-empty mid-call, so the
        triggering utterance is the one it re-classifies.
        """
        logger.info(
            LOG_NEW_INTENT,
            extra={
                "detected_intent": detected_intent,
                "previous_intent": state.get("call_intent", ""),
                "route": "intake_rescreen",
            },
        )
        updates = reset_for_new_intent(state, detected_intent)
        updates["call_intent"] = ""  # force intake to re-classify + re-screen
        updates["pending_intent"] = ""  # intake → verification routes by call_intent
        updates["reverify_bridge_pending"] = False  # intake delivers its own bridge
        updates["next_node"] = AgentNode.INTAKE.value
        updates["is_interrupt"] = False
        updates["active_agent"] = self.AGENT_NAME
        updates["metadata_events"] = []
        updates["app_run_id"] = state.get("app_run_id", "")
        return updates


async def follow_up_agent(state: State) -> dict:
    logger.info(LOG_ENTERED, extra={"call_intent": state.get("call_intent", "")})
    return await FollowUpAgent.from_state(state).execute(state)
