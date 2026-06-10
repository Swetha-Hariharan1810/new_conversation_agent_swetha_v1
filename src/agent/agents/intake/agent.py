"""
agent.py — Greeting and intent classification.

Flow:
  Turn 1: No messages yet → send greeting, wait for member
  Turn 2+: Run guards → classify intent → route to verification

Adding new intents:
  1. Add a value to IntentTag in models.py
  2. Update the intake.md prompt to describe the new intent
  3. Update SUPPORTED_TOPICS in constants.py if needed
"""

from __future__ import annotations

import random
import uuid

from agent.agents.intake.constants import (
    GREETING,
    INTENT_BRIDGE_MSGS,
    LOG_INTAKE_GREETING,
    LOG_INTENT_CLASSIFIED,
    MAX_CLARIFICATION_ATTEMPTS,
    OFFTOPIC_ESCALATION,
    OFFTOPIC_REASON,
)
from agent.agents.intake.handlers import (
    _get_clarification_attempts,
    handle_out_of_scope_intent,
    handle_unclear_intent,
)
from agent.agents.intake.llm import extract_intake_intent
from agent.agents.intake.models import IntentTag
from agent.core.agent import BaseAgent
from agent.llm.config import get_extraction_llm
from agent.llm.schema import EventType
from agent.logger import get_logger
from agent.orchestration.orchestration import AgentNode
from agent.state import State
from agent.utils import (
    _last_assistant_msg,
    _last_user_msg,
    build_extraction_prompt_core,
)

logger = get_logger(__name__)


class IntakeAgent(BaseAgent):
    AGENT_NAME = "intake_agent"

    def get_system_prompt(self, state: State) -> str:
        return build_extraction_prompt_core("extraction/intake.md")

    async def run(self, state: State) -> dict:
        app_run_id = state.get("app_run_id") or str(uuid.uuid4())

        # Guard: if intent already classified in a prior turn, skip
        # re-classification and hand off to verification immediately.
        # This prevents re-entry from sending the bridge message again.
        if state.get("call_intent"):
            logger.info(
                LOG_INTENT_CLASSIFIED,
                extra={"intent": state["call_intent"], "app_run_id": app_run_id},
            )
            return self.signal_complete(
                state=state,
                message="",
                resolved_intents=["intake"],
                context_updates={"app_run_id": app_run_id, "call_intent": state["call_intent"]},
                reasoning=f"Intent already classified as {state['call_intent']}",
            )

        # Turn 1: no messages yet — send greeting
        if not state.get("messages"):
            logger.info(LOG_INTAKE_GREETING, extra={"app_run_id": app_run_id})
            result = self.ask_member(state, GREETING)
            result["app_run_id"] = app_run_id
            return result

        messages = list(state.get("messages") or [])
        last_user = _last_user_msg(messages)
        last_agent = _last_assistant_msg(messages)
        attempts = _get_clarification_attempts(state)

        result = await extract_intake_intent(
            get_extraction_llm(),
            self.get_system_prompt(state),
            last_agent_message=last_agent,
            last_user_message=last_user,
            attempt=attempts,
            recent_messages=messages,
        )

        if interrupt := await self.run_conversation_guards(
            state,
            user_text=last_user,
            result=result,
        ):
            if getattr(result, "guard", "") == "OFFTOPIC_AGENT":
                if attempts >= MAX_CLARIFICATION_ATTEMPTS:
                    return self.signal_escalate(
                        state, OFFTOPIC_ESCALATION, OFFTOPIC_REASON, initiator="Agent"
                    )
            return interrupt

        intent_value = (result.extracted or {}).get("intent", IntentTag.UNCLEAR.value)

        if intent_value == IntentTag.OUT_OF_SCOPE.value:
            return await handle_out_of_scope_intent(agent=self, state=state, result=result)

        if intent_value == IntentTag.UNCLEAR.value:
            return await handle_unclear_intent(agent=self, state=state, result=result)

        # Intent is classified — check if caller also said something extra
        # that needs acknowledging before we route to verification.
        if result.event_type == EventType.ANSWERED_WITH_FOLLOWUP:
            logger.info(
                "IntakeAgent: answered_with_followup — calling generation LLM",
                extra={"intent": intent_value, "app_run_id": app_run_id},
            )
            # Intake has no slot state — call the generation LLM directly
            # rather than going through _generate_slot_retry_response.
            from agent.llm.response_generator import generate_recovery_message

            msg = await generate_recovery_message(
                slot_name="intent",
                attempt=0,
                guard="ANSWERED_WITH_FOLLOWUP",
                last_messages=messages[-4:],
                user_utterance=last_user,
                confirmed_slots={},
                extracted_value=intent_value,
                slot_label_override=(
                    "confirm back what they're calling about, address their "
                    "repeat/confirmation request, then ask for their first name "
                    "to get started — ask for nothing else"
                ),
            )
            # The caller's next utterance is their first name — route straight
            # to verification instead of bouncing through intake's re-entry guard.
            bridge = self.ask_member(state, msg)
            bridge["call_intent"] = intent_value
            bridge["app_run_id"] = app_run_id
            bridge["resolved_intents"] = ["intake"]
            bridge["next_node"] = AgentNode.VERIFICATION.value
            bridge["metadata_events"] = []
            return bridge

        # Clean answered path — fire bridge and route to verification
        logger.info(LOG_INTENT_CLASSIFIED, extra={"intent": intent_value, "app_run_id": app_run_id})
        bridge = self.ask_member(state, random.choice(INTENT_BRIDGE_MSGS))
        bridge["call_intent"] = intent_value
        bridge["app_run_id"] = app_run_id
        bridge["resolved_intents"] = ["intake"]
        bridge["next_node"] = AgentNode.VERIFICATION.value
        # bridge["metadata_events"] = [
        #     {
        #         "eventType": "CallAgentField",
        #         "data": {"field": "call_intent", "value": intent_value},
        #     }
        # ]
        bridge["metadata_events"] = []
        return bridge


async def intake_agent(state: State) -> dict:
    return await IntakeAgent.from_state(state).execute(state)
