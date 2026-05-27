"""
agent.py — CareWellnessAgent: sends Care Coach program details using the
already-confirmed delivery contact from session context.
Never re-collects delivery information.
"""

from __future__ import annotations

import random

from agent.agents.care_wellness.constants import (
    AGENT_NAME,
    CARE_COACH_INTRO_TEMPLATES,
    LOG_DETAILS_SENT,
    LOG_ENTERED,
    LOG_PORTAL_SHARED,
    MSG_NO_CONTACT,
    REWARDS_PORTAL_TEMPLATES,
    WELLNESS_PORTAL_URL,
)
from agent.agents.care_wellness.handlers import _resolve_delivery_contact, dispatch_care_coach
from agent.core.agent import BaseAgent
from agent.logger import get_logger
from agent.state import State
from agent.utils import pick

logger = get_logger(__name__)


class CareWellnessAgent(BaseAgent):
    AGENT_NAME = AGENT_NAME

    async def run(self, state: State) -> dict:
        # ── EARLY EXIT: already dispatched ──────────────────────────────────
        if state.get("care_coach_details_sent"):
            return self.signal_complete(
                state,
                message="",
                resolved_intents=["care_wellness"],
                context_updates=self._completion_context(state),
            )

        # ── Resolve delivery contact from session context ────────────────────
        method, contact = _resolve_delivery_contact(state)
        if not method or not contact:
            return self.signal_escalate(
                state, pick(MSG_NO_CONTACT), reason="no_delivery_contact_for_care_coach"
            )

        # ── Dispatch Care Coach details ──────────────────────────────────────
        if fail := await dispatch_care_coach(self, state, method, contact):
            return fail

        logger.info(LOG_DETAILS_SENT, extra={"method": method, "contact_tail": contact[-4:]})

        # ── Build confirmation message ────────────────────────────────────────
        # Format contact for display: fax numbers as digits, emails as-is
        display_contact = contact
        if method == "fax" and contact.isdigit() and len(contact) == 10:
            display_contact = contact  # digits only — matches sample script style

        message = random.choice(CARE_COACH_INTRO_TEMPLATES).format(
            method=method, contact=display_contact
        )

        logger.info(LOG_DETAILS_SENT, extra={"method": method, "contact_tail": contact[-4:]})

        return self.signal_complete(
            state,
            message=message,
            resolved_intents=["care_wellness"],
            context_updates=self._completion_context(state),
        )

    @staticmethod
    def _completion_context(state: State) -> dict:
        return {
            "care_coach_offered": True,
            "care_coach_details_sent": True,
            "rewards_portal_shared": state.get("rewards_portal_shared", False),
            "delivery_method": state.get("delivery_method", ""),
            "fax": state.get("fax", ""),
            "email": state.get("email", ""),
        }


async def care_wellness_agent(state: State) -> dict:
    logger.info(LOG_ENTERED, extra={"call_intent": state.get("call_intent", "")})
    return await CareWellnessAgent.from_state(state).execute(state)
