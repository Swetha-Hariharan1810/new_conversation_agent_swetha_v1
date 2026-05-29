"""
agent.py — BenefitsAgent: fetches plan benefits from Salesforce,
explains them to the member, and makes the Care Coach proactive offer.
"""

from __future__ import annotations

import random

from agent.agents.benefits.constants import (
    AGENT_NAME,
    BENEFITS_EXPLANATION_TEMPLATE,
    BENEFITS_NOEXPLANATION_TEMPLATES,
    CARE_COACH_OFFER_TEMPLATES,
    LOG_BENEFITS_EXPLAINED,
    LOG_BENEFITS_FETCHED,
    LOG_ENTERED,
)
from agent.agents.benefits.handlers import fetch_benefits
from agent.agents.benefits.llm import extract_benefits_decision
from agent.core.agent import BaseAgent
from agent.llm.config import get_extraction_llm
from agent.logger import get_logger
from agent.slots.normalizers import normalize_yes_no
from agent.state import State
from agent.utils import _last_assistant_msg, _last_user_msg, build_extraction_prompt, pick

logger = get_logger(__name__)

_CARE_COACH_SLOT = "care_coach_response"


def _clean_amount(val) -> str:
    """
    Convert a Salesforce Decimal/int/str amount to a clean integer string.
    "600.0" → "600",  "600.00" → "600",  "600" → "600",  None → "0"
    Only strips trailing zeros when a decimal point is present — avoids
    the str.rstrip(".0") pitfall which treats chars individually.
    """
    s = str(val or "").strip()
    if not s:
        return "0"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


class BenefitsAgent(BaseAgent):
    AGENT_NAME = AGENT_NAME

    async def run(self, state: State) -> dict:  # noqa: C901
        messages = list(state.get("messages") or [])
        last_user = _last_user_msg(messages)
        last_agent = _last_assistant_msg(messages)
        current_awaiting = state.get("awaiting_slot", "")

        # ── EARLY EXIT: only gate on care_coach_offered; benefits_explained may be False on NO path ─
        if state.get("care_coach_offered"):
            return self.signal_complete(
                state,
                message="",
                resolved_intents=["benefits_inquiry"],
                context_updates=self._completion_context(state, bool(state.get("proactive_offer_available"))),
                proactive_offer_available=bool(state.get("proactive_offer_available")),
            )

        # ── PHASE B: Member responded to the Care Coach offer ────────────────
        if current_awaiting == _CARE_COACH_SLOT:
            return await self._handle_care_coach_response(state, messages, last_user, last_agent)

        # ── PHASE A: Branch on proactive_offer_available ─────────────────────
        benefits_wanted = bool(state.get("proactive_offer_available"))

        if benefits_wanted:
            # YES path — fetch SF, explain benefits, append care coach offer
            benefits, interrupt = await fetch_benefits(self, state)
            if interrupt:
                return interrupt

            logger.info(LOG_BENEFITS_FETCHED)

            indv_ded = _clean_amount(benefits.get("individual_deductible"))
            fam_ded = _clean_amount(benefits.get("family_deductible"))
            coins_pct = _clean_amount(benefits.get("coinsurance_percent"))
            indv_oop = _clean_amount(benefits.get("individual_oop_max"))
            fam_oop = _clean_amount(benefits.get("family_oop_max"))

            explanation = BENEFITS_EXPLANATION_TEMPLATE.format(
                individual_deductible=indv_ded,
                family_deductible=fam_ded,
                coinsurance_percent=coins_pct,
                individual_oop_max=indv_oop,
                family_oop_max=fam_oop,
            )
            care_coach_offer = random.choice(CARE_COACH_OFFER_TEMPLATES)
            full_message = f"{explanation}\n\n{care_coach_offer}"

            logger.info(LOG_BENEFITS_EXPLAINED)

            result = self.ask_member(state, full_message)
            result["awaiting_slot"] = _CARE_COACH_SLOT
            result["benefits_explained"] = True
            result["individual_deductible"] = indv_ded
            result["family_deductible"] = fam_ded
            result["coinsurance_percent"] = coins_pct
            result["individual_oop_max"] = indv_oop
            result["family_oop_max"] = fam_oop
            return result

        else:
            # NO path — skip SF fetch, go straight to Care Coach offer
            message = pick(BENEFITS_NOEXPLANATION_TEMPLATES)
            result = self.ask_member(state, message)
            result["awaiting_slot"] = _CARE_COACH_SLOT
            result["benefits_explained"] = False
            return result

    # -------------------------------------------------------------------------

    async def _handle_care_coach_response(
        self, state: State, messages: list, last_user: str, last_agent: str
    ) -> dict:
        """Extract and act on the member's yes/no to the Care Coach offer."""
        attempts_dict = state.get("slot_attempts") or {}
        current_attempt = attempts_dict.get(_CARE_COACH_SLOT, {})
        attempt_count = current_attempt.get("attempt_count", 0) if isinstance(current_attempt, dict) else 0

        result = await extract_benefits_decision(
            get_extraction_llm(),
            build_extraction_prompt("extraction/benefits.md"),
            awaiting_slot=_CARE_COACH_SLOT,
            last_agent_message=last_agent,
            last_user_message=last_user,
            confirmed_slots={},
            attempt=attempt_count,
            recent_messages=messages[-6:],
        )

        if interrupt := await self.run_conversation_guards(state, user_text=last_user, result=result):
            return interrupt

        extracted = (result.extracted or {}) if result else {}
        raw_response = extracted.get("care_coach_response", "")
        normalized = normalize_yes_no(raw_response) if raw_response else ""

        if normalized == "yes":
            return self.signal_complete(
                state,
                message="",
                resolved_intents=["benefits_inquiry"],
                context_updates=self._completion_context(state, True),
                proactive_offer_available=True,
            )

        if normalized == "no":
            return self.signal_complete(
                state,
                message="",
                resolved_intents=["benefits_inquiry"],
                context_updates=self._completion_context(state, False),
                proactive_offer_available=False,
            )

        # No clear yes/no — retry or exhaust gracefully
        self.slot_fail(_CARE_COACH_SLOT)
        if self.get_slot(_CARE_COACH_SLOT).is_exhausted():
            return self.signal_complete(
                state,
                message="",
                resolved_intents=["benefits_inquiry"],
                context_updates=self._completion_context(state, False),
                proactive_offer_available=False,
            )

        retry_msg = random.choice(CARE_COACH_OFFER_TEMPLATES)
        retry_result = self.ask_member(state, retry_msg)
        retry_result["awaiting_slot"] = _CARE_COACH_SLOT
        retry_result["benefits_explained"] = True
        return retry_result

    @staticmethod
    def _completion_context(state: State, care_coach_accepted: bool) -> dict:
        return {
            "benefits_explained": state.get("benefits_explained", False),
            "care_coach_offered": True,
            "care_coach_offer_made": True,
            "proactive_offer_available": care_coach_accepted,
            "individual_deductible": state.get("individual_deductible", ""),
            "family_deductible": state.get("family_deductible", ""),
            "coinsurance_percent": state.get("coinsurance_percent", ""),
            "individual_oop_max": state.get("individual_oop_max", ""),
            "family_oop_max": state.get("family_oop_max", ""),
        }


async def benefits_agent(state: State) -> dict:
    logger.info(LOG_ENTERED, extra={"call_intent": state.get("call_intent", "")})
    return await BenefitsAgent.from_state(state).execute(state)
