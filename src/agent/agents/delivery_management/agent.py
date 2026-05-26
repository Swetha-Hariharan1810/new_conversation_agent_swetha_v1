"""
agent.py — DeliveryManagementAgent: confirms delivery contact, dispatches provider list,
makes proactive benefits offer.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone

from agent.agents.delivery_management.constants import (
    BENEFITS_OFFER_TEMPLATES,
    DELIVERY_WINDOW_MSG,
    EMAIL_READBACK_TEMPLATES,
    EMAIL_UPDATE_PROMPT,
    FAX_READBACK_TEMPLATES,
    FAX_UPDATE_PROMPT,
    LOG_CONTACT_CONFIRMED,
    LOG_CONTACT_UPDATED,
    LOG_ENTERED,
    LOG_LIST_DISPATCHED,
    LOG_METHOD_COLLECTED,
    MSG_CONTACT_EXHAUST,
    MSG_DISPATCH_FAIL,
)
from agent.agents.delivery_management.handlers import (
    dispatch_provider_list,
    update_email_in_salesforce,
    update_fax_in_salesforce,
)
from agent.agents.delivery_management.llm import extract_delivery_management_decision
from agent.agents.delivery_management.pipelines import (
    build_delivery_method_pipeline,
    build_email_pipeline,
    build_fax_pipeline,
)
from agent.core.agent import BaseAgent
from agent.llm.config import get_extraction_llm
from agent.logger import get_logger
from agent.slots.normalizers import normalize_email, normalize_fax_number, normalize_yes_no
from agent.slots.validators import validate_email, validate_fax_number
from agent.state import State
from agent.utils import _last_assistant_msg, _last_user_msg, build_extraction_prompt, pick

logger = get_logger(__name__)


class DeliveryManagementAgent(BaseAgent):
    AGENT_NAME = "delivery_management_agent"

    def __init__(self) -> None:
        super().__init__()
        self._delivery_method_pipeline = build_delivery_method_pipeline(self)
        self._fax_pipeline = build_fax_pipeline(self)
        self._email_pipeline = build_email_pipeline(self)

    async def run(self, state: State) -> dict:  # noqa: C901
        messages = list(state.get("messages") or [])
        last_user = _last_user_msg(messages)
        last_agent = _last_assistant_msg(messages)

        delivery_method = (state.get("delivery_method") or "").strip()
        fax_on_file = (state.get("fax") or "").strip()
        email_on_file = (state.get("email") or "").strip()

        current_awaiting = state.get("awaiting_slot", "") or "delivery_method"

        # ── EARLY EXIT: already dispatched + benefits offered (re-entry) ─────
        # Check before LLM extraction to avoid unnecessary calls.
        if state.get("provider_list_sent") and state.get("benefits_offer_made") and current_awaiting != "benefits_response":
            return self.signal_complete(
                state,
                message="",
                resolved_intents=["delivery_management"],
                context_updates=self._completion_context(state, delivery_method, False),
            )

        # ── RECOVERY: dispatched but offer not yet made ──────────────────────
        if state.get("provider_list_sent") and not state.get("benefits_offer_made") and current_awaiting != "benefits_response":
            provider_type = (state.get("provider_type") or "provider").strip()
            benefits_msg = random.choice(BENEFITS_OFFER_TEMPLATES).format(provider_type=provider_type)
            offer_result = self.ask_member(state, benefits_msg)
            offer_result["awaiting_slot"] = "benefits_response"
            offer_result["benefits_offer_made"] = True
            return offer_result

        state = {**state, "awaiting_slot": current_awaiting}

        # LLM extraction
        confirmed_slots: dict = {}
        if delivery_method:
            confirmed_slots["delivery_method"] = delivery_method

        attempts_dict = state.get("slot_attempts") or {}
        current_attempt = attempts_dict.get(current_awaiting, {})
        attempt_count = current_attempt.get("attempt_count", 0) if isinstance(current_attempt, dict) else 0

        result = await extract_delivery_management_decision(
            get_extraction_llm(),
            build_extraction_prompt("extraction/delivery_management.md"),
            awaiting_slot=current_awaiting,
            last_agent_message=last_agent,
            last_user_message=last_user,
            confirmed_slots=confirmed_slots,
            attempt=attempt_count,
            recent_messages=messages[-6:],
        )

        # Conversation guards
        if interrupt := await self.run_conversation_guards(state, user_text=last_user, result=result):
            return interrupt

        # ── BENEFITS RESPONSE PHASE ──────────────────────────────────────────
        if current_awaiting == "benefits_response":
            return await self._handle_benefits_response(state, result)

        # ── COLLECT DELIVERY METHOD ──────────────────────────────────────────
        if not delivery_method:
            collected: dict = {"delivery_method": ""}
            if interrupt := await self._delivery_method_pipeline.collect(
                state, messages, collected, decision=result
            ):
                return interrupt
            delivery_method = collected["delivery_method"]
            logger.info(LOG_METHOD_COLLECTED, extra={"delivery_method": delivery_method})
            # Immediately proceed to contact confirmation in the same turn
            return self._ask_contact_confirmation(state, delivery_method, fax_on_file, email_on_file)

        # ── FAX CONFIRMATION ─────────────────────────────────────────────────
        if current_awaiting == "fax_confirmed":
            extracted = (result.extracted or {}) if result else {}
            new_fax_raw = extracted.get("fax", "")
            contact_conf_raw = extracted.get("contact_confirmed", "")

            if new_fax_raw:
                normalized = normalize_fax_number(str(new_fax_raw))
                if normalized and validate_fax_number(normalized).valid:
                    if fail := await update_fax_in_salesforce(self, state, normalized):
                        return fail
                    logger.info(LOG_CONTACT_UPDATED, extra={"fax_tail": normalized[-4:]})
                    return await self._proceed_to_dispatch(state, delivery_method, normalized)
                ask_result = self.ask_member(state, FAX_UPDATE_PROMPT)
                ask_result["awaiting_slot"] = "fax"
                return ask_result

            contact_conf = normalize_yes_no(contact_conf_raw) if contact_conf_raw else ""
            if contact_conf == "yes":
                logger.info(LOG_CONTACT_CONFIRMED, extra={"method": "fax"})
                return await self._proceed_to_dispatch(state, delivery_method, fax_on_file)
            if contact_conf == "no":
                ask_result = self.ask_member(state, FAX_UPDATE_PROMPT)
                ask_result["awaiting_slot"] = "fax"
                return ask_result

            # No clear yes/no — retry or exhaust
            self.slot_fail("fax_confirmed")
            if self.get_slot("fax_confirmed").is_exhausted():
                return self.signal_escalate(
                    state, pick(MSG_CONTACT_EXHAUST), reason="fax_confirmed_exhausted"
                )
            retry_msg = random.choice(FAX_READBACK_TEMPLATES).format(fax=fax_on_file)
            retry_result = self.ask_member(state, retry_msg)
            retry_result["awaiting_slot"] = "fax_confirmed"
            return retry_result

        # ── FAX UPDATE ───────────────────────────────────────────────────────
        if current_awaiting == "fax":
            fax_state = {**state, "fax": ""}
            collected_fax: dict = {"fax": ""}
            if interrupt := await self._fax_pipeline.collect(
                fax_state, messages, collected_fax, decision=result
            ):
                return interrupt
            new_fax = collected_fax["fax"]
            if fail := await update_fax_in_salesforce(self, state, new_fax):
                return fail
            logger.info(LOG_CONTACT_UPDATED, extra={"fax_tail": new_fax[-4:]})
            return await self._proceed_to_dispatch(state, delivery_method, new_fax)

        # ── EMAIL CONFIRMATION ───────────────────────────────────────────────
        if current_awaiting == "email_confirmed":
            extracted = (result.extracted or {}) if result else {}
            new_email_raw = extracted.get("email", "")
            contact_conf_raw = extracted.get("contact_confirmed", "")

            if new_email_raw:
                normalized = normalize_email(str(new_email_raw))
                if normalized and validate_email(normalized).valid:
                    if fail := await update_email_in_salesforce(self, state, normalized):
                        return fail
                    logger.info(LOG_CONTACT_UPDATED, extra={"method": "email"})
                    return await self._proceed_to_dispatch(state, delivery_method, normalized)
                ask_result = self.ask_member(state, EMAIL_UPDATE_PROMPT)
                ask_result["awaiting_slot"] = "email"
                return ask_result

            contact_conf = normalize_yes_no(contact_conf_raw) if contact_conf_raw else ""
            if contact_conf == "yes":
                logger.info(LOG_CONTACT_CONFIRMED, extra={"method": "email"})
                return await self._proceed_to_dispatch(state, delivery_method, email_on_file)
            if contact_conf == "no":
                ask_result = self.ask_member(state, EMAIL_UPDATE_PROMPT)
                ask_result["awaiting_slot"] = "email"
                return ask_result

            # No clear yes/no — retry or exhaust
            self.slot_fail("email_confirmed")
            if self.get_slot("email_confirmed").is_exhausted():
                return self.signal_escalate(
                    state, pick(MSG_CONTACT_EXHAUST), reason="email_confirmed_exhausted"
                )
            retry_msg = random.choice(EMAIL_READBACK_TEMPLATES).format(email=email_on_file)
            retry_result = self.ask_member(state, retry_msg)
            retry_result["awaiting_slot"] = "email_confirmed"
            return retry_result

        # ── EMAIL UPDATE ─────────────────────────────────────────────────────
        if current_awaiting == "email":
            email_state = {**state, "email": ""}
            collected_email: dict = {"email": ""}
            if interrupt := await self._email_pipeline.collect(
                email_state, messages, collected_email, decision=result
            ):
                return interrupt
            new_email = collected_email["email"]
            if fail := await update_email_in_salesforce(self, state, new_email):
                return fail
            logger.info(LOG_CONTACT_UPDATED, extra={"method": "email"})
            return await self._proceed_to_dispatch(state, delivery_method, new_email)

        # ── FALLBACK: delivery_method known but awaiting_slot not matched ────
        # Re-ask for contact confirmation (handles unexpected re-entry)
        return self._ask_contact_confirmation(state, delivery_method, fax_on_file, email_on_file)

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    async def _handle_benefits_response(self, state: State, result) -> dict:
        """Process the member's yes/no response to the benefits offer."""
        extracted = (result.extracted or {}) if result else {}
        benefits_raw = extracted.get("benefits_response", "")
        benefits_conf = normalize_yes_no(benefits_raw) if benefits_raw else ""

        if benefits_conf in ("yes", "no"):
            return self.signal_complete(
                state,
                message="",
                resolved_intents=["delivery_management"],
                context_updates=self._completion_context(
                    state,
                    state.get("delivery_method", ""),
                    benefits_conf == "yes",
                ),
                proactive_offer_available=(benefits_conf == "yes"),
            )

        # No clear yes/no — retry or exhaust gracefully
        self.slot_fail("benefits_response")
        if self.get_slot("benefits_response").is_exhausted():
            return self.signal_complete(
                state,
                message="",
                resolved_intents=["delivery_management"],
                context_updates=self._completion_context(
                    state, state.get("delivery_method", ""), False
                ),
                proactive_offer_available=False,
            )

        provider_type = (state.get("provider_type") or "provider").strip()
        retry_msg = random.choice(BENEFITS_OFFER_TEMPLATES).format(provider_type=provider_type)
        retry_result = self.ask_member(state, retry_msg)
        retry_result["awaiting_slot"] = "benefits_response"
        retry_result["provider_list_sent"] = True
        retry_result["benefits_offer_made"] = True
        return retry_result

    async def _proceed_to_dispatch(
        self, state: State, delivery_method: str, confirmed_destination: str
    ) -> dict:
        """Dispatch the provider list then make the benefits offer."""
        if fail := await dispatch_provider_list(self, state, delivery_method, confirmed_destination):
            return fail

        logger.info(
            LOG_LIST_DISPATCHED,
            extra={"method": delivery_method, "dest_tail": confirmed_destination[-4:]},
        )

        timestamp = datetime.now(timezone.utc).isoformat()
        provider_type = (state.get("provider_type") or "provider").strip()
        window_msg = pick(DELIVERY_WINDOW_MSG)
        benefits_msg = random.choice(BENEFITS_OFFER_TEMPLATES).format(provider_type=provider_type)
        combined_msg = f"{window_msg} {benefits_msg}"

        offer_result = self.ask_member(state, combined_msg)
        offer_result["awaiting_slot"] = "benefits_response"
        offer_result["provider_list_sent"] = True
        offer_result["benefits_offer_made"] = True
        offer_result["delivery_method"] = delivery_method
        offer_result["delivery_timestamp"] = timestamp
        if delivery_method == "fax":
            offer_result["fax"] = confirmed_destination
        else:
            offer_result["email"] = confirmed_destination
        return offer_result

    def _ask_contact_confirmation(
        self,
        state: State,
        delivery_method: str,
        fax_on_file: str,
        email_on_file: str,
    ) -> dict:
        """Ask for confirmation of the contact details on file (or collect new ones)."""
        if delivery_method == "fax":
            if fax_on_file:
                msg = random.choice(FAX_READBACK_TEMPLATES).format(fax=fax_on_file)
                result = self.ask_member(state, msg)
                result["awaiting_slot"] = "fax_confirmed"
                result["delivery_method"] = delivery_method
            else:
                result = self.ask_member(state, FAX_UPDATE_PROMPT)
                result["awaiting_slot"] = "fax"
                result["delivery_method"] = delivery_method
        elif delivery_method == "email":
            if email_on_file:
                msg = random.choice(EMAIL_READBACK_TEMPLATES).format(email=email_on_file)
                result = self.ask_member(state, msg)
                result["awaiting_slot"] = "email_confirmed"
                result["delivery_method"] = delivery_method
            else:
                result = self.ask_member(state, EMAIL_UPDATE_PROMPT)
                result["awaiting_slot"] = "email"
                result["delivery_method"] = delivery_method
        else:
            result = self.signal_escalate(
                state, pick(MSG_CONTACT_EXHAUST), reason="invalid_delivery_method"
            )
        return result

    @staticmethod
    def _completion_context(state: State, delivery_method: str, proactive: bool) -> dict:
        return {
            "provider_list_sent": True,
            "delivery_method": delivery_method,
            "delivery_timestamp": state.get("delivery_timestamp", ""),
            "fax": state.get("fax", ""),
            "email": state.get("email", ""),
            "benefits_offer_made": True,
            "proactive_offer_available": proactive,
        }


async def delivery_management_agent(state: State) -> dict:
    logger.info(LOG_ENTERED, extra={"call_intent": state.get("call_intent", "")})
    return await DeliveryManagementAgent.from_state(state).execute(state)
