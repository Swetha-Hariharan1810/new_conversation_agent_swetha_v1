"""
agent.py — NotificationSetupAgent (Sub-Agent 6b)

Flow:
  PHASE 0: Re-entry guard — if notification_channel is set, signal_complete
  PHASE 1: Collect notification_method (sms or email)
           NOTE: If entering from records_coordination, the bridge message
           "I can send status updates to your email or SMS" was already delivered.
           The member's answer is in last_user — skip to extraction immediately.
  PHASE 2: Confirm contact detail on file for chosen method
  PHASE 3: If confirmed → save preference → signal_complete
           If declined → collect new contact → save preference → signal_complete
"""

from __future__ import annotations

import random

from agent.agents.notification_setup.constants import (
    AGENT_NAME,
    EMAIL_READBACK_TEMPLATES,
    EMAIL_UPDATE_PROMPTS,
    LOG_ENTERED,
    LOG_METHOD_COLLECTED,
    LOG_N2_PREFERENCE_SAVED,
    LOG_PREFERENCE_SAVED,
    MSG_CONTACT_EXHAUST,
    MSG_METHOD_EXHAUST,
    MSG_TIMELINE_ANSWER,
    N2_EMAIL_CONFIRM,
    N2_METHOD_ASK,
    N2_PHONE_CONFIRM,
    NOTIFICATION_METHOD_ASK,
    PHONE_READBACK_TEMPLATES,
    PHONE_UPDATE_PROMPTS,
    PREFERENCE_SAVED_TEMPLATES,
    TIMELINE_BRIDGE_TEMPLATES,
)
from agent.agents.notification_setup.handlers import (
    save_notification_preference,
    save_timeline_notification_preference,
)
from agent.agents.notification_setup.llm import extract_notification_decision
from agent.conversation.context import ConversationContext
from agent.core.agent import BaseAgent
from agent.llm.config import get_extraction_llm
from agent.logger import get_logger
from agent.slots.normalizers import (
    normalize_email,
    normalize_notification_method,
    normalize_phone_number,
    normalize_yes_no,
)
from agent.slots.validators import validate_email, validate_phone_number
from agent.state import State
from agent.utils import _last_assistant_msg, _last_user_msg, build_extraction_prompt_extraction, pick

logger = get_logger(__name__)


class NotificationSetupAgent(BaseAgent):
    AGENT_NAME = AGENT_NAME

    async def run(self, state: State) -> dict:  # noqa: C901
        # ── PHASE 0: Re-entry guard ────────────────────────────────────────────
        n1_done = state.get("notification_channel") and state.get("notification_channel") != "not_set"
        n2_done = (
            state.get("claim_timeline_notification_channel")
            and state.get("claim_timeline_notification_channel") != "not_set"
        )
        if n1_done and n2_done:
            return self._signal_done(state)

        messages = list(state.get("messages") or [])
        last_user = _last_user_msg(messages)
        last_agent = _last_assistant_msg(messages)
        current_awaiting = state.get("awaiting_slot", "")
        notification_method = (state.get("notification_channel") or "").strip()

        # ── Determine phase from awaiting_slot ─────────────────────────────────
        if not current_awaiting:
            # Check if entering from records_coordination (bridge already delivered)
            # by checking if Personal Guide was just triggered
            if state.get("personal_guide_outreach_requested") or state.get("upload_link_sent"):
                # Bridge was delivered by records_coordination — member's answer is last_user
                current_awaiting = "notification_method"
            else:
                # Fresh entry — ask the question first
                result = self.ask_member(state, pick(NOTIFICATION_METHOD_ASK))
                result["awaiting_slot"] = "notification_method"
                return result

        state = {**state, "awaiting_slot": current_awaiting}

        # ── LLM extraction ────────────────────────────────────────────────────
        attempts_dict = state.get("slot_attempts") or {}
        current_attempt = attempts_dict.get(current_awaiting, {})
        attempt_count = current_attempt.get("attempt_count", 0) if isinstance(current_attempt, dict) else 0
        confirmed_slots: dict = {}
        # Only carry forward N1's notification_method as a confirmed slot when
        # we are still in the N1 contact-confirmation phases. For N2 extraction
        # (n2_notification_method) the state's notification_channel is N1's value
        # and passing it as confirmed would tell the LLM the slot is already
        # resolved, causing it to produce an acknowledgement turn instead of
        # extracting the member's N2 preference.
        if notification_method and current_awaiting not in ("n2_notification_method",):
            confirmed_slots["notification_method"] = notification_method

        result = await extract_notification_decision(
            get_extraction_llm(),
            build_extraction_prompt_extraction("extraction/notification_setup.md"),
            awaiting_slot=current_awaiting,
            last_agent_message=last_agent,
            last_user_message=last_user,
            confirmed_slots=confirmed_slots,
            attempt=attempt_count,
            recent_messages=messages[-6:],
        )

        if interrupt := await self.run_conversation_guards(state, user_text=last_user, result=result):
            return interrupt

        extracted = (result.extracted or {}) if result else {}

        # ── timeline_question: member response to timeline offer ───────────────
        if current_awaiting == "timeline_question":
            timeline_resp = extracted.get("timeline_response", "")

            if timeline_resp == "question":
                combined = f"{pick(MSG_TIMELINE_ANSWER)}\n\n{pick(N2_METHOD_ASK)}"
                ask_result = self.ask_member(state, combined)
                ask_result["awaiting_slot"] = "n2_notification_method"
                return ask_result

            if timeline_resp in ("yes", "no"):
                ask_result = self.ask_member(state, pick(N2_METHOD_ASK))
                ask_result["awaiting_slot"] = "n2_notification_method"
                return ask_result

            # Ambiguous — proper slot retry pattern
            self.slot_fail("timeline_question")
            if self.get_slot("timeline_question").is_exhausted():
                return self.signal_escalate(
                    state,
                    pick(MSG_METHOD_EXHAUST),
                    reason="timeline_question_exhausted",
                )
            ctx = ConversationContext.from_state(state)
            msg = await self._generate_slot_retry_response(
                state, "timeline_question", ctx, messages, guard="RETRY"
            )
            retry = self.ask_member(state, msg)
            retry["awaiting_slot"] = "timeline_question"
            return retry

        # ── PHASE 1: Collect notification_method ──────────────────────────────
        if current_awaiting == "notification_method":
            raw_method = extracted.get("notification_method", "")
            method = normalize_notification_method(raw_method) if raw_method else ""

            if method == "sms":
                logger.info(LOG_METHOD_COLLECTED, extra={"method": "sms"})
                return self._ask_contact_confirmation(state, "sms")
            if method == "email":
                logger.info(LOG_METHOD_COLLECTED, extra={"method": "email"})
                return self._ask_contact_confirmation(state, "email")

            self.slot_fail("notification_method")
            if self.get_slot("notification_method").is_exhausted():
                return self.signal_escalate(
                    state, pick(MSG_METHOD_EXHAUST), reason="notification_method_exhausted"
                )
            ctx = ConversationContext.from_state(state)
            msg = await self._generate_slot_retry_response(state, "notification_method", ctx, messages)
            retry = self.ask_member(state, msg)
            retry["awaiting_slot"] = "notification_method"
            return retry

        # ── PHASE 2: Phone confirmation (SMS path) ─────────────────────────────
        if current_awaiting == "phone_confirmed":
            new_phone_raw = extracted.get("phone", "")
            contact_conf_raw = extracted.get("contact_confirmed", "")
            phone_on_file = (state.get("phone_number") or "").strip()
            pending_phone = (state.get("pending_phone") or "").strip()

            contact_conf = normalize_yes_no(contact_conf_raw) if contact_conf_raw else ""
            # Extraction contract: a replacement phone and contact_confirmed are
            # mutually exclusive. If both arrive, phone is an echo of the Confirmed:
            # context line — discard it so the yes/no is honored.
            if contact_conf in ("yes", "no"):
                new_phone_raw = ""

            if new_phone_raw:
                normalized = normalize_phone_number(str(new_phone_raw))
                if normalized and validate_phone_number(normalized).valid:
                    if normalized == normalize_phone_number(phone_on_file):
                        # Same number we have on file — the preference write must
                        # still happen; only NEW contact values are deferred.
                        done = await self._save_and_complete(state, "sms", phone_on_file)
                        done["pending_phone"] = ""
                        return done
                    confirm = self.ask_member(
                        state,
                        f"Just to be sure I have it right — your phone number is "
                        f"{normalized[:3]}-{normalized[3:6]}-{normalized[6:]}, correct?",
                    )
                    confirm["awaiting_slot"] = "phone_confirmed"
                    confirm["pending_phone"] = normalized
                    confirm["notification_channel"] = "sms"
                    return confirm
                ask_result = self.ask_member(state, pick(PHONE_UPDATE_PROMPTS))
                ask_result["awaiting_slot"] = "phone"
                ask_result["pending_phone"] = ""
                return ask_result

            if contact_conf == "yes":
                done = await self._save_and_complete(state, "sms", pending_phone or phone_on_file)
                done["pending_phone"] = ""
                return done
            if contact_conf == "no":
                ask_result = self.ask_member(state, pick(PHONE_UPDATE_PROMPTS))
                ask_result["awaiting_slot"] = "phone"
                ask_result["pending_phone"] = ""
                return ask_result

            self.slot_fail("phone_confirmed")
            if self.get_slot("phone_confirmed").is_exhausted():
                return self.signal_escalate(
                    state, pick(MSG_CONTACT_EXHAUST), reason="phone_confirmed_exhausted_in_notification"
                )
            ctx = ConversationContext.from_state(state)
            retry_msg = await self._generate_slot_retry_response(state, "phone_confirmed", ctx, messages)
            retry_result = self.ask_member(state, retry_msg)
            retry_result["awaiting_slot"] = "phone_confirmed"
            retry_result["notification_channel"] = "sms"
            return retry_result

        # ── PHASE 2: Phone update ──────────────────────────────────────────────
        if current_awaiting == "phone":
            phone_raw = extracted.get("phone", "")
            if phone_raw:
                normalized = normalize_phone_number(str(phone_raw))
                if normalized and validate_phone_number(normalized).valid:
                    # Hold the new phone as pending until the member confirms
                    confirm = self.ask_member(
                        state,
                        f"Just to be sure I have it right — your phone number is "
                        f"{normalized[:3]}-{normalized[3:6]}-{normalized[6:]}, correct?",
                    )
                    confirm["awaiting_slot"] = "phone_confirmed"
                    confirm["pending_phone"] = normalized
                    confirm["notification_channel"] = "sms"
                    return confirm
            self.slot_fail("phone")
            if self.get_slot("phone").is_exhausted():
                return self.signal_escalate(
                    state, pick(MSG_CONTACT_EXHAUST), reason="phone_update_exhausted_in_notification"
                )
            ctx = ConversationContext.from_state(state)
            msg = await self._generate_slot_retry_response(state, "phone", ctx, messages)
            ask_result = self.ask_member(state, msg)
            ask_result["awaiting_slot"] = "phone"
            return ask_result

        # ── PHASE 2: Email confirmation (email path) ───────────────────────────
        if current_awaiting == "email_confirmed":
            new_email_raw = extracted.get("email", "")
            contact_conf_raw = extracted.get("contact_confirmed", "")
            email_on_file = (state.get("email") or "").strip()
            pending_email = (state.get("pending_email") or "").strip()

            contact_conf = normalize_yes_no(contact_conf_raw) if contact_conf_raw else ""
            # Extraction contract: a replacement email and contact_confirmed are
            # mutually exclusive. If both arrive, email is an echo of the Confirmed:
            # context line — discard it so the yes/no is honored.
            if contact_conf in ("yes", "no"):
                new_email_raw = ""

            if new_email_raw:
                normalized = normalize_email(str(new_email_raw))
                if normalized and validate_email(normalized).valid:
                    if normalized == normalize_email(email_on_file):
                        # Same email we have on file — the preference write must
                        # still happen; only NEW contact values are deferred.
                        done = await self._save_and_complete(state, "email", email_on_file)
                        done["pending_email"] = ""
                        return done
                    display_email = normalized.replace("@", " at ")
                    confirm = self.ask_member(
                        state,
                        f"Just to be sure I have it right — your email address is "
                        f"{display_email}, correct?",
                    )
                    confirm["awaiting_slot"] = "email_confirmed"
                    confirm["pending_email"] = normalized
                    confirm["notification_channel"] = "email"
                    return confirm
                ask_result = self.ask_member(state, pick(EMAIL_UPDATE_PROMPTS))
                ask_result["awaiting_slot"] = "email"
                ask_result["pending_email"] = ""
                return ask_result

            if contact_conf == "yes":
                done = await self._save_and_complete(state, "email", pending_email or email_on_file)
                done["pending_email"] = ""
                return done
            if contact_conf == "no":
                ask_result = self.ask_member(state, pick(EMAIL_UPDATE_PROMPTS))
                ask_result["awaiting_slot"] = "email"
                ask_result["pending_email"] = ""
                return ask_result

            self.slot_fail("email_confirmed")
            if self.get_slot("email_confirmed").is_exhausted():
                return self.signal_escalate(
                    state, pick(MSG_CONTACT_EXHAUST), reason="email_confirmed_exhausted_in_notification"
                )
            from agent.llm.response_generator import generate_recovery_message

            display_email = (pending_email or email_on_file).replace("@", " at ")
            ctx = ConversationContext.from_state(state)
            retry_msg = await generate_recovery_message(
                slot_name="email_confirmed",
                attempt=self.get_slot("email_confirmed").attempt_count,
                guard="RETRY",
                last_messages=messages[-4:],
                slot_label_override=f"whether the email address"
                f"{display_email} is correct for notifications (yes or no)",
                caller_name=ctx.caller_first_name,
                confirmed_slots=dict.fromkeys(ctx.confirmed_slots, "confirmed"),
                user_utterance=_last_user_msg(messages),
            )
            retry_result = self.ask_member(state, retry_msg)
            retry_result["awaiting_slot"] = "email_confirmed"
            retry_result["notification_channel"] = "email"
            return retry_result

        # ── PHASE 2: Email update ─────────────────────────────────────────────
        if current_awaiting == "email":
            email_raw = extracted.get("email", "")
            if email_raw:
                normalized = normalize_email(str(email_raw))
                if normalized and validate_email(normalized).valid:
                    # Hold the new email as pending until the member confirms
                    display_email = normalized.replace("@", " at ")
                    confirm = self.ask_member(
                        state,
                        f"Just to be sure I have it right — your email address is "
                        f"{display_email}, correct?",
                    )
                    confirm["awaiting_slot"] = "email_confirmed"
                    confirm["pending_email"] = normalized
                    confirm["notification_channel"] = "email"
                    return confirm
            self.slot_fail("email")
            if self.get_slot("email").is_exhausted():
                return self.signal_escalate(
                    state, pick(MSG_CONTACT_EXHAUST), reason="email_update_exhausted_in_notification"
                )
            ctx = ConversationContext.from_state(state)
            msg = await self._generate_slot_retry_response(state, "email", ctx, messages)
            ask_result = self.ask_member(state, msg)
            ask_result["awaiting_slot"] = "email"
            return ask_result

        # ── PHASE: n2_notification_method — second notification preference ──────
        if current_awaiting == "n2_notification_method":
            raw_method = extracted.get("notification_method", "")
            method = normalize_notification_method(raw_method) if raw_method else ""

            if method == "sms":
                phone_on_file = (state.get("phone_number") or "").strip()
                if phone_on_file:
                    digits = "".join(c for c in phone_on_file if c.isdigit())
                    formatted = (
                        f"{digits[:3]}-{digits[3:6]}-{digits[6:]}" if len(digits) == 10 else phone_on_file
                    )
                    confirm_msg = random.choice(N2_PHONE_CONFIRM).format(phone=formatted)
                    return await self._n2_save_and_complete(state, "sms", phone_on_file, confirm_msg)
                # No phone on file — fall through to exhaust/escalate
            elif method == "email":
                email_on_file = (state.get("email") or "").strip()
                if email_on_file:
                    # Use the raw email from state — never replace @ with "at".
                    # The @ form is what the member expects to hear confirmed.
                    confirm_msg = random.choice(N2_EMAIL_CONFIRM).format(email=email_on_file)
                    return await self._n2_save_and_complete(state, "email", email_on_file, confirm_msg)
                # No email on file — fall through to exhaust/escalate

            opted_out = extracted.get("notification_opted_out", "")
            if opted_out == "yes":
                logger.info(LOG_N2_PREFERENCE_SAVED + ": opted out")
                from agent.agents.follow_up.constants import MSG_FOLLOW_UP_ASK

                handoff = pick(MSG_FOLLOW_UP_ASK)
                result = self.ask_member(state, handoff)
                result["next_node"] = "follow_up_agent"
                result.update(self._n2_completion_context(state, "not_set", ""))
                result["last_agent_signal"] = {
                    "status": "complete",
                    "resolved_intents": ["notification_setup"],
                    "closure_requested": False,
                    "context_updates": {},
                    "proactive_offer_available": False,
                    "escalation_reason": None,
                    "reasoning": "notification_setup_agent: n2 opted out by member",
                }
                return result

            self.slot_fail("n2_notification_method")
            if self.get_slot("n2_notification_method").is_exhausted():
                return self.signal_escalate(
                    state, pick(MSG_METHOD_EXHAUST), reason="n2_notification_method_exhausted"
                )
            ctx = ConversationContext.from_state(state)
            msg = await self._generate_slot_retry_response(state, "n2_notification_method", ctx, messages)
            retry = self.ask_member(state, msg)
            retry["awaiting_slot"] = "n2_notification_method"
            return retry

        # Fallback
        result = self.ask_member(state, pick(NOTIFICATION_METHOD_ASK))
        result["awaiting_slot"] = "notification_method"
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _ask_contact_confirmation(self, state: State, method: str) -> dict:
        """Ask the member to confirm the contact detail on file for their chosen method."""
        if method == "sms":
            phone_on_file = (state.get("phone_number") or "").strip()
            if phone_on_file:
                digits = "".join(c for c in phone_on_file if c.isdigit())
                formatted = f"{digits[:3]}-{digits[3:6]}-{digits[6:]}" if len(digits) == 10 else phone_on_file
                msg = random.choice(PHONE_READBACK_TEMPLATES).format(phone=formatted)
                result = self.ask_member(state, msg)
                result["awaiting_slot"] = "phone_confirmed"
                result["notification_channel"] = "sms"
            else:
                result = self.ask_member(state, pick(PHONE_UPDATE_PROMPTS))
                result["awaiting_slot"] = "phone"
                result["notification_channel"] = "sms"
        else:  # email
            email_on_file = (state.get("email") or "").strip()
            if email_on_file:
                display_email = email_on_file.replace("@", " at ")
                msg = random.choice(EMAIL_READBACK_TEMPLATES).format(email=display_email)
                result = self.ask_member(state, msg)
                result["awaiting_slot"] = "email_confirmed"
                result["notification_channel"] = "email"
            else:
                result = self.ask_member(state, pick(EMAIL_UPDATE_PROMPTS))
                result["awaiting_slot"] = "email"
                result["notification_channel"] = "email"
        return result

    async def _save_and_complete(self, state: State, method: str, contact: str) -> dict:
        """Save N1 preference to Salesforce, deliver timeline bridge, pause for member response."""
        if fail := await save_notification_preference(self, state, method, contact):
            return fail

        logger.info(LOG_PREFERENCE_SAVED, extra={"method": method})

        display_contact = contact.replace("@", " at ") if method == "email" else contact
        confirm_msg = random.choice(PREFERENCE_SAVED_TEMPLATES).format(method=method, contact=display_contact)
        timeline_bridge = pick(TIMELINE_BRIDGE_TEMPLATES)
        combined = f"{confirm_msg} {timeline_bridge}"

        # Pause to let the member ask timeline questions; N2 ask follows in timeline_question phase.
        result = self.ask_member(state, combined)
        result["notification_channel"] = method
        result["claim_notification_contact"] = contact
        result["awaiting_slot"] = "timeline_question"
        return result

    async def _n2_save_and_complete(self, state: State, method: str, contact: str, confirm_msg: str) -> dict:
        """Save N2 (timeline) preference, deliver bridge message, and pause for user.

        Sends the N2 confirmation + follow-up opening question as a single AI turn
        via ask_member (is_interrupt=True), so the graph pauses and waits for the
        user's actual follow-up response before routing to follow_up_agent.
        This prevents the N2 method answer (e.g. "email them to me") from being
        misclassified as a follow-up intent by follow_up_agent.
        """
        if fail := await save_timeline_notification_preference(self, state, method, contact):
            return fail

        logger.info(LOG_N2_PREFERENCE_SAVED, extra={"method": method})

        from agent.agents.follow_up.constants import MSG_FOLLOW_UP_ASK

        handoff = f"{confirm_msg}\n{pick(MSG_FOLLOW_UP_ASK)}"

        # Use ask_member so the graph pauses here and the next human turn
        # is the true follow-up response — not the N2 method answer.
        result = self.ask_member(state, handoff)
        result["next_node"] = "follow_up_agent"
        result.update(self._n2_completion_context(state, method, contact))
        # Signal notification_setup is done so orchestrator fast-path doesn't re-enter it
        result["last_agent_signal"] = {
            "status": "complete",
            "resolved_intents": ["notification_setup"],
            "closure_requested": False,
            "context_updates": {},
            "proactive_offer_available": False,
            "escalation_reason": None,
            "reasoning": "notification_setup_agent: n2 complete — waiting for follow-up",
        }
        return result

    @staticmethod
    def _n2_completion_context(state: State, method: str, contact: str) -> dict:
        return {
            "notification_channel": state.get("notification_channel", ""),
            "claim_notification_contact": state.get("claim_notification_contact", ""),
            "claim_timeline_notification_channel": method,
            "claim_timeline_notification_contact": contact,
            "claim_flow_complete": True,
        }

    def _signal_done(self, state: State) -> dict:
        return self.signal_complete(
            state,
            message="",
            resolved_intents=["notification_setup"],
            context_updates={
                "notification_channel": state.get("notification_channel", ""),
                "claim_notification_contact": state.get("claim_notification_contact", ""),
                "claim_timeline_notification_channel": state.get("claim_timeline_notification_channel", ""),
                "claim_timeline_notification_contact": state.get("claim_timeline_notification_contact", ""),
                "claim_flow_complete": True,
            },
            closure_requested=False,
        )


async def notification_setup_agent(state: State) -> dict:
    logger.info(LOG_ENTERED, extra={"call_intent": state.get("call_intent", "")})
    return await NotificationSetupAgent.from_state(state).execute(state)
