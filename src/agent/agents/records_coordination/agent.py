"""
agent.py — RecordsCoordinationAgent (Sub-Agent 6a)

Four-branch decision tree for obtaining medical records:
  Branch A (member_upload):  generate upload link → send to email → confirm sent
  Branch B (doctor_direct):  acknowledge → offer upload link → if accepted send it
                              → offer Personal Guide
  Branch C (personal_guide): with explicit consent → trigger SF workflow
  Branch D (decline):        signal_escalate after all options exhausted

State machine phases (tracked via awaiting_slot):
  ""                    → initial turn: present Member upload/doctor option
  "upload_method"       → waiting for member's initial records preference
  "upload_consent"      → waiting for yes/no to upload link offer
  "email_confirmed"     → waiting for email confirmation before sending link
  "email"               → waiting for corrected email address
  "personal_guide_consent" → waiting for explicit consent to trigger Personal Guide
"""

from __future__ import annotations

import random

from agent.agents.records_coordination.constants import (
    AGENT_NAME,
    EMAIL_READBACK_FOR_UPLOAD,
    LOG_DOCTOR_DIRECT,
    LOG_ENTERED,
    LOG_GUIDE_TRIGGERED,
    LOG_UPLOAD_LINK_SENT,
    MSG_DECLINE_ESCALATE,
    MSG_DOCTOR_DIRECT_ACK,
    MSG_EMAIL_UPDATE_PROMPT,
    MSG_GUIDE_SCHEDULED,
    MSG_NOTIFICATION_BRIDGE,
    MSG_PERSONAL_GUIDE_OFFER,
    MSG_UPLOAD_OFFER,
    MSG_UPLOAD_SENT,
)
from agent.agents.records_coordination.handlers import dispatch_personal_guide, dispatch_upload_link
from agent.agents.records_coordination.llm import extract_records_decision
from agent.conversation.context import ConversationContext
from agent.core.agent import BaseAgent
from agent.llm.config import get_extraction_llm
from agent.logger import get_logger
from agent.slots.normalizers import normalize_email, normalize_yes_no
from agent.slots.validators import validate_email
from agent.state import State
from agent.utils import _last_assistant_msg, _last_user_msg, build_extraction_prompt_extraction, pick

logger = get_logger(__name__)

_MAX_GUIDE_CONSENT_ATTEMPTS = 3


class RecordsCoordinationAgent(BaseAgent):
    AGENT_NAME = AGENT_NAME

    async def run(self, state: State) -> dict:  # noqa: C901
        # ── Re-entry guard ────────────────────────────────────────────────────
        if state.get("records_branch_taken"):
            return self._signal_done(state)

        messages = list(state.get("messages") or [])
        last_user = _last_user_msg(messages)
        last_agent = _last_assistant_msg(messages)
        current_awaiting = state.get("awaiting_slot", "")

        # ── First entry fast-path ─────────────────────────────────────────────
        # The ClaimAdjustmentAgent already asked "Can you send it over?"
        # and the member's answer is now last_user. We go straight to extraction.
        if not current_awaiting:
            current_awaiting = "upload_method"
            state = {**state, "awaiting_slot": current_awaiting}

        # ── LLM extraction ────────────────────────────────────────────────────
        attempts_dict = state.get("slot_attempts") or {}
        current_attempt = attempts_dict.get(current_awaiting, {})
        attempt_count = current_attempt.get("attempt_count", 0) if isinstance(current_attempt, dict) else 0

        result = await extract_records_decision(
            get_extraction_llm(),
            build_extraction_prompt_extraction("extraction/records_coordination.md"),
            awaiting_slot=current_awaiting,
            last_agent_message=last_agent,
            last_user_message=last_user,
            confirmed_slots={},
            attempt=attempt_count,
            recent_messages=messages[-6:],
        )

        if interrupt := await self.run_conversation_guards(state, user_text=last_user, result=result):
            return interrupt

        extracted = (result.extracted or {}) if result else {}

        # ── BRANCH ROUTING ────────────────────────────────────────────────────

        # Phase: upload_method — initial intent from member
        if current_awaiting == "upload_method":
            upload_method = extracted.get("upload_method", "")

            if upload_method == "member_upload":
                # Member wants to upload themselves — offer the link
                offer_result = self.ask_member(state, pick(MSG_UPLOAD_OFFER))
                offer_result["awaiting_slot"] = "upload_consent"
                return offer_result

            if upload_method == "doctor_direct":
                # Doctor will send it — acknowledge, then offer upload link anyway
                logger.info(LOG_DOCTOR_DIRECT)
                ack = pick(MSG_DOCTOR_DIRECT_ACK)
                offer = pick(MSG_UPLOAD_OFFER)
                combined = f"{ack}\n\n{offer}"
                offer_result = self.ask_member(state, combined)
                offer_result["awaiting_slot"] = "upload_consent"
                return offer_result

            if upload_method == "personal_guide":
                # Member immediately wants Personal Guide
                return await self._handle_guide_consent_ask(state)

            if upload_method == "decline":
                # Member declined this step entirely → escalate
                return self.signal_escalate(
                    state,
                    pick(MSG_DECLINE_ESCALATE),
                    reason="member_declined_all_records_options",
                )

            # No clear extraction — re-ask (retry once before moving on)
            self.slot_fail("upload_method")
            if self.get_slot("upload_method").is_exhausted():
                return self.signal_escalate(
                    state,
                    pick(MSG_DECLINE_ESCALATE),
                    reason="records_upload_method_exhausted",
                )
            ctx = ConversationContext.from_state(state)
            msg = await self._generate_slot_retry_response(state, "upload_method", ctx, messages)
            retry = self.ask_member(state, msg)
            retry["awaiting_slot"] = "upload_method"
            return retry

        # Phase: upload_consent — did member agree to receive the link?
        if current_awaiting == "upload_consent":
            upload_consent = normalize_yes_no(extracted.get("upload_consent", ""))

            if upload_consent == "yes":
                # Confirm email before sending
                email_on_file = (state.get("email") or "").strip()
                if email_on_file:
                    display_email = email_on_file.replace("@", " at ")
                    msg = random.choice(EMAIL_READBACK_FOR_UPLOAD).format(email=display_email)
                    confirm_result = self.ask_member(state, msg)
                    confirm_result["awaiting_slot"] = "email_confirmed"
                    return confirm_result
                else:
                    ask_result = self.ask_member(state, pick(MSG_EMAIL_UPDATE_PROMPT))
                    ask_result["awaiting_slot"] = "email"
                    return ask_result

            if upload_consent == "no":
                # Member declined link — offer Personal Guide
                return await self._handle_guide_consent_ask(state)

            # Ambiguous — retry
            self.slot_fail("upload_consent")
            if self.get_slot("upload_consent").is_exhausted():
                return await self._handle_guide_consent_ask(state)
            ctx = ConversationContext.from_state(state)
            msg = await self._generate_slot_retry_response(state, "upload_consent", ctx, messages)
            retry = self.ask_member(state, msg)
            retry["awaiting_slot"] = "upload_consent"
            return retry

        # Phase: email_confirmed — is email on file correct?
        if current_awaiting == "email_confirmed":
            new_email_raw = extracted.get("email", "")
            contact_conf_raw = extracted.get("email_confirmed", extracted.get("contact_confirmed", ""))
            email_on_file = (state.get("email") or "").strip()

            if new_email_raw:
                normalized = normalize_email(str(new_email_raw))
                if normalized and validate_email(normalized).valid:
                    return await self._send_link_and_proceed(state, normalized)
                ask_result = self.ask_member(state, pick(MSG_EMAIL_UPDATE_PROMPT))
                ask_result["awaiting_slot"] = "email"
                return ask_result

            contact_conf = normalize_yes_no(contact_conf_raw) if contact_conf_raw else ""
            if contact_conf == "yes":
                return await self._send_link_and_proceed(state, email_on_file)
            if contact_conf == "no":
                ask_result = self.ask_member(state, pick(MSG_EMAIL_UPDATE_PROMPT))
                ask_result["awaiting_slot"] = "email"
                return ask_result

            self.slot_fail("email_confirmed")
            if self.get_slot("email_confirmed").is_exhausted():
                return self.signal_escalate(
                    state,
                    "I wasn't able to confirm your email. Let me connect you with a representative.",
                    reason="email_confirmed_exhausted_in_records",
                )
            from agent.llm.response_generator import generate_recovery_message

            display_email = email_on_file.replace("@", " at ")
            ctx = ConversationContext.from_state(state)
            retry_msg = await generate_recovery_message(
                slot_name="email_confirmed",
                attempt=self.get_slot("email_confirmed").attempt_count,
                guard="RETRY",
                last_messages=messages[-4:],
                slot_label_override=f"whether the email address"
                f" {display_email} is correct for sending the upload link (yes or no)",
                caller_name=ctx.caller_first_name,
                confirmed_slots=dict.fromkeys(ctx.confirmed_slots, "confirmed"),
                user_utterance=_last_user_msg(messages),
            )
            retry_result = self.ask_member(state, retry_msg)
            retry_result["awaiting_slot"] = "email_confirmed"
            return retry_result

        # Phase: email — collecting a new / corrected email
        if current_awaiting == "email":
            new_email_raw = extracted.get("email", "")
            if new_email_raw:
                normalized = normalize_email(str(new_email_raw))
                if normalized and validate_email(normalized).valid:
                    return await self._send_link_and_proceed(state, normalized)
            self.slot_fail("email")
            if self.get_slot("email").is_exhausted():
                return self.signal_escalate(
                    state,
                    "I wasn't able to capture your email after a few tries. "
                    "Let me connect you with a representative.",
                    reason="email_exhausted_in_records",
                )
            ctx = ConversationContext.from_state(state)
            msg = await self._generate_slot_retry_response(state, "email", ctx, messages)
            ask_result = self.ask_member(state, msg)
            ask_result["awaiting_slot"] = "email"
            return ask_result

        # Phase: personal_guide_consent — explicit consent required
        if current_awaiting == "personal_guide_consent":
            guide_consent = normalize_yes_no(extracted.get("personal_guide_consent", ""))

            if guide_consent == "yes":
                return await self._trigger_guide_and_proceed(state)

            if guide_consent == "no":
                return self.signal_escalate(
                    state,
                    pick(MSG_DECLINE_ESCALATE),
                    reason="member_declined_personal_guide",
                )

            # Ambiguous
            self.slot_fail("personal_guide_consent")
            if self.get_slot("personal_guide_consent").is_exhausted():
                return self.signal_escalate(
                    state,
                    pick(MSG_DECLINE_ESCALATE),
                    reason="personal_guide_consent_exhausted",
                )
            ctx = ConversationContext.from_state(state)
            msg = await self._generate_slot_retry_response(state, "personal_guide_consent", ctx, messages)
            retry = self.ask_member(state, msg)
            retry["awaiting_slot"] = "personal_guide_consent"
            return retry

        # Fallback re-ask
        fallback = self.ask_member(state, pick(MSG_UPLOAD_OFFER))
        fallback["awaiting_slot"] = "upload_consent"
        return fallback

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    async def _handle_guide_consent_ask(self, state: State) -> dict:
        """Offer Personal Guide outreach and wait for explicit consent."""
        result = self.ask_member(state, pick(MSG_PERSONAL_GUIDE_OFFER))
        result["awaiting_slot"] = "personal_guide_consent"
        return result

    async def _send_link_and_proceed(self, state: State, email: str) -> dict:
        """
        Dispatch the upload link to email, then offer Personal Guide outreach.
        Corresponds to Branch A (and the B flow that goes through upload).
        """
        if fail := await dispatch_upload_link(self, state, email):
            return fail

        logger.info(LOG_UPLOAD_LINK_SENT, extra={"email_tail": email[-8:]})

        # upload_link_sent=True means the upload_method decision is resolved —
        # mark the slot confirmed so no dangling unconfirmed slot remains.
        self.get_slot("upload_method").record_attempt("upload_link", success=True)

        sent_msg = pick(MSG_UPLOAD_SENT)
        guide_msg = pick(MSG_PERSONAL_GUIDE_OFFER)
        combined = f"{sent_msg}\n\n{guide_msg}"

        result = self.ask_member(state, combined)
        result["upload_link_sent"] = True
        result["email"] = email
        result["awaiting_slot"] = "personal_guide_consent"
        return result

    async def _trigger_guide_and_proceed(self, state: State) -> dict:
        """
        Trigger Personal Guide workflow and transition to Notification Setup.
        Corresponds to Branch C completion.
        """
        if fail := await dispatch_personal_guide(self, state):
            return fail

        logger.info(LOG_GUIDE_TRIGGERED)

        scheduled_msg = pick(MSG_GUIDE_SCHEDULED)
        notification_bridge = pick(MSG_NOTIFICATION_BRIDGE)
        combined = f"{scheduled_msg}\n\n{notification_bridge}"

        result = self.ask_member(state, combined)
        result["personal_guide_outreach_requested"] = True
        result["records_branch_taken"] = "personal_guide"
        result["next_node"] = "notification_setup_agent"
        result["awaiting_slot"] = ""
        return result

    def _signal_done(self, state: State) -> dict:
        return self.signal_complete(
            state,
            message="",
            resolved_intents=["records_coordination"],
            context_updates={
                "records_branch_taken": state.get("records_branch_taken", ""),
                "upload_link_sent": state.get("upload_link_sent", False),
                "personal_guide_outreach_requested": state.get("personal_guide_outreach_requested", False),
            },
        )


async def records_coordination_agent(state: State) -> dict:
    logger.info(LOG_ENTERED, extra={"call_intent": state.get("call_intent", "")})
    return await RecordsCoordinationAgent.from_state(state).execute(state)
