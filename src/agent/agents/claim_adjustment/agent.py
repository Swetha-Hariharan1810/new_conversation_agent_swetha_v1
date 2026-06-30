"""
agent.py — ClaimAdjustmentAgent

Flow:
  PHASE 0: Re-entry guard — if claim_flow_complete, signal_complete
  PHASE 1: Collect reference_number directly (first-entry fast-path: skip LLM)
  PHASE 2: Salesforce lookup — find_adjustment(reference_number, member_id)
           Not found → signal_escalate with MSG_REF_NOT_FOUND
  PHASE 3: Report status and last_update_date to member
  PHASE 4: If records_required → signal_complete to route to records_coordination_agent
           If not records_required → signal_complete to route to notification_setup_agent
"""

from __future__ import annotations

import random

from agent.agents.claim_adjustment.constants import (
    AGENT_NAME,
    LOG_ENTERED,
    LOG_REF_COLLECTED,
    LOG_STATUS_REPORTED,
    MSG_RECORDS_NEEDED,
    MSG_REF_EXHAUST,
    MSG_REF_NOT_FOUND,
    MSG_REF_NOT_FOUND_RETRY,
    REFERENCE_NUMBER_BRIDGE_MSGS,
    STATUS_REPORT_TEMPLATES,
)
from agent.agents.claim_adjustment.handlers import lookup_adjustment
from agent.agents.claim_adjustment.llm import extract_claim_adjustment_decision
from agent.agents.verification.constants import MAX_LOOKUP_ATTEMPTS
from agent.conversation.context import ConversationContext
from agent.core.agent import BaseAgent
from agent.llm.config import get_extraction_llm
from agent.logger import get_logger
from agent.orchestration.invalidation import clear_dirty
from agent.slots.normalizers import normalize_reference_number
from agent.slots.validators import validate_reference_number
from agent.state import State
from agent.utils import (
    _last_assistant_msg,
    _last_user_msg,
    build_extraction_prompt_extraction,
    detect_cannot_provide,  # ← NEW IMPORT
    pick,
)

logger = get_logger(__name__)

# Escalation message when member says they don't have the reference number.
# Consistent with the tone used across the codebase for cannot-provide cases.
_MSG_REF_CANNOT_PROVIDE = ["No problem at all — let me connect you with a representative "]


class ClaimAdjustmentAgent(BaseAgent):
    AGENT_NAME = AGENT_NAME

    async def run(self, state: State) -> dict:  # noqa: C901
        # ── PHASE 0: Re-entry guard ────────────────────────────────────────────
        if state.get("claim_flow_complete"):
            return self.signal_complete(
                state,
                message="",
                resolved_intents=["claim_services"],
                context_updates=self._completion_context(state),
            )

        messages = list(state.get("messages") or [])
        last_user = _last_user_msg(messages)
        last_agent = _last_assistant_msg(messages)

        raw_awaiting = state.get("awaiting_slot", "")
        reference_number = (state.get("reference_number") or "").strip()

        # ── PHASE 1 FAST PATH: first entry — ask without LLM ─────────────────
        if not raw_awaiting and not reference_number:
            interrupt = self.ask_member(state, pick(REFERENCE_NUMBER_BRIDGE_MSGS))
            interrupt["awaiting_slot"] = "reference_number"
            return interrupt

        # ── PHASE 1: Collect reference_number directly (no pipeline) ──────────
        if not reference_number:
            current_awaiting = raw_awaiting or "reference_number"
            state = {**state, "awaiting_slot": current_awaiting}

            # ── CHANGE: cannot-provide check BEFORE any LLM call ─────────────
            # "No, I don't have it" / "I lost the letter" / "I never received
            # a reference number" → escalate immediately on the first turn.
            # This runs before the extraction LLM so there is zero latency cost.
            if detect_cannot_provide(last_user):
                logger.info(
                    "claim_adjustment_agent: cannot-provide detected for reference_number — "
                    "escalating immediately"
                )
                return self.signal_escalate(
                    state,
                    pick(_MSG_REF_CANNOT_PROVIDE),
                    reason="reference_number_cannot_provide",
                )
            # ── END CHANGE ────────────────────────────────────────────────────

            attempts_dict = state.get("slot_attempts") or {}
            current_attempt = attempts_dict.get(current_awaiting, {})
            attempt_count = (
                current_attempt.get("attempt_count", 0) if isinstance(current_attempt, dict) else 0
            )

            result = await extract_claim_adjustment_decision(
                get_extraction_llm(),
                build_extraction_prompt_extraction("extraction/claim_adjustment.md"),
                awaiting_slot=current_awaiting,
                last_agent_message=last_agent,
                last_user_message=last_user,
                confirmed_slots={},
                attempt=attempt_count,
                recent_messages=messages[-6:],
            )

            if interrupt := await self.run_conversation_guards(state, user_text=last_user, result=result):
                # Guard fired (transfer, abuse, offtopic, etc.).
                # Still count this as a failed slot attempt so the exhaustion
                # counter advances correctly across turns.
                self.slot_fail("reference_number")
                # Patch the guard's interrupt with the updated slot_attempts.
                # (ask_member was already called inside the guard path, so we
                # must overwrite slot_attempts in the already-built dict.)
                interrupt["slot_attempts"] = self.slots_dict()
                if self.get_slot("reference_number").is_exhausted():
                    return self.signal_escalate(
                        state,
                        pick(MSG_REF_EXHAUST),
                        reason="reference_number_exhausted",
                    )
                return interrupt

            extracted_raw = (result.extracted or {}).get("reference_number", "") if result else ""
            normalized = normalize_reference_number(extracted_raw) if extracted_raw else ""

            if normalized and validate_reference_number(normalized).valid:
                reference_number = normalized
                self.slot_ok("reference_number", reference_number)
                logger.info(LOG_REF_COLLECTED)
                state = {**state, "reference_number": reference_number}
            else:
                # Use core slot_fail directly — ensures slot_attempts is captured
                # in the next ask_member call (via slots_dict()), matching the
                # pattern used by every other agent in the codebase
                self.slot_fail("reference_number")
                if self.get_slot("reference_number").is_exhausted():
                    return self.signal_escalate(
                        state,
                        pick(MSG_REF_EXHAUST),
                        reason="reference_number_exhausted",
                    )
                msg = await self._generate_slot_retry_response(
                    state,
                    "reference_number",
                    ConversationContext.from_state(state),
                    messages,
                    extracted_this_turn=normalized if normalized else extracted_raw,
                    guard="RETRY",
                )
                retry = self.ask_member(state, msg)
                retry["awaiting_slot"] = "reference_number"
                return retry

        # ── PHASE 2: Salesforce lookup ─────────────────────────────────────────
        adjustment_record = None
        if not state.get("claim_status"):
            adjustment_record, interrupt = await lookup_adjustment(self, state)
            if interrupt:
                return interrupt
            if not adjustment_record:
                # SF returned no match for this reference number.
                # Allow one retry (MAX_LOOKUP_ATTEMPTS=2 total attempts, same as
                # verification_agent) before escalating.
                if escalation := self.guard_loop_limit(
                    state,
                    "ref_lookup_fail",
                    MAX_LOOKUP_ATTEMPTS,
                    escalate_message=pick(MSG_REF_NOT_FOUND),
                    escalate_reason="adjustment_reference_not_found",
                ):
                    return escalation
                # Under the limit — clear reference_number so PHASE 1 re-collects it,
                # tell the member this number wasn't found and ask for another.
                msg = pick(MSG_REF_NOT_FOUND_RETRY)
                retry = self.ask_member(state, msg)
                retry["reference_number"] = ""  # force PHASE 1 to re-run
                retry["awaiting_slot"] = "reference_number"
                return retry

        # ── PHASE 3: Report status ─────────────────────────────────────────────
        if not state.get("claim_status") and adjustment_record:
            claim_status = adjustment_record.get("claim_status", "open for Review from our adjustment team")
            last_update_date = adjustment_record.get("claim_update_date", "")
            records_required = bool(adjustment_record.get("records_required", True))

            logger.info(LOG_STATUS_REPORTED)

            # The reference number is freshly resolved and looked up, so any
            # claim artifacts that were stale on a disputed reference are clean
            # again (clears the stale-reference guard before records/notification).
            _clean = clear_dirty(state.get("dirty_artifacts"), "upload_link")
            _clean = clear_dirty(_clean, "personal_guide_outreach")

            status_msg = random.choice(STATUS_REPORT_TEMPLATES).format(
                status=claim_status,
                last_update_date=last_update_date,
            )

            if records_required:
                records_msg = pick(MSG_RECORDS_NEEDED)
                full_msg = f"{status_msg}\n{records_msg}"
                result = self.ask_member(state, full_msg)
                result["reference_number"] = reference_number
                result["claim_status"] = claim_status
                result["last_update_date"] = last_update_date
                result["records_required"] = True
                result["next_node"] = "records_coordination_agent"
                result["awaiting_slot"] = ""
                result["dirty_artifacts"] = _clean
                return result
            else:
                result = self.ask_member(state, status_msg)
                result["reference_number"] = reference_number
                result["claim_status"] = claim_status
                result["last_update_date"] = last_update_date
                result["records_required"] = False
                result["next_node"] = "notification_setup_agent"
                result["awaiting_slot"] = ""
                result["dirty_artifacts"] = _clean
                return result

        # ── PHASE 4: signal_complete — both sub-agents already ran ─────────────
        return self.signal_complete(
            state,
            message="",
            resolved_intents=["claim_services"],
            context_updates=self._completion_context(state),
        )

    @staticmethod
    def _completion_context(state: State) -> dict:
        return {
            "claim_flow_complete": True,
            "reference_number": state.get("reference_number", ""),
            "claim_status": state.get("claim_status", ""),
            "last_update_date": state.get("last_update_date", ""),
            "records_required": state.get("records_required", False),
            "records_branch_taken": state.get("records_branch_taken", ""),
            "upload_link_sent": state.get("upload_link_sent", False),
            "personal_guide_outreach_requested": state.get("personal_guide_outreach_requested", False),
            "notification_channel": state.get("notification_channel", "not_set"),
            "claim_notification_contact": state.get("claim_notification_contact", ""),
        }


async def claim_adjustment_agent(state: State) -> dict:
    logger.info(LOG_ENTERED, extra={"call_intent": state.get("call_intent", "")})
    return await ClaimAdjustmentAgent.from_state(state).execute(state)
