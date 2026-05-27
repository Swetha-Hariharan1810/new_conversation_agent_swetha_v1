"""
verification_agent.py — Identity verification orchestrator.

run() is the only logic here. Everything else is delegated:
  pipelines.py                         — slot collection configuration (what to collect)
  handlers.py                          — SF lookup, post-lookup, corrections, off-topic redirect
  llm.py                               — LLM extraction
  constants.py                         — slot ordering, keywords, log names
  src/agent/responses/message_builders — verification prompt builders
"""

from __future__ import annotations

import random

from agent.agents.verification.constants import (
    IDENTITY_SLOT_ORDER,
    LOG_ENTERED,
    LOG_VERIFIED,
    VERIFIED_MSG_TEMPLATES,
)
from agent.agents.verification.handlers import (
    _NORMALIZERS,
    _VALIDATORS,
    apply_corrections,
    collect_post_lookup,
    lookup_and_verify,
    redirect_off_topic,
)
from agent.agents.verification.llm import extract_verification_decision
from agent.agents.verification.pipelines import (
    build_claims_pipeline,
    build_identity_pipeline,
    build_provider_pipeline,
)
from agent.core.agent import BaseAgent
from agent.llm.config import get_extraction_llm
from agent.logger import get_logger
from agent.state import State
from agent.utils import (
    _last_assistant_msg,
    _last_user_msg,
    build_extraction_prompt,
)

logger = get_logger(__name__)


class VerificationAgent(BaseAgent):
    AGENT_NAME = "verification_agent"

    def __init__(self) -> None:
        super().__init__()
        self._identity_pipeline = build_identity_pipeline(self)
        self._claims_pipeline = build_claims_pipeline(self)
        self._provider_pipeline = build_provider_pipeline(self)

    async def run(self, state: State) -> dict:  # noqa: C901
        # Early exit: member already fully verified on re-entry — skip all slot collection.
        # Guard requires awaiting_slot to be empty: if a post-lookup slot (relationship,
        # phone_confirmed) is still being collected, the pipeline must run to completion
        # or escalation. Firing here while awaiting_slot is set skips exhaustion checks
        # and incorrectly signals verification complete mid-pipeline.
        if state.get("member_status_verify") and not state.get("awaiting_slot"):
            collected = {k: (state.get(k) or "").strip() for k in IDENTITY_SLOT_ORDER}
            if state.get("phone_confirmed") is True:
                collected["phone_confirmed"] = True
            member_record = self._member_record_from_state(state)
            return self._signal_verified(state, collected, member_record)

        messages = list(state.get("messages") or [])
        last_user = _last_user_msg(messages)
        call_intent = state.get("call_intent", "")

        _prompt_file = (
            "extraction/verification_claims.md"
            if call_intent == "claim_services"
            else "extraction/verification_provider.md"
        )
        system_prompt = build_extraction_prompt(_prompt_file)
        collected = {k: (state.get(k) or "").strip() for k in IDENTITY_SLOT_ORDER}

        if call_intent == "claim_services":
            slot_order = ["first_name", "last_name", "member_id", "dob", "phone_confirmed"]
        else:
            slot_order = ["first_name", "last_name", "member_id", "dob", "relationship"]
        awaiting_slot = state.get("awaiting_slot") or next(
            (s for s in slot_order if not str(state.get(s) or "").strip()),
            IDENTITY_SLOT_ORDER[-1],  # "dob" — all identity slots collected; accurate
        )  # context so the LLM detects corrections correctly
        # Write computed awaiting_slot into state so _collect_slot can route CORRECTED/
        # AMBIGUOUS events correctly when the slot was not explicitly set by a prior turn.
        state = {**state, "awaiting_slot": awaiting_slot}

        last_agent = _last_assistant_msg(messages)
        confirmed_slots = {k: v for k, v in collected.items() if v and v.strip()}
        current_attempt = (state.get("slot_attempts") or {}).get(awaiting_slot, {})
        attempt_count = current_attempt.get("attempt_count", 0) if isinstance(current_attempt, dict) else 0
        restart_index = state.get("verification_restart_index") or 0
        if restart_index:
            # Include up to 2 pre-restart messages as context so the extraction
            # LLM can re-extract slots the caller already stated in round 1.
            pre_restart_context = messages[max(0, restart_index - 2) : restart_index]
            post_restart = messages[restart_index:]
            recent_messages = (pre_restart_context + post_restart)[-8:]
        else:
            recent_messages = messages[-6:]
        result = await extract_verification_decision(
            get_extraction_llm(),
            system_prompt,
            awaiting_slot=awaiting_slot,
            last_agent_message=last_agent,
            last_user_message=last_user,
            confirmed_slots=confirmed_slots,
            attempt=attempt_count,
            recent_messages=recent_messages,
        )

        if interrupt := await self.run_conversation_guards(
            state,
            user_text=last_user,
            result=result,
        ):
            if getattr(result, "guard", "") == "OFFTOPIC_AGENT":
                return redirect_off_topic(self, state, collected, self._identity_pipeline)
            return interrupt

        corrected_fields: list[str] = []
        if last_user:
            corrected_fields = apply_corrections(self, collected, state, result) or []

        # Cascade clears: apply_corrections zeroes collected["last_name"] when first_name
        # is corrected and collected["dob"] when member_id is corrected. But _collect_slot
        # reads from state (not collected) for the "already valid" check, so we must also
        # clear the relevant state keys to prevent the old value from being silently reused.
        _corr = getattr(result, "corrections", {}) or {}
        if _corr.get("first_name") and not collected.get("last_name"):
            state = {**state, "last_name": ""}
        if _corr.get("member_id") and not collected.get("dob"):
            state = {**state, "dob": ""}

        # ── correction_return_to: set when correcting a field other than awaiting slot ──
        if corrected_fields and awaiting_slot not in corrected_fields:
            # Caller corrected a confirmed slot; pipeline must return to awaiting_slot after.
            state = {**state, "correction_return_to": awaiting_slot}
        elif not corrected_fields and state.get("correction_return_to"):
            # No correction this turn — preserve existing pointer from prior turn.
            pass  # state already has the right value

        # ── ambiguous_counts: carry forward so the counter accumulates across turns ──
        # _collect_slot writes updated counts into the interrupt dict each turn.
        # On re-entry, those counts arrive in state — no additional wiring needed here,
        # but confirm ambiguous_counts is present with a safe default if state is fresh:
        if "ambiguous_counts" not in state:
            state = {**state, "ambiguous_counts": {}}

        # ── Pre-save bonus extractions ────────────────────────────────────────
        # The pipeline processes slots IN ORDER and stops at the first failure.
        # If the user provided a valid value for a slot that comes AFTER the
        # currently failing slot (e.g., gave DOB while awaiting member_id), that
        # value is in result.extracted but will be discarded when the pipeline
        # returns early for the failing slot.
        # Pre-populating collected here ensures the pipeline skips those slots
        # on the next iteration instead of re-asking.
        if result and result.extracted:
            for _bonus_slot, _bonus_raw in result.extracted.items():
                if (
                    _bonus_slot in IDENTITY_SLOT_ORDER
                    and _bonus_slot != awaiting_slot
                    and not collected.get(_bonus_slot)
                    and _bonus_raw
                ):
                    _norm_fn = _NORMALIZERS.get(_bonus_slot)
                    _val_fn = _VALIDATORS.get(_bonus_slot)
                    if _norm_fn and _val_fn:
                        _normalized = _norm_fn(str(_bonus_raw))
                        if _normalized and _val_fn(_normalized).valid:
                            collected[_bonus_slot] = _normalized
                            self.slot_ok(_bonus_slot, _normalized)
                            logger.info(
                                "VerificationAgent: bonus extraction saved",
                                extra={"slot": _bonus_slot, "awaiting": awaiting_slot},
                            )

        # Collect first_name → last_name → member_id → dob
        if interrupt := await self._identity_pipeline.collect(state, messages, collected, decision=result):
            return interrupt

        # Salesforce lookup
        if not state.get("member_status_verify"):
            member_record, interrupt = await lookup_and_verify(self, state, collected)
            if interrupt:
                return interrupt
        else:
            member_record = self._member_record_from_state(state)

        # Eagerly merge SF lookup fields into a local state snapshot so they are
        # readable during post-lookup slot collection retries (relationship label,
        # phone_confirmed label). _signal_verified also writes these at the end,
        # but only when the node returns — they must be readable mid-execution.
        if member_record:
            state = {
                **state,
                "phone_number": member_record.get("phone_number") or state.get("phone_number", ""),
                "relationship": member_record.get("relationship") or state.get("relationship", ""),
            }

        # Phone confirmation (claims) or relationship (provider)
        if interrupt := await collect_post_lookup(
            self,
            state,
            messages,
            collected,
            call_intent,
            member_record,
            result,
            self._claims_pipeline,
            self._provider_pipeline,
        ):
            return interrupt

        return self._signal_verified(state, collected, member_record)

    # -------------------------------------------------------------------------
    # Private helpers (state loading + final signal)
    # -------------------------------------------------------------------------

    def _member_record_from_state(self, state: State) -> dict:
        """Reconstruct a minimal member record from already-verified state fields."""
        return {k: state.get(k, "") for k in ["phone_number", "zip_code", "fax", "email", "relationship"]}

    def _signal_verified(self, state: State, collected: dict, member_record: dict | None) -> dict:
        """Emit COMPLETE signal with all verified identity fields as context updates."""
        context_updates = {"member_status_verify": True, "verification_restart_index": 0, **collected}
        if member_record:
            for field in ["zip_code", "phone_number", "fax", "email", "relationship"]:
                if val := member_record.get(field):
                    context_updates[field] = val
        logger.info(LOG_VERIFIED)
        return self.signal_complete(
            state,
            message=(
                ""
                if state.get("call_intent") == "provider_services"
                else random.choice(VERIFIED_MSG_TEMPLATES).format(first_name=collected["first_name"])
            ),
            resolved_intents=["verification"],
            context_updates=context_updates,
            reasoning="Identity verified — routing to domain agent",
        )


async def verification_agent(state: State) -> dict:
    logger.info(LOG_ENTERED, extra={"call_intent": state.get("call_intent", "")})
    return await VerificationAgent.from_state(state).execute(state)
