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

from agent.agents.intake.constants import INTENT_BRIDGE_MSGS
from agent.agents.verification.constants import (
    IDENTITY_SLOT_ORDER,
    LOG_ENTERED,
    LOG_NAME_CONFIRM_EXHAUST,
    LOG_NAME_CONFIRMED,
    LOG_NAME_CORRECTED,
    LOG_NAME_READBACK,
    LOG_VERIFIED,
    MAX_NAME_CONFIRM_ATTEMPTS,
    MSG_NAME_CONFIRM_EXHAUST,
    NAME_CORRECTION_PROMPTS,
    NAME_READBACK_TEMPLATES,
)
from agent.agents.verification.handlers import (
    _NORMALIZERS,
    _VALIDATORS,
    apply_corrections,
    collect_post_lookup,
    lookup_and_verify,
    redirect_off_topic,
)
from agent.agents.verification.llm import extract_name_confirmation, extract_verification_decision
from agent.agents.verification.pipelines import (
    build_claims_pipeline,
    build_identity_pipeline,
    build_provider_pipeline,
)
from agent.conversation.context import ConversationContext
from agent.core.agent import BaseAgent
from agent.llm.config import get_extraction_llm
from agent.llm.schema import EventType
from agent.logger import get_logger
from agent.orchestration.orchestration import AgentNode
from agent.slots.normalizers import normalize_name
from agent.slots.validators import validate_name
from agent.state import State
from agent.utils import (
    _last_assistant_msg,
    _last_user_msg,
    build_extraction_prompt,
    build_extraction_prompt_extraction,
    pick,
)

logger = get_logger(__name__)

_NAME_CONFIRM_SLOT = "name_confirmed"
_NAME_CORRECTION_SLOT = "name_correction"

# Mid-call intent switch → domain node. When follow_up stages a new intent via
# reset_for_new_intent, verification consumes pending_intent on success and
# dispatches straight to that intent's node. Keys are the intent-tag vocabulary
# (call_intent / detected_intent), values are the registered graph node names.
_PENDING_INTENT_NODE = {
    "provider_services": AgentNode.PROVIDER_SEARCH.value,
    "claim_services": AgentNode.CLAIM_ADJUSTMENT.value,
}


def _spell_name(first: str, last: str) -> str:
    """
    Emily Carter -> "Emily Carter, E-M-I-L-Y C-A-R-T-E-R"
    """

    def _spell(word: str) -> str:
        return "-".join(ch.upper() for ch in word if ch.isalpha() or ch == "-")

    full_name = " ".join(part for part in [first, last] if part).strip()
    spelled_name = " ".join(_spell(part) for part in [first, last] if part)

    return f"{full_name}. That's spelled {spelled_name}"


def _build_name_readback_message(first: str, last: str) -> str:
    """Pick a random readback template and fill in the spelled name."""
    spelled = _spell_name(first, last)
    return random.choice(NAME_READBACK_TEMPLATES).format(spelled=spelled)


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

        # ── Mid-call re-verification first-name bridge (one-shot) ────────────
        # reset_for_new_intent sets reverify_bridge_pending=True when a fresh
        # intent is detected mid-call. The utterance that triggered the switch
        # carries no identity data, so the extraction LLM call below would be
        # wasted. Instead deliver the same deterministic first-name bridge intake
        # uses and pause for the member's reply. The flag is one-shot — keyed off
        # reverify_bridge_pending, NOT pending_intent (which persists across every
        # re-verification turn and would re-fire the bridge each turn).
        # Edge case (acceptable tradeoff): if the trigger utterance happened to
        # include a name, we still re-ask for first name via the bridge rather
        # than extracting it — identity is re-collected from scratch by design.
        if state.get("reverify_bridge_pending"):
            msg = random.choice(INTENT_BRIDGE_MSGS)
            result = self.ask_member(state, msg)  # sets is_interrupt=True, next_node=verification_agent
            result["reverify_bridge_pending"] = False  # one-shot: clear so next turn extracts normally
            result["awaiting_slot"] = "first_name"  # correct slot context for next-turn extraction
            logger.info("VerificationAgent: re-verify first-name bridge delivered (LLM call skipped)")
            return result

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

        # ── NAME CONFIRMATION GATE ────────────────────────────────────────────
        # Fires once both names are in state and before member_id is collected.
        # The name_confirmed flag prevents re-entry on subsequent turns.
        _fn = (state.get("first_name") or "").strip()
        _ln = (state.get("last_name") or "").strip()
        if _fn and _ln and not state.get("name_confirmed"):
            return await self._handle_name_confirmation(state, messages, last_user)
        # ─────────────────────────────────────────────────────────────────────

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

        # Verification flow: never pause for side questions. Treat
        # answered_with_followup as a plain answer so the pipeline confirms
        # the slot and immediately moves to the next one.
        if result and result.event_type == EventType.ANSWERED_WITH_FOLLOWUP:
            result.event_type = EventType.ANSWERED

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

        # sync corrected first_name into state and context immediately
        if "first_name" in corrected_fields:
            ctx = ConversationContext.from_state(state)
            ctx.update_caller_name(collected["first_name"])
            state = {
                **state,
                "first_name": collected["first_name"],
                "conversation_context": ctx.to_dict(),
            }

        # Cascade clears: apply_corrections zeroes collected["last_name"] when first_name
        # is corrected and collected["dob"] when member_id is corrected. But _collect_slot
        # reads from state (not collected) for the "already valid" check, so we must also
        # clear the relevant state keys to prevent the old value from being silently reused.
        # only cascade-clear if value not provided in same utterance
        _corr = getattr(result, "corrections", {}) or {}
        _extracted_this_turn = (result.extracted or {}) if result else {}
        if (
            _corr.get("first_name")
            and not collected.get("last_name")
            and not _extracted_this_turn.get("last_name")
        ):
            state = {**state, "last_name": ""}
        if _corr.get("member_id") and not collected.get("dob") and not _extracted_this_turn.get("dob"):
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

        # ── NAME-PAIR INTERCEPT — fire readback before collecting member_id ──
        # The identity pipeline collects first_name → last_name → member_id → dob
        # in one pass. On the turn the member supplies the last name (first name
        # already on file), the pipeline confirms last_name and continues straight
        # to asking for member_id, skipping the name-confirmation readback. Resolve
        # the name pair here — including last_name from THIS turn's extraction,
        # which bonus-extraction skips because it equals the awaiting slot — and
        # deliver the readback before the pipeline can advance.
        if not state.get("name_confirmed"):
            _extracted_now = (result.extracted or {}) if result else {}

            def _resolve_name(slot: str) -> str:
                v = (collected.get(slot) or state.get(slot) or "").strip()
                if v:
                    return v
                raw = _extracted_now.get(slot, "")
                if raw:
                    norm = normalize_name(raw)
                    if norm and validate_name(norm).valid:
                        return norm
                return ""

            _fn_pair = _resolve_name("first_name")
            _ln_pair = _resolve_name("last_name")

            # ── FALLBACK: LLM extracted only first_name but last_name was
            # also in the utterance (e.g. "emily carter", "John Smith").
            # When the utterance is exactly two whitespace-separated tokens
            # and the first token matches the extracted first name, treat
            # the second token as the last name. Zero LLM cost; prevents
            # the spurious "What's your last name?" re-ask when both names
            # were already given in a single utterance.
            if _fn_pair and not _ln_pair and last_user:
                _tokens = last_user.strip().split()
                if len(_tokens) == 2:
                    _candidate_last = normalize_name(_tokens[1])
                    if (
                        _candidate_last
                        and validate_name(_candidate_last).valid
                        and normalize_name(_tokens[0]).lower() == _fn_pair.lower()
                    ):
                        _ln_pair = _candidate_last
                        logger.info(
                            "VerificationAgent: last_name recovered from two-token utterance fallback",
                            extra={"first": _fn_pair, "last": _candidate_last},
                        )
            # ── END FALLBACK ──────────────────────────────────────────────

            if _fn_pair and _ln_pair:
                self.slot_ok("first_name", _fn_pair)
                self.slot_ok("last_name", _ln_pair)
                collected["first_name"] = _fn_pair
                collected["last_name"] = _ln_pair
                state = {**state, "first_name": _fn_pair, "last_name": _ln_pair}
                return await self._handle_name_confirmation(state, messages, last_user)

        # Collect first_name → last_name → member_id → dob
        if interrupt := await self._identity_pipeline.collect(state, messages, collected, decision=result):
            return interrupt

        # Both names just confirmed this turn — fire name readback immediately.
        _fn_now = (collected.get("first_name") or state.get("first_name") or "").strip()
        _ln_now = (collected.get("last_name") or state.get("last_name") or "").strip()
        if _fn_now and _ln_now and not state.get("name_confirmed"):
            state = {**state, "first_name": _fn_now, "last_name": _ln_now}
            return await self._handle_name_confirmation(state, messages, last_user)

        # Salesforce lookup
        #
        # ── Partial re-ask round-trip (wrong DOB only) ───────────────────────────
        # Trace of the targeted re-ask path, e.g. caller's DOB is wrong but name +
        # Member ID match:
        #
        #   Turn N (all four slots collected):
        #     pipeline.collect() completes → lookup_and_verify() runs →
        #     full match fails → lookup_member returns member_id_found=True with
        #     field_matches={first_name:T, last_name:T, dob:F} →
        #     handlers._partial_reask(mismatched=["dob"]) returns an interrupt that:
        #       • clears ONLY dob (""); keeps first_name, last_name, member_id
        #       • leaves name_confirmed=True untouched (no name field mismatched)
        #       • sets awaiting_slot="dob" and verification_restart_index=len(msgs)
        #       • asks MSG_REASK_DOB  ("…date of birth once more?")
        #     member_status_verify is NOT set, so the loop stays open.
        #
        #   Turn N+1 (caller restates DOB):
        #     • name gate (line ~136) skipped: name_confirmed is True
        #     • both readback intercepts (lines ~260, ~316) skipped: name_confirmed True
        #       → NO spelled-name read-back, NO name/Member-ID re-ask
        #     • awaiting_slot="dob" gives the extractor the correct slot context
        #     • pipeline.collect() skips the still-populated first_name / last_name /
        #       member_id and collects only dob (first empty slot in identity order)
        #     • member_status_verify still falsy → lookup_and_verify() runs AGAIN
        #       with the corrected dob → fresh full match → _signal_verified().
        # ─────────────────────────────────────────────────────────────────────────
        return await self._finish_after_identity(state, collected, messages, call_intent, result)

    async def _finish_after_identity(
        self, state: State, collected: dict, messages: list, call_intent: str, decision
    ) -> dict:
        """Salesforce lookup → post-lookup slot → verified signal.

        Shared by run()'s main path and _name_confirmed_proceed's all-slots-present
        branch (name-only partial re-ask), so a corrected name flows straight to the
        lookup instead of re-asking an already-known Member ID.
        """
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
            decision,
            self._claims_pipeline,
            self._provider_pipeline,
        ):
            return interrupt

        return self._signal_verified(state, collected, member_record)

    # =========================================================================
    # Name confirmation phase
    # =========================================================================

    async def _handle_name_confirmation(self, state: State, messages: list, last_user: str) -> dict:
        """
        Router for the name readback → confirm / correct loop.

        Entry routing by awaiting_slot:
          ""                      → first entry: deliver readback
          _NAME_CONFIRM_SLOT      → member just responded to a readback
          _NAME_CORRECTION_SLOT   → member just gave the corrected name (after bare no)
        """
        current_awaiting = state.get("awaiting_slot", "")
        if current_awaiting == _NAME_CONFIRM_SLOT:
            return await self._process_name_readback_response(state, messages, last_user)
        if current_awaiting == _NAME_CORRECTION_SLOT:
            return await self._collect_name_correction(state, messages, last_user)
        # First entry or any unrecognised slot — deliver the readback.
        return self._deliver_name_readback(state)

    def _deliver_name_readback(self, state: State) -> dict:
        """Send the spelled-out name readback and set awaiting_slot=name_confirmed."""
        first = (state.get("first_name") or "").strip()
        last = (state.get("last_name") or "").strip()

        ctx = ConversationContext.from_state(state)
        ctx.update_caller_name(first)
        msg = _build_name_readback_message(first, last)

        logger.info(LOG_NAME_READBACK, extra={"first": first, "last": last})
        result = self.ask_member(state, msg)
        result["awaiting_slot"] = _NAME_CONFIRM_SLOT
        result["first_name"] = first
        result["last_name"] = last
        result["conversation_context"] = ctx.to_dict()
        return result

    async def _process_name_readback_response(self, state: State, messages: list, last_user: str) -> dict:
        """
        Extract the member's response to the spelled-out name readback.

        Three outcomes — see name_confirmation.md for the extraction contract:
          1. name_confirmed="yes"            → proceed to member_id
          2. first_name / last_name present  → inline correction; re-read back
          3. name_confirmed="no", no names   → ask for correct name separately
          4. ambiguous                       → slot_fail → retry readback or escalate
        """
        last_agent = _last_assistant_msg(messages)
        attempt_count = self.get_slot(_NAME_CONFIRM_SLOT).attempt_count

        result = await extract_name_confirmation(
            get_extraction_llm(),
            build_extraction_prompt_extraction("extraction/name_confirmation.md"),
            last_agent_message=last_agent,
            last_user_message=last_user,
            attempt=attempt_count,
            recent_messages=messages[-4:],
        )

        if interrupt := await self.run_conversation_guards(state, user_text=last_user, result=result):
            return interrupt

        extracted = (result.extracted or {}) if result else {}
        name_conf_raw = extracted.get("name_confirmed", "")
        corrected_first_raw = extracted.get("first_name", "")
        corrected_last_raw = extracted.get("last_name", "")

        # ── OUTCOME 1: confirmed ─────────────────────────────────────────────
        if name_conf_raw == "yes":
            logger.info(LOG_NAME_CONFIRMED)
            return await self._name_confirmed_proceed(state, messages)

        # ── OUTCOME 2: inline correction ─────────────────────────────────────
        corrected_first = normalize_name(corrected_first_raw) if corrected_first_raw else ""
        corrected_last = normalize_name(corrected_last_raw) if corrected_last_raw else ""
        first_ok = bool(corrected_first) and validate_name(corrected_first).valid
        last_ok = bool(corrected_last) and validate_name(corrected_last).valid

        if first_ok or last_ok:
            new_first = corrected_first if first_ok else (state.get("first_name") or "").strip()
            new_last = corrected_last if last_ok else (state.get("last_name") or "").strip()
            logger.info(LOG_NAME_CORRECTED, extra={"new_first": new_first, "new_last": new_last})

            attempts = (state.get("name_confirm_attempts") or 0) + 1
            if attempts >= MAX_NAME_CONFIRM_ATTEMPTS:
                logger.warning(LOG_NAME_CONFIRM_EXHAUST)
                return self.signal_escalate(
                    state, pick(MSG_NAME_CONFIRM_EXHAUST), reason="name_confirm_exhausted"
                )
            new_state = {
                **state,
                "first_name": new_first,
                "last_name": new_last,
                "name_confirm_attempts": attempts,
            }
            return self._deliver_name_readback(new_state)

        # ── OUTCOME 3: bare no ───────────────────────────────────────────────
        if name_conf_raw == "no":
            attempts = (state.get("name_confirm_attempts") or 0) + 1
            if attempts >= MAX_NAME_CONFIRM_ATTEMPTS:
                logger.warning(LOG_NAME_CONFIRM_EXHAUST)
                return self.signal_escalate(
                    state, pick(MSG_NAME_CONFIRM_EXHAUST), reason="name_confirm_exhausted"
                )
            ask = self.ask_member(state, pick(NAME_CORRECTION_PROMPTS))
            ask["awaiting_slot"] = _NAME_CORRECTION_SLOT
            ask["name_confirm_attempts"] = attempts
            return ask

        # ── OUTCOME 4: ambiguous — retry readback ────────────────────────────
        self.slot_fail(_NAME_CONFIRM_SLOT)
        if self.get_slot(_NAME_CONFIRM_SLOT).is_exhausted():
            logger.warning(LOG_NAME_CONFIRM_EXHAUST)
            return self.signal_escalate(
                state, pick(MSG_NAME_CONFIRM_EXHAUST), reason="name_confirm_exhausted"
            )
        return self._deliver_name_readback(state)

    async def _collect_name_correction(self, state: State, messages: list, last_user: str) -> dict:
        """
        The member said bare 'no' to the readback. We asked for the correct name.
        Extract the corrected name, then re-deliver the readback with that name.
        """
        last_agent = _last_assistant_msg(messages)
        attempt_count = self.get_slot(_NAME_CORRECTION_SLOT).attempt_count

        result = await extract_name_confirmation(
            get_extraction_llm(),
            build_extraction_prompt_extraction("extraction/name_confirmation.md"),
            last_agent_message=last_agent,
            last_user_message=last_user,
            attempt=attempt_count,
            recent_messages=messages[-4:],
        )

        if interrupt := await self.run_conversation_guards(state, user_text=last_user, result=result):
            return interrupt

        extracted = (result.extracted or {}) if result else {}
        corrected_first_raw = extracted.get("first_name", "")
        corrected_last_raw = extracted.get("last_name", "")

        corrected_first = normalize_name(corrected_first_raw) if corrected_first_raw else ""
        corrected_last = normalize_name(corrected_last_raw) if corrected_last_raw else ""
        first_ok = bool(corrected_first) and validate_name(corrected_first).valid
        last_ok = bool(corrected_last) and validate_name(corrected_last).valid

        if first_ok or last_ok:
            new_first = corrected_first if first_ok else (state.get("first_name") or "").strip()
            new_last = corrected_last if last_ok else (state.get("last_name") or "").strip()
            logger.info(
                LOG_NAME_CORRECTED,
                extra={"new_first": new_first, "new_last": new_last, "source": "correction_slot"},
            )
            # name_confirm_attempts was already incremented when we entered the
            # correction slot — do not increment again.
            new_state = {**state, "first_name": new_first, "last_name": new_last}
            return self._deliver_name_readback(new_state)

        # Nothing extractable — retry the "what is the correct name?" question.
        self.slot_fail(_NAME_CORRECTION_SLOT)
        if self.get_slot(_NAME_CORRECTION_SLOT).is_exhausted():
            logger.warning(LOG_NAME_CONFIRM_EXHAUST)
            return self.signal_escalate(
                state, pick(MSG_NAME_CONFIRM_EXHAUST), reason="name_confirm_exhausted"
            )
        ctx = ConversationContext.from_state(state)
        retry_msg = await self._generate_slot_retry_response(
            state, _NAME_CORRECTION_SLOT, ctx, messages, guard="RETRY"
        )
        retry = self.ask_member(state, retry_msg)
        retry["awaiting_slot"] = _NAME_CORRECTION_SLOT
        return retry

    async def _name_confirmed_proceed(self, state: State, messages: list) -> dict:
        """
        Mark the name confirmed and continue identity collection.

        Normal first-time flow: member_id / dob are still empty, so deliver the
        next-slot transition prompt and pause (is_interrupt=True) so the next
        human turn is the real member_id/dob answer. We must not re-enter run()
        with is_interrupt=False on this path, because that reprocesses the same
        stale "yes": run() would fire a second extraction LLM call (classified
        against member_id) and the pipeline would treat the "yes" as a non-answer,
        firing a recovery-message LLM call — so the member hears a retry prompt
        instead of the clean member_id ask.

        Name-only partial re-ask: member_id AND dob were retained, so there is no
        empty identity slot to ask. Re-asking would produce a spurious Member-ID
        prompt, so instead proceed straight to the Salesforce lookup with the
        corrected name. (No stale-"yes" hazard here: with every slot filled, the
        pipeline has nothing to misclassify the "yes" against.)
        """
        from agent.responses.builder import build_transition_prompt
        from agent.slots.types import SlotType

        first = (state.get("first_name") or "").strip()
        last = (state.get("last_name") or "").strip()

        # Persist confirmed names so the gate never re-fires.
        self.slot_ok("first_name", first)
        self.slot_ok("last_name", last)

        ctx = ConversationContext.from_state(state)
        ctx.update_caller_name(first)
        state = {
            **state,
            "first_name": first,
            "last_name": last,
            "name_confirmed": True,
            "name_confirm_attempts": 0,
            "conversation_context": ctx.to_dict(),
        }

        # Next identity slot to collect (member_id, then dob), or None if every
        # identity slot is already present (name-only partial re-ask).
        next_slot = next((s for s in IDENTITY_SLOT_ORDER if not str(state.get(s) or "").strip()), None)

        if next_slot is None:
            # All identity slots present → re-run the lookup with the corrected
            # name rather than re-asking an already-known slot.
            collected = {k: (state.get(k) or "").strip() for k in IDENTITY_SLOT_ORDER}
            call_intent = state.get("call_intent", "")
            result = await self._finish_after_identity(state, collected, messages, call_intent, None)
            # Persist the just-confirmed name on the RETURNED dict. On success
            # _finish_after_identity returns the post-lookup interrupt
            # (relationship / phone) or the COMPLETE signal — neither carries
            # name_confirmed, so without this the gate would re-fire the read-back
            # on the next (post-lookup) turn. If the lookup instead returned a
            # re-ask that deliberately set name_confirmed (full restart, or a
            # fresh name mismatch → False), respect that value.
            if isinstance(result, dict) and "name_confirmed" not in result:
                result["name_confirmed"] = True
                result["name_confirm_attempts"] = 0
            return result

        slot_type = SlotType.MEMBER_ID if next_slot == "member_id" else SlotType.DOB
        msg = build_transition_prompt(slot_type, ctx)

        result = self.ask_member(state, msg)
        result["awaiting_slot"] = next_slot
        result["name_confirmed"] = True
        result["name_confirm_attempts"] = 0
        result["first_name"] = first
        result["last_name"] = last
        result["conversation_context"] = ctx.to_dict()
        return result

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
            # Pass prefetched benefits fields into state so benefits_agent can skip
            # its own Salesforce call. Fields are only written if non-empty strings
            # to avoid overwriting existing state values with empty placeholders.
            for field in [
                "individual_deductible",
                "family_deductible",
                "coinsurance_percent",
                "individual_oop_max",
                "family_oop_max",
            ]:
                val = member_record.get(field, "")
                if val:
                    context_updates[field] = val
        logger.info(LOG_VERIFIED)
        result = self.signal_complete(
            state,
            # message=(
            #     ""
            #     if state.get("call_intent") in ("provider_services", "claim_services")
            #     else random.choice(VERIFIED_MSG_TEMPLATES).format(first_name=collected["first_name"])
            # ),
            message="",
            resolved_intents=["verification"],
            context_updates=context_updates,
            reasoning="Identity verified — routing to domain agent",
        )

        # ── Mid-call intent switch dispatch ──────────────────────────────────
        # follow_up stages a new intent via reset_for_new_intent, which sets
        # pending_intent. On successful re-verification, route straight to that
        # intent's domain node and consume pending_intent. First-ever verification
        # has no pending_intent → keep next_node="orchestrator" so the fast-path
        # routes by call_intent (existing behavior).
        pending = (state.get("pending_intent") or "").strip()
        if pending:
            result["pending_intent"] = ""  # consumed
            domain_node = _PENDING_INTENT_NODE.get(pending)
            if domain_node:
                result["next_node"] = domain_node
                logger.info(
                    "verification: pending_intent dispatch",
                    extra={"pending_intent": pending, "next_node": domain_node},
                )
        return result


async def verification_agent(state: State) -> dict:
    logger.info(LOG_ENTERED, extra={"call_intent": state.get("call_intent", "")})
    return await VerificationAgent.from_state(state).execute(state)
