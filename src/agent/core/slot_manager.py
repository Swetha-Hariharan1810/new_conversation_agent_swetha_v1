"""
slot_manager.py — SlotManagerMixin: slot state + per-turn slot collection.

Two concerns kept together because _collect_slot directly uses slot state:
  1. Slot CRUD (get_slot, slot_ok, slot_fail, slots_dict, guard_loop_limit)
  2. _collect_slot — the reusable per-turn collector used by every agent

_collect_slot returns:
  (value, None)      → slot confirmed, caller proceeds
  (None, interrupt)  → ask/retry dict, caller returns immediately
  (None, escalation) → retries exhausted, caller returns immediately

See git history for migration notes from v4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

from agent.conversation.context import ConversationContext
from agent.core.constants import MAX_WAIT_TURNS
from agent.core.models import SlotAttempt
from agent.responses.builder import (
    build_initial_prompt,
    build_transition_prompt,
)
from agent.responses.static import MSG_WAIT_ACK, MSG_WAIT_NUDGE, build_slot_exhausted_message
from agent.slots.types import SlotType
from agent.state import State
from agent.utils import _last_user_msg, detect_cannot_provide, detect_wait_request, pick

# ── Empathetic "cannot provide" escalation message ────────────────────────────
# Slot-aware: {slot_label} is filled at runtime from the SlotType label.
# Keeps exactly the same warm tone as the rest of the codebase.
_CANNOT_PROVIDE_MSG = "No problem — let me connect you with a representative "


def _mk_session_ctx(
    *,
    extracted_val: str | None = None,
    followup_query: str | None = None,
    confirmed_values: dict | None = None,
) -> dict:
    """Build a lightweight session-context dict for _generate_slot_retry_response."""
    ctx: dict = {}
    if extracted_val is not None:
        ctx["extracted_val"] = extracted_val
    if followup_query:
        ctx["followup_query"] = followup_query
    if confirmed_values is not None:
        ctx["confirmed_values"] = confirmed_values
    return ctx


# WorkerResult.followup_disposition → generation-LLM guard label.
# Missing/none defaults to FOLLOWUP_DECLINE: acknowledge and move on is always
# safe, whereas answering (FOLLOWUP_ANSWER) risks inventing an answer.
_DISPOSITION_GUARDS: dict[str, str] = {
    "answer_now": "FOLLOWUP_ANSWER",
    "park": "FOLLOWUP_PARK",
    "decline": "FOLLOWUP_DECLINE",
}

# Max detours per update target before escalating (counter: f"update_{target}").
_MAX_UPDATE_DETOURS = 2


# _InternalSlotConfig is the low-level config used by _collect_slot.
# The public SlotConfig used by SlotPipeline lives in slots/pipeline.py.
# pipeline.py imports this class for constructing _collect_slot calls.
@dataclass
class _InternalSlotConfig:
    """Low-level slot configuration consumed by _collect_slot."""

    slot_name: str
    prompt: str
    normalizer: Callable
    validator: Callable
    slot_type: Optional["SlotType"] = None


class SlotManagerMixin:
    """Mixin that adds slot management and collection to BaseAgent."""

    # -------------------------------------------------------------------------
    # Slot CRUD
    # -------------------------------------------------------------------------

    def get_slot(self, name: str) -> SlotAttempt:
        if name not in self._slots:
            self._slots[name] = SlotAttempt(name)
        return self._slots[name]

    def slot_ok(self, name: str, value: Any) -> None:
        """Mark slot confirmed. Also queues a CallAgentField event."""
        self.get_slot(name).record_attempt(value, success=True)
        self._newly_confirmed.add(name)

    def slot_fail(self, name: str, value: Any = None, is_asr: bool = False) -> None:
        """Record a failed slot attempt."""
        self.get_slot(name).record_attempt(value, success=False, is_asr=is_asr)

    def slot_exhausted(self, name: str) -> bool:
        return self.get_slot(name).is_exhausted()

    def slots_dict(self) -> dict:
        """Serialize slot state for LangGraph persistence."""
        return {
            k: {"attempt_count": v.attempt_count, "confirmed": v.confirmed, "last_value": v.last_value}
            for k, v in self._slots.items()
        }

    @staticmethod
    def _restore_slot(slot_name: str, data) -> SlotAttempt:
        slot = SlotAttempt(slot_name)
        slot.attempt_count = data.get("attempt_count", 0)
        slot.confirmed = data.get("confirmed", False)
        slot.last_value = data.get("last_value")
        return slot

    # -------------------------------------------------------------------------
    # ask_member_with_context — carries context in every interrupt
    # -------------------------------------------------------------------------

    def ask_member_with_context(
        self,
        state: State,
        message: str,
        context: ConversationContext,
    ) -> dict:
        """
        Like ask_member() but also persists the updated ConversationContext
        into state so it's available on the next turn.
        """
        result = self.ask_member(state, message)
        result["conversation_context"] = context.to_dict()
        return result

    # -------------------------------------------------------------------------
    # Loop guard
    # -------------------------------------------------------------------------

    def guard_loop_limit(
        self,
        state: State,
        counter_name: str,
        max_attempts: int,
        escalate_message: str,
        escalate_reason: str,
        *,
        initiator: str = "Agent",
    ) -> Optional[dict]:
        """
        Increment a named counter and return an escalation dict if the limit is hit.
        Returns None while still under the limit.
        """
        self.slot_fail(counter_name)
        count = self.get_slot(counter_name).attempt_count
        if count >= max_attempts:
            return self.signal_escalate(state, escalate_message, escalate_reason, initiator=initiator)
        return None

    # -------------------------------------------------------------------------
    # LLM 2 retry response helper
    # -------------------------------------------------------------------------

    async def _generate_slot_retry_response(
        self,
        state: State,
        slot_name: str,
        ctx: ConversationContext,
        messages: list[dict],
        *,
        guard: str = "RETRY",
        session_context: dict | None = None,
        extracted_this_turn: str | None = None,
    ) -> str:
        # Lazy import: core.slot_manager → llm.response_generator → llm.config → core (via schema);
        # importing at module level would create a core → llm → core cycle.
        from agent.llm.response_generator import generate_recovery_message

        slot_state = self.get_slot(slot_name)
        slot_label_override: str | None = None
        if slot_name == "relationship" and state.get("relationship"):
            slot_label_override = "relationship — whether they are the plan holder or dependent"
        elif slot_name in ("phone_confirmed", "phone_confirmation") and state.get("phone_number"):
            digits = "".join(c for c in state["phone_number"] if c.isdigit())
            formatted = (
                f"{digits[:3]}-{digits[3:6]}-{digits[6:]}" if len(digits) == 10 else state["phone_number"]
            )
            slot_label_override = (
                f"phone confirmation — whether {formatted} is still the number on file (yes or no)"
            )
        sc = session_context or {}
        text = await generate_recovery_message(
            slot_name=slot_name,
            attempt=slot_state.attempt_count,
            guard=guard,
            last_messages=messages[-4:],
            slot_label_override=slot_label_override,
            caller_name=ctx.caller_first_name,
            # FOLLOWUP_ANSWER passes real (masked) values so the LLM can answer
            # from Confirmed; every other guard keeps the "confirmed" placeholder.
            confirmed_slots=sc.get("confirmed_values") or dict.fromkeys(ctx.confirmed_slots, "confirmed"),
            user_utterance=_last_user_msg(messages),
            extracted_value=extracted_this_turn
            if extracted_this_turn is not None
            else sc.get("extracted_val"),
            followup_query=sc.get("followup_query"),
        )
        return text

    async def _generate_correction_ack(
        self,
        state: State,
        corrected_fields: list[str],
        awaiting_slot: str,
        ctx: "ConversationContext",
        messages: list,
        *,
        decision=None,
    ) -> str:
        from agent.agents.verification.handlers import _NORMALIZERS
        from agent.llm.response_generator import generate_recovery_message

        corrected_label = corrected_fields[0].replace("_", " ") if corrected_fields else "that"

        corrected_value = ""
        if corrected_fields:
            raw = getattr(decision, "corrections", {}) if decision else {}
            norm = _NORMALIZERS.get(corrected_fields[0])
            corrected_value = (
                norm(str(raw.get(corrected_fields[0], ""))) if norm else str(raw.get(corrected_fields[0], ""))
            )

        # explicit correction ack for name slots
        if corrected_fields and corrected_fields[0] in ("first_name", "last_name") and corrected_value:
            # Names: read back explicitly so caller hears their name confirmed
            slot_label_override = (
                f"caller corrected their {corrected_label} to '{corrected_value}' — "
                f"acknowledge by explicitly saying the corrected {corrected_label} "
                f"is '{corrected_value}', "
                f"then ask for their {awaiting_slot.replace('_', ' ')}"
            )
        elif corrected_value:
            # Sensitive slots (member_id, dob etc.): acknowledge WITHOUT reading
            # the value back out loud — just confirm the update and re-ask
            slot_label_override = (
                f"caller corrected their {corrected_label} — "
                f"acknowledge the correction without repeating the value, "
                f"then ask for their {awaiting_slot.replace('_', ' ')}"
            )
        else:
            # No new value provided: re-ask naturally for awaiting_slot
            slot_label_override = (
                f"caller corrected {corrected_label}" + f" — now re-ask for {awaiting_slot.replace('_', ' ')}"
            )

        return await generate_recovery_message(
            slot_name=awaiting_slot,
            attempt=0,
            guard="CORRECTION",
            last_messages=messages[-6:],
            slot_label_override=slot_label_override,
            caller_name=ctx.caller_first_name,
            confirmed_slots=dict.fromkeys(ctx.confirmed_slots, "confirmed"),
            user_utterance=_last_user_msg(messages[-6:]),
        )

    # -------------------------------------------------------------------------
    # Update / correction detours (Phase 4)
    # -------------------------------------------------------------------------

    @staticmethod
    def _resolve_slot_tools(slot_name: str, slot_configs: Optional[dict]) -> tuple:
        """Resolve (normalizer, validator) for a slot.

        The current pipeline's configs win, so any agent using SlotPipeline can
        validate corrections against its own slot definitions; the verification
        maps are the fallback for identity slots corrected mid-flow elsewhere.
        Returns (None, None) when the slot is unknown to both.
        """
        cfg = (slot_configs or {}).get(slot_name)
        if cfg is not None and getattr(cfg, "normalizer", None) and getattr(cfg, "validator", None):
            return cfg.normalizer, cfg.validator
        from agent.agents.verification.handlers import _NORMALIZERS, _VALIDATORS

        return _NORMALIZERS.get(slot_name), _VALIDATORS.get(slot_name)

    def _update_target_allowed(
        self,
        target: str,
        ctx: ConversationContext,
        state: State,
        slot_configs: Optional[dict],
    ) -> bool:
        """Belt-and-braces whitelist for caller-requested slot updates.

        A target is updatable only when it is already confirmed (in ctx or
        non-empty in state), is not a caller-locked field, and is collectable
        by the current pipeline. Anything else is treated as a decline.
        A target owned by a LATER agent could instead be parked, but that needs
        a slot→owning-agent registry which doesn't exist yet.
        TODO(followup-routing): park instead of decline once a SLOT_OWNERSHIP
        map is introduced.
        """
        from agent.agents.verification.handlers import CALLER_LOCKED_SLOTS

        if not target or target in CALLER_LOCKED_SLOTS:
            return False
        if slot_configs is not None and target not in slot_configs:
            return False
        return target in (ctx.confirmed_slots or []) or bool(str(state.get(target) or "").strip())

    def _confirmed_slot_values(
        self,
        ctx: ConversationContext,
        state: State,
        collected: Optional[dict] = None,
    ) -> dict:
        """Real confirmed slot values for the FOLLOWUP_ANSWER Confirmed: line.

        member_id / dob are never passed raw — masked as "on file".
        """
        from agent.llm.redaction import MASKED_SLOTS

        values: dict = {}
        for s in ctx.confirmed_slots:
            if s in MASKED_SLOTS:
                values[s] = "on file"
                continue
            v = (collected or {}).get(s) or state.get(s) or ""
            values[s] = str(v) if v else "confirmed"
        return values

    def _next_slot_ask(self, next_slot: str, slot_configs: Optional[dict], ctx: ConversationContext) -> str:
        """Static ask for the next pending slot, appended after the Gemini text.

        Typed slots use the transition template; untyped slots fall back to the
        config prompt (callable prompts get an empty collected dict — they only
        read already-collected values, all irrelevant to a next-slot ask).
        """
        cfg = (slot_configs or {}).get(next_slot)
        slot_type = getattr(cfg, "slot_type", None) if cfg is not None else None
        if slot_type:
            return build_transition_prompt(slot_type, ctx)
        prompt = getattr(cfg, "prompt", "") if cfg is not None else ""
        if callable(prompt):
            try:
                prompt = prompt({})
            except Exception:
                prompt = ""
        return prompt or f"Could you provide your {next_slot.replace('_', ' ')}?"

    async def _open_update_detour(  # noqa: C901
        self,
        state: State,
        target: str,
        return_to: str,
        ctx: ConversationContext,
        messages: list,
        *,
        flavor: str,
        collected: Optional[dict] = None,
        extracted_this_turn: str | None = None,
        followup_query: str = "",
    ) -> dict:
        """Open a detour to re-collect ``target`` before returning to ``return_to``.

        flavor:
          "answer"  — Case B: awaiting slot was just confirmed; FOLLOWUP_ANSWER
                      message acknowledges the answer AND asks for the new value.
          "bare"    — Case C2: bare update request; CORRECTION message asks for
                      the new value.
          "invalid" — Case C1: corrected value failed validation; CORRECTION
                      message says the new value didn't look right.

        Returns an interrupt dict (or an escalation dict when the per-target
        detour budget is exhausted).
        """
        from agent.llm.response_generator import generate_recovery_message

        # Loop guard: at most _MAX_UPDATE_DETOURS detours per target per call.
        if escalation := self.guard_loop_limit(
            state,
            f"update_{target}",
            _MAX_UPDATE_DETOURS,
            escalate_message=build_slot_exhausted_message(target),
            escalate_reason=f"update_{target} exhausted",
        ):
            return escalation

        target_label = target.replace("_", " ")
        if flavor == "answer":
            guard = "FOLLOWUP_ANSWER"
            override = (
                f"caller answered the current question AND asked to update their {target_label} "
                f"without giving a new value — acknowledge the captured answer, "
                f"then ask for the new {target_label}"
            )
        elif flavor == "invalid":
            guard = "CORRECTION"
            override = (
                f"caller gave a new {target_label} but it didn't look right — "
                f"acknowledge without repeating the value and ask for the correct {target_label}"
            )
        else:  # "bare"
            guard = "CORRECTION"
            override = (
                f"caller wants to update their {target_label} — "
                f"acknowledge and ask for the new {target_label}"
            )

        msg = await generate_recovery_message(
            slot_name=target,
            attempt=0,
            guard=guard,
            last_messages=messages[-6:],
            slot_label_override=override,
            caller_name=ctx.caller_first_name,
            confirmed_slots=dict.fromkeys(ctx.confirmed_slots, "confirmed"),
            user_utterance=_last_user_msg(messages),
            extracted_value=extracted_this_turn,
            followup_query=followup_query or None,
            # Case B (FOLLOWUP_ANSWER flavor): the sentence must end by asking
            # for the new value — followup_answer.md keys off this line.
            ask_for_new_value=(flavor == "answer"),
        )

        # Reset the target's slot record BEFORE ask_member: a stale confirmed
        # last_value would otherwise be re-persisted into later interrupts and
        # silently resurrect the value we're about to clear.
        self.get_slot(target).reset()
        ctx.confirmed_slots = [s for s in ctx.confirmed_slots if s != target]
        if target == "first_name":
            ctx.caller_first_name = ""

        interrupt = self.ask_member_with_context(state, msg, ctx)
        interrupt["awaiting_slot"] = target
        interrupt["correction_return_to"] = return_to
        interrupt[target] = ""  # clear the state key so the pipeline re-collects
        interrupt["wait_count"] = 0  # non-WAIT turn resets the wait streak
        if collected is not None:
            collected[target] = ""

        # Re-verification consequence: updating an identity slot invalidates the
        # completed Salesforce verification — clear the flag so lookup_and_verify
        # re-runs once the new value is collected (the detour completion path).
        from agent.agents.verification.constants import IDENTITY_SLOT_ORDER

        if target in IDENTITY_SLOT_ORDER and state.get("member_status_verify"):
            interrupt["member_status_verify"] = False

        return interrupt

    async def _handle_answered_followup(  # noqa: C901
        self,
        state: State,
        config: _InternalSlotConfig,
        messages: list,
        normalized: str,
        ctx: ConversationContext,
        *,
        decision: Any,
        pending_slots: list[str] | None,
        slot_configs: Optional[dict],
        collected: Optional[dict],
    ) -> Tuple[Optional[str], Optional[dict]]:
        """ANSWERED_WITH_FOLLOWUP: confirm the slot, route the follow-up disposition.

        Handles update Cases A (answer + valid corrections) and B (answer +
        value-less update_target) inline; plain follow-ups route to the
        FOLLOWUP_ANSWER / FOLLOWUP_PARK / FOLLOWUP_DECLINE guard and get the
        next pending slot's static ask appended (Option A: Gemini never asks
        for a slot — Python appends the ask).
        """
        from agent.agents.verification.constants import IDENTITY_SLOT_ORDER
        from agent.agents.verification.handlers import CALLER_LOCKED_SLOTS

        slot_name = config.slot_name
        extracted = getattr(decision, "extracted", None) or {}
        corrections = {k: v for k, v in (getattr(decision, "corrections", None) or {}).items() if v}
        update_target = (getattr(decision, "update_target", None) or "").strip()
        followup_query = (getattr(decision, "followup_query", None) or "").strip()
        disposition = getattr(decision, "followup_disposition", None)
        disposition_value = str(getattr(disposition, "value", disposition) or "none")

        # ── Case A: apply validated corrections + cascade clears BEFORE the
        # awaiting slot is confirmed, so the confirm order matches apply_corrections.
        applied: list[str] = []
        cascade_cleared: list[str] = []
        invalid_target = ""
        for target, raw in corrections.items():
            if target in CALLER_LOCKED_SLOTS or target == slot_name:
                continue
            norm_fn, val_fn = self._resolve_slot_tools(target, slot_configs)
            if not (norm_fn and val_fn):
                continue
            corrected_val = norm_fn(str(raw))
            if corrected_val and val_fn(corrected_val).valid:
                self.slot_ok(target, corrected_val)
                if collected is not None:
                    collected[target] = corrected_val
                applied.append(target)
            elif not invalid_target:
                # New value failed validation — never applied. After the awaiting
                # slot confirms, a detour re-collects this slot (Phase 7).
                invalid_target = target

        # Existing cascade clears (mirrors handlers.apply_corrections): a new
        # first_name invalidates last_name, a new member_id invalidates dob —
        # unless the dependent value arrived in the same utterance.
        if "first_name" in applied and slot_name != "last_name" and not extracted.get("last_name"):
            cascade_cleared.append("last_name")
        if "member_id" in applied and slot_name != "dob" and not extracted.get("dob"):
            cascade_cleared.append("dob")
        for cleared in cascade_cleared:
            self.get_slot(cleared).reset()
            ctx.confirmed_slots = [s for s in ctx.confirmed_slots if s != cleared]
            if collected is not None:
                collected[cleared] = ""

        clear_verify = bool(state.get("member_status_verify")) and any(
            t in IDENTITY_SLOT_ORDER for t in applied
        )

        # ── Confirm the awaiting slot ────────────────────────────────────────
        self.slot_ok(slot_name, normalized)
        ctx.record_slot_success(slot_name)
        if slot_name == "first_name":
            ctx.update_caller_name(normalized)
        self._pending_ambiguous_resets.add(slot_name)
        if collected is not None:
            collected[slot_name] = normalized

        # Option-A next ask target: first pending slot after this one. pending
        # was computed before this slot confirmed, so it still contains slot_name;
        # cascade-cleared slots are re-collected on later turns by the pipeline.
        pending = list(pending_slots or [])
        if slot_name in pending:
            remaining = pending[pending.index(slot_name) + 1 :]
        else:
            remaining = [s for s in pending if s != slot_name]
        next_slot = next((s for s in remaining if s not in applied), "")

        # ── Case B: answer + value-less update request → open a detour.
        # The detour ask REPLACES the normal next-slot static ask.
        if update_target and update_target not in applied:
            if self._update_target_allowed(update_target, ctx, state, slot_configs):
                detour = await self._open_update_detour(
                    state,
                    update_target,
                    next_slot,
                    ctx,
                    messages,
                    flavor="answer",
                    collected=collected,
                    extracted_this_turn=normalized,
                    followup_query=followup_query,
                )
                for cleared in cascade_cleared:
                    detour.setdefault(cleared, "")
                if clear_verify:
                    detour.setdefault("member_status_verify", False)
                return normalized, detour
            # Target not updatable here — fall through as a decline.
            disposition_value = "decline"

        # ── Answer + INVALID corrected value: the awaiting slot IS confirmed,
        # but the correction was never applied — open a detour to re-collect
        # the corrected slot instead of silently dropping the bad value.
        # (Case B above wins when both an update_target and an invalid
        # correction arrive in the same turn.)
        if invalid_target and self._update_target_allowed(invalid_target, ctx, state, slot_configs):
            detour = await self._open_update_detour(
                state,
                invalid_target,
                next_slot,
                ctx,
                messages,
                flavor="invalid",
                collected=collected,
                extracted_this_turn=normalized,
                followup_query=followup_query,
            )
            for cleared in cascade_cleared:
                detour.setdefault(cleared, "")
            if clear_verify:
                detour.setdefault("member_status_verify", False)
            return normalized, detour

        # ── Disposition routing (answer_now / park / decline / none) ────────
        guard = _DISPOSITION_GUARDS.get(disposition_value, "FOLLOWUP_DECLINE")
        session_context = _mk_session_ctx(
            extracted_val=normalized,
            followup_query=followup_query,
            confirmed_values=(
                self._confirmed_slot_values(ctx, state, collected) if guard == "FOLLOWUP_ANSWER" else None
            ),
        )
        msg = await self._generate_slot_retry_response(
            state,
            slot_name,
            ctx,
            messages,
            guard=guard,
            session_context=session_context,
            extracted_this_turn=normalized,
        )

        if next_slot:
            msg = msg.rstrip() + " " + self._next_slot_ask(next_slot, slot_configs, ctx)

        interrupt = self.ask_member_with_context(state, msg, ctx)
        interrupt["awaiting_slot"] = next_slot  # "" when the pipeline finished
        interrupt["wait_count"] = 0  # non-WAIT turn resets the wait streak
        for cleared in cascade_cleared:
            interrupt[cleared] = ""
        if clear_verify:
            interrupt["member_status_verify"] = False
        if guard == "FOLLOWUP_PARK" and followup_query:
            parked = list(state.get("parked_followups") or [])
            parked.append(followup_query)
            interrupt["parked_followups"] = parked
        # Return normalized so slot_ok's persistence carries the value forward;
        # with no next_slot this matches the pre-Phase-4 return exactly, so the
        # agent's post-pipeline logic proceeds next turn.
        return normalized, interrupt

    # -------------------------------------------------------------------------
    # Per-turn slot collector — with contextual response generation
    # -------------------------------------------------------------------------

    async def _collect_slot(  # noqa: C901
        self,
        state: State,
        config: _InternalSlotConfig,
        messages: list,
        pre_extracted: str = "",
        *,
        context: Optional[ConversationContext] = None,
        is_transition: bool = False,
        decision: Optional[Any] = None,
        pending_slots: Optional[list] = None,
        slot_configs: Optional[dict] = None,
        collected: Optional[dict] = None,
    ) -> Tuple[Optional[str], Optional[dict]]:
        slot_name = config.slot_name
        normalizer = config.normalizer
        validator = config.validator
        slot_type = config.slot_type

        ctx = context or ConversationContext()
        ctx.increment_turn(self.AGENT_NAME)

        # ------------------------------------------------------------------
        # Build a human-readable slot label for use in the cannot-provide
        # escalation message (e.g. "member ID", "date of birth").
        # ------------------------------------------------------------------
        if slot_type:
            slot_label = slot_type.value.replace("_", " ")
        else:
            slot_label = slot_name.replace("_", " ")

        # ------------------------------------------------------------------
        # Response builder: use contextual builder when slot_type provided,
        # fall back to static strings for backward compatibility.
        # ------------------------------------------------------------------
        def _ask_first() -> str:
            if slot_type:
                return (
                    build_transition_prompt(slot_type, ctx)
                    if is_transition
                    else build_initial_prompt(slot_type)
                )
            return config.prompt

        # ------------------------------------------------------------------
        # 1. Already valid in state — nothing to do
        # ------------------------------------------------------------------
        existing = (state.get(slot_name) or "").strip()
        if existing:
            r = validator(existing)
            if r.valid if hasattr(r, "valid") else bool(r):
                return existing, None

        slot = self.get_slot(slot_name)

        # ------------------------------------------------------------------
        # 2. LLM pre-extracted a candidate value
        # ------------------------------------------------------------------
        if pre_extracted:
            normalized = normalizer(pre_extracted)
            if normalized:
                r = validator(normalized)
                valid = r.valid if hasattr(r, "valid") else bool(r)
                if valid:
                    # Check whether caller also said something that needs addressing.
                    # Import here to avoid circular imports at module level.
                    from agent.llm.schema import EventType

                    event_type = getattr(decision, "event_type", None)

                    if event_type == EventType.ANSWERED_WITH_FOLLOWUP:
                        # Slot will be confirmed inside the handler — corrections
                        # (Case A) must apply BEFORE slot_ok of the awaiting slot.
                        # Attempt counter is never incremented on this path.
                        return await self._handle_answered_followup(
                            state,
                            config,
                            messages,
                            normalized,
                            ctx,
                            decision=decision,
                            pending_slots=pending_slots,
                            slot_configs=slot_configs,
                            collected=collected,
                        )

                    # Clean answered path — confirm and move on immediately
                    self.slot_ok(slot_name, normalized)
                    ctx.record_slot_success(slot_name)
                    if slot_name == "first_name":
                        ctx.update_caller_name(normalized)
                    self._pending_ambiguous_resets.add(slot_name)
                    return normalized, None

            # ── CHANGE 1: cannot-provide check on rejected extraction ─────
            # The LLM extracted something but it failed normalisation /
            # validation (e.g. member_id without M-prefix, partial number).
            # Before burning a retry attempt, check whether the raw text is
            # actually "I don't have it" in disguise.
            last_user = _last_user_msg(messages)
            if detect_cannot_provide(last_user):
                self.logger.info(
                    "_collect_slot: cannot-provide detected (rejected-extraction path) — "
                    "escalating immediately",
                    extra={"slot": slot_name},
                )
                return None, self.signal_escalate(
                    state,
                    _CANNOT_PROVIDE_MSG.format(slot_label=slot_label),
                    f"{slot_name}_cannot_provide",
                    initiator="Agent",
                )
            # ── END CHANGE 1 ──────────────────────────────────────────────

            # Extracted but normaliser/validator rejected it — count as failure
            self.slot_fail(slot_name, pre_extracted)
            if slot.is_exhausted():
                return None, self.signal_escalate(
                    state,
                    build_slot_exhausted_message(slot_name),
                    f"{slot_name} exhausted",
                    initiator="Agent",
                )
            msg = await self._generate_slot_retry_response(state, slot_name, ctx, messages)
            interrupt = self.ask_member_with_context(state, msg, ctx)
            interrupt["awaiting_slot"] = slot_name
            interrupt["wait_count"] = 0  # non-WAIT turn resets the wait streak
            return None, interrupt

        # ------------------------------------------------------------------
        # 2b. No extraction — classify the turn before counting a failure
        # ------------------------------------------------------------------
        if state.get("awaiting_slot") == slot_name:
            from agent.llm.schema import EventType

            event_type = getattr(decision, "event_type", None)
            event_value = event_type.value if event_type is not None else EventType.ANSWERED.value

            # ── WAIT: caller asked for time — not a failure, not ambiguous ─
            # Detection is deliberately post-extraction only: a pre-GPT regex
            # short-circuit in agent.run() would swallow utterances like
            # "hold on... it's M451982" where a valid value follows the wait
            # phrase — extraction must see the turn first so the value wins
            # (a turn with a valid value never reaches this block).
            # The regex fallback also rescues waits the LLM mislabels as
            # ambiguous. cannot-provide outranks wait: "I don't have it"
            # falls through to the escalation checks below.
            last_user = _last_user_msg(messages)
            if (
                event_value == EventType.WAIT.value or detect_wait_request(last_user)
            ) and not detect_cannot_provide(last_user):
                wait_count = int(state.get("wait_count") or 0) + 1
                if wait_count < MAX_WAIT_TURNS:
                    msg = pick(MSG_WAIT_ACK)
                else:
                    msg = pick(MSG_WAIT_NUDGE).format(slot_label=slot_label)
                # Static response only: no slot_fail, no ambiguous_counts,
                # no generation-LLM call — waiting is not a failed attempt.
                interrupt = self.ask_member_with_context(state, msg, ctx)
                interrupt["awaiting_slot"] = slot_name
                interrupt["wait_count"] = wait_count
                return None, interrupt

            # ── CORRECTED: caller fixed a confirmed slot, not answering us ─
            if event_value == EventType.CORRECTED.value:
                corrections = {k: v for k, v in (getattr(decision, "corrections", None) or {}).items() if v}
                update_target = (getattr(decision, "update_target", None) or "").strip()

                if not corrections and not update_target:
                    # Downgrade only when BOTH corrections and update_target are
                    # empty — a bare update request (C2) is handled below.
                    self.logger.warning(
                        "_collect_slot: CORRECTED event with empty corrections{} and no "
                        "update_target — treating as ANSWERED. Slot: %s. "
                        "This is a prompt-following failure.",
                        slot_name,
                    )
                    event_value = EventType.ANSWERED.value
                elif not corrections and update_target:
                    # ── C2: bare update request ("I need to change my email") ─
                    # Detour: target becomes awaiting, current awaiting slot is
                    # preserved in correction_return_to for the pipeline to resume.
                    if self._update_target_allowed(update_target, ctx, state, slot_configs):
                        detour = await self._open_update_detour(
                            state,
                            update_target,
                            slot_name,
                            ctx,
                            messages,
                            flavor="bare",
                            collected=collected,
                        )
                        return None, detour
                    # Target not updatable here — decline: acknowledge without
                    # counting a failed attempt and re-ask the awaiting slot.
                    msg = await self._generate_slot_retry_response(
                        state,
                        slot_name,
                        ctx,
                        messages,
                        guard="FOLLOWUP_DECLINE",
                        session_context=_mk_session_ctx(
                            followup_query=f"update {update_target.replace('_', ' ')}",
                        ),
                    )
                    interrupt = self.ask_member_with_context(state, msg, ctx)
                    interrupt["awaiting_slot"] = slot_name
                    interrupt["wait_count"] = 0  # non-WAIT turn resets the wait streak
                    return None, interrupt
                else:
                    # ── C1: corrections with values — validate before acking ──
                    # An invalid corrected value must NOT be applied; open a
                    # detour to re-collect that slot instead of acknowledging it.
                    invalid_target = ""
                    for target, raw in corrections.items():
                        norm_fn, val_fn = self._resolve_slot_tools(target, slot_configs)
                        if not (norm_fn and val_fn):
                            continue
                        corrected_val = norm_fn(str(raw))
                        if not (corrected_val and val_fn(corrected_val).valid):
                            invalid_target = target
                            break
                    if invalid_target and self._update_target_allowed(
                        invalid_target, ctx, state, slot_configs
                    ):
                        detour = await self._open_update_detour(
                            state,
                            invalid_target,
                            slot_name,
                            ctx,
                            messages,
                            flavor="invalid",
                            collected=collected,
                        )
                        return None, detour

                    # Existing CORRECTED path: valid corrections were already
                    # applied by the agent (apply_corrections) — acknowledge and
                    # re-ask the awaiting slot.
                    corrected_fields = list(corrections.keys())
                    msg = await self._generate_correction_ack(
                        state, corrected_fields, slot_name, ctx, messages, decision=decision
                    )
                    interrupt = self.ask_member_with_context(state, msg, ctx)
                    interrupt["awaiting_slot"] = slot_name
                    ambiguous_counts = dict(state.get("ambiguous_counts") or {})
                    ambiguous_counts[slot_name] = 0
                    interrupt["ambiguous_counts"] = ambiguous_counts
                    interrupt["wait_count"] = 0  # non-WAIT turn resets the wait streak
                    return None, interrupt

            # ── AMBIGUOUS: caller signalled correction intent with no value ─
            if event_value == EventType.AMBIGUOUS.value:
                ambiguous_counts = dict(state.get("ambiguous_counts") or {})
                ambiguous_counts[slot_name] = ambiguous_counts.get(slot_name, 0) + 1

                # ── cannot-provide check in AMBIGUOUS branch ──────────────
                # Utterances like "sorry I dont have it my member ID",
                # "I lost the letter", "I never received a card" are frequently
                last_user = _last_user_msg(messages)
                if detect_cannot_provide(last_user):
                    self.logger.info(
                        "_collect_slot: cannot-provide detected (ambiguous path) — escalating immediately",
                        extra={"slot": slot_name},
                    )
                    return None, self.signal_escalate(
                        state,
                        _CANNOT_PROVIDE_MSG.format(slot_label=slot_label),
                        f"{slot_name}_cannot_provide",
                        initiator="Agent",
                    )
                # ── END cannot-provide check ──────────────────────────────

                # Two consecutive AMBIGUOUS turns → treat as a genuine non-answer.
                # (The count was just incremented, so the first ambiguous turn
                # arrives here as 1 and must fall through to the CLARIFY block —
                # no attempt cost; the second, as 2, burns an attempt.)
                if ambiguous_counts[slot_name] >= 2:
                    self.slot_fail(slot_name, None, is_asr=True)
                    if slot.is_exhausted():
                        return None, self.signal_escalate(
                            state,
                            build_slot_exhausted_message(slot_name),
                            f"{slot_name} exhausted",
                            initiator="Agent",
                        )
                    msg = await self._generate_slot_retry_response(state, slot_name, ctx, messages)
                    interrupt = self.ask_member_with_context(state, msg, ctx)
                    interrupt["awaiting_slot"] = slot_name
                    interrupt["ambiguous_counts"] = ambiguous_counts
                    interrupt["wait_count"] = 0  # non-WAIT turn resets the wait streak
                    return None, interrupt

                # First AMBIGUOUS turn — ask for clarification without counting a failure
                corrections = getattr(decision, "corrections", None) or {}
                if corrections:
                    corrected_fields = list(corrections.keys())
                    msg = await self._generate_correction_ack(
                        state, corrected_fields, slot_name, ctx, messages, decision=decision
                    )
                else:
                    from agent.llm.response_generator import generate_recovery_message

                    _sl_override: str | None = None
                    if slot_name == "relationship" and state.get("relationship"):
                        _sl_override = "relationship — whether they are the plan holder or dependent"
                    elif slot_name in ("phone_confirmed", "phone_confirmation") and state.get("phone_number"):
                        _digits = "".join(c for c in state["phone_number"] if c.isdigit())
                        _fmt = (
                            f"{_digits[:3]}-{_digits[3:6]}-{_digits[6:]}"
                            if len(_digits) == 10
                            else state["phone_number"]
                        )
                        _sl_override = (
                            f"phone confirmation — whether {_fmt} is still the number on file (yes or no)"
                        )
                    msg = await generate_recovery_message(
                        slot_name=slot_name,
                        attempt=0,
                        guard="CLARIFY",
                        last_messages=messages[-6:],
                        slot_label_override=_sl_override,
                        caller_name=ctx.caller_first_name,
                        confirmed_slots=dict.fromkeys(ctx.confirmed_slots, "confirmed"),
                        user_utterance=_last_user_msg(messages),
                    )
                interrupt = self.ask_member_with_context(state, msg, ctx)
                interrupt["awaiting_slot"] = slot_name
                interrupt["ambiguous_counts"] = ambiguous_counts
                interrupt["wait_count"] = 0  # non-WAIT turn resets the wait streak
                return None, interrupt

            # ── ANSWERED (default): genuine non-answer — check cannot-provide
            #    BEFORE counting a failure and retrying ─────────────────────

            # ── CHANGE 2: cannot-provide check on genuine non-answer ──────
            last_user = _last_user_msg(messages)
            if detect_cannot_provide(last_user):
                self.logger.info(
                    "_collect_slot: cannot-provide detected (no-extraction path) — escalating immediately",
                    extra={"slot": slot_name},
                )
                return None, self.signal_escalate(
                    state,
                    _CANNOT_PROVIDE_MSG.format(slot_label=slot_label),
                    f"{slot_name}_cannot_provide",
                    initiator="Agent",
                )
            # ── END CHANGE 2 ──────────────────────────────────────────────

            ambiguous_counts = dict(state.get("ambiguous_counts") or {})
            ambiguous_counts[slot_name] = 0
            self.slot_fail(slot_name, None, is_asr=True)
            if slot.is_exhausted():
                return None, self.signal_escalate(
                    state,
                    build_slot_exhausted_message(slot_name),
                    f"{slot_name} exhausted",
                    initiator="Agent",
                )
            msg = await self._generate_slot_retry_response(state, slot_name, ctx, messages)
            interrupt = self.ask_member_with_context(state, msg, ctx)
            interrupt["awaiting_slot"] = slot_name
            interrupt["ambiguous_counts"] = ambiguous_counts
            interrupt["wait_count"] = 0  # non-WAIT turn resets the wait streak
            return None, interrupt

        # ------------------------------------------------------------------
        # 3. First ask
        # ------------------------------------------------------------------
        interrupt = self.ask_member_with_context(state, _ask_first(), ctx)
        interrupt["awaiting_slot"] = slot_name
        interrupt["wait_count"] = 0  # non-WAIT turn resets the wait streak
        return None, interrupt
