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
from agent.core.models import SlotAttempt
from agent.responses.builder import (
    build_initial_prompt,
    build_transition_prompt,
)
from agent.responses.static import build_slot_exhausted_message
from agent.slots.types import SlotType
from agent.state import State
from agent.utils import _last_user_msg, _resolve_contact, detect_cannot_provide

# Contact slot types whose spoken form (digits/email) is prone to ASR /
# extraction mangling. For these, the rejected-extraction path retries the
# deterministic normalizer on the RAW user utterance before counting a failure.
_RAW_FALLBACK_SLOT_TYPES = frozenset({SlotType.FAX, SlotType.EMAIL, SlotType.PHONE_NUMBER})

# ── Empathetic "cannot provide" escalation message ────────────────────────────
# Slot-aware: {slot_label} is filled at runtime from the SlotType label.
# Keeps exactly the same warm tone as the rest of the codebase.
_CANNOT_PROVIDE_MSG = "No problem — let me connect you with a representative "


def _mk_session_ctx(
    *,
    extracted_val: str | None = None,
    pending_slots: list[str] | None = None,
) -> dict:
    """Build a lightweight session-context dict for _generate_slot_retry_response."""
    ctx: dict = {}
    if extracted_val is not None:
        ctx["extracted_val"] = extracted_val
    if pending_slots is not None:
        ctx["pending_slots"] = pending_slots
    return ctx


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
            confirmed_slots=dict.fromkeys(ctx.confirmed_slots, "confirmed"),
            user_utterance=_last_user_msg(messages),
            extracted_value=extracted_this_turn
            if extracted_this_turn is not None
            else sc.get("extracted_val"),
            pending_slots=sc.get("pending_slots"),
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
                    # Confirm the slot — happens once regardless of event_type
                    self.slot_ok(slot_name, normalized)
                    ctx.record_slot_success(slot_name)
                    if slot_name == "first_name":
                        ctx.update_caller_name(normalized)
                    self._pending_ambiguous_resets.add(slot_name)

                    # Check whether caller also said something that needs addressing.
                    # Import here to avoid circular imports at module level.
                    from agent.llm.schema import EventType

                    event_type = getattr(decision, "event_type", None)

                    if event_type == EventType.ANSWERED_WITH_FOLLOWUP:
                        # Slot is confirmed — do NOT increment attempt counter.
                        # Call generation LLM to acknowledge the value and address
                        # whatever the caller asked alongside their answer.
                        msg = await self._generate_slot_retry_response(
                            state,
                            slot_name,
                            ctx,
                            messages,
                            guard="ANSWERED_WITH_FOLLOWUP",
                            session_context=_mk_session_ctx(extracted_val=normalized),
                            extracted_this_turn=normalized,
                        )
                        interrupt = self.ask_member_with_context(state, msg, ctx)
                        # awaiting_slot is empty — slot is confirmed, not waiting again
                        interrupt["awaiting_slot"] = ""
                        # Return normalized so pipeline records it in collected
                        return normalized, interrupt

                    # Clean answered path — move on immediately
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

            # ── Raw-utterance fallback for contact slots ──────────────────
            # The extraction LLM mangles spoken contact details (e.g. a fax
            # dictated word-by-word comes back with an extra digit and fails
            # 10-digit validation). The deterministic normalizer applied to the
            # RAW utterance frequently recovers the correct value, so try it
            # before counting a failed attempt / re-asking. Gated to contact
            # slot types so other slots keep their stricter extraction contract.
            if slot_type in _RAW_FALLBACK_SLOT_TYPES:
                recovered = _resolve_contact(pre_extracted, last_user, normalizer, validator)
                if recovered:
                    self.logger.info(
                        "_collect_slot: recovered contact value from raw utterance "
                        "after extraction failed validation",
                        extra={"slot": slot_name},
                    )
                    self.slot_ok(slot_name, recovered)
                    ctx.record_slot_success(slot_name)
                    self._pending_ambiguous_resets.add(slot_name)
                    return recovered, None
            # ── END raw-utterance fallback ────────────────────────────────

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
            return None, interrupt

        # ------------------------------------------------------------------
        # 2b. No extraction — classify the turn before counting a failure
        # ------------------------------------------------------------------
        if state.get("awaiting_slot") == slot_name:
            from agent.llm.schema import EventType

            event_type = getattr(decision, "event_type", None)
            event_value = event_type.value if event_type is not None else EventType.ANSWERED.value

            # ── CORRECTED: caller fixed a confirmed slot, not answering us ─
            if event_value == EventType.CORRECTED.value:
                if not (getattr(decision, "corrections", None) or {}):
                    self.logger.warning(
                        "_collect_slot: CORRECTED event with empty corrections{} — "
                        "treating as ANSWERED. Slot: %s. This is a prompt-following failure.",
                        slot_name,
                    )
                    event_value = EventType.ANSWERED.value
                else:
                    corrected_fields = list((getattr(decision, "corrections", None) or {}).keys())
                    msg = await self._generate_correction_ack(
                        state, corrected_fields, slot_name, ctx, messages, decision=decision
                    )
                    interrupt = self.ask_member_with_context(state, msg, ctx)
                    interrupt["awaiting_slot"] = slot_name
                    ambiguous_counts = dict(state.get("ambiguous_counts") or {})
                    ambiguous_counts[slot_name] = 0
                    interrupt["ambiguous_counts"] = ambiguous_counts
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

                # Two consecutive AMBIGUOUS turns → treat as a genuine non-answer
                if ambiguous_counts[slot_name] >= 1:
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
            return None, interrupt

        # ------------------------------------------------------------------
        # 3. First ask
        # ------------------------------------------------------------------
        interrupt = self.ask_member_with_context(state, _ask_first(), ctx)
        interrupt["awaiting_slot"] = slot_name
        return None, interrupt
