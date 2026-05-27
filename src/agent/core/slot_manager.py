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
from agent.responses.static import MSG_SLOT_EXHAUSTED
from agent.slots.types import SlotType
from agent.state import State
from agent.utils import _last_user_msg


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
    ) -> str:
        # Lazy import: core.slot_manager → llm.response_generator → llm.config → core (via schema);
        # importing at module level would create a core → llm → core cycle.
        from agent.llm.response_generator import generate_recovery_message

        slot_state = self.get_slot(slot_name)
        # For dynamic slots whose valid options are only known from the SF lookup,
        # build a compact context label from state rather than the static _SLOT_LABELS
        # dict. These are LLM context strings — not spoken sentences — so they follow
        # the same concise, instructional style as the existing static entries.
        slot_label_override: str | None = None
        if slot_name == "relationship" and state.get("relationship"):
            # e.g. "relationship — whether they are the plan holder, subscriber, or spouse"
            slot_label_override = f"relationship — whether they are the {state['relationship']}"
        elif slot_name in ("phone_confirmed", "phone_confirmation") and state.get("phone_number"):
            # Format raw digits to readable form: "6175554101" → "617-555-4101"
            digits = "".join(c for c in state["phone_number"] if c.isdigit())
            formatted = (
                f"{digits[:3]}-{digits[3:6]}-{digits[6:]}" if len(digits) == 10 else state["phone_number"]
            )
            # e.g. "phone confirmation — whether 617-555-4101 is still the number on file (yes or no)"
            slot_label_override = (
                f"phone confirmation — whether {formatted} is still the number on file (yes or no)"
            )
        text = await generate_recovery_message(
            slot_name=slot_name,
            attempt=slot_state.attempt_count,
            guard="RETRY",  # attempt_count already incremented; LLM re-asks firmly
            last_messages=messages[-4:],
            slot_label_override=slot_label_override,
            caller_name=ctx.caller_first_name,
            confirmed_slots=dict.fromkeys(ctx.confirmed_slots, "confirmed"),
            user_utterance=_last_user_msg(messages),
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

        return await generate_recovery_message(
            slot_name=awaiting_slot,
            attempt=0,
            guard="CORRECTION",
            last_messages=messages[-6:],
            slot_label_override=(
                f"caller corrected {corrected_label}"
                + (f" to {corrected_value}" if corrected_value else "")
                + f" — now re-ask for {awaiting_slot.replace('_', ' ')}"
            ),
            caller_name=ctx.caller_first_name,
            confirmed_slots=dict.fromkeys(ctx.confirmed_slots, "confirmed"),
            user_utterance=_last_user_msg(messages[-6:]),
        )

    # -------------------------------------------------------------------------
    # Per-turn slot collector — with contextual response generation
    # -------------------------------------------------------------------------

    async def _collect_slot(
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
                if r.valid if hasattr(r, "valid") else bool(r):
                    self.slot_ok(slot_name, normalized)
                    ctx.record_slot_success(slot_name)
                    if slot_name == "first_name":
                        ctx.update_caller_name(normalized)
                    self._pending_ambiguous_resets.add(slot_name)
                    return normalized, None

            self.slot_fail(slot_name, pre_extracted)
            if slot.is_exhausted():
                return None, self.signal_escalate(
                    state, MSG_SLOT_EXHAUSTED, f"{slot_name} exhausted", initiator="Agent"
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
                    # Fall through to ANSWERED branch by overriding event_value
                    event_value = EventType.ANSWERED.value
                else:
                    corrected_fields = list((getattr(decision, "corrections", None) or {}).keys())
                    msg = await self._generate_correction_ack(
                        state, corrected_fields, slot_name, ctx, messages, decision=decision
                    )
                    interrupt = self.ask_member_with_context(state, msg, ctx)
                    interrupt["awaiting_slot"] = slot_name
                    # Reset ambiguous counter for this slot — clean slate after correction
                    ambiguous_counts = dict(state.get("ambiguous_counts") or {})
                    ambiguous_counts[slot_name] = 0
                    interrupt["ambiguous_counts"] = ambiguous_counts
                    # No slot_fail — attempt counter untouched
                    return None, interrupt

            # ── AMBIGUOUS: caller signalled correction intent with no value ─
            if event_value == EventType.AMBIGUOUS.value:
                ambiguous_counts = dict(state.get("ambiguous_counts") or {})
                ambiguous_counts[slot_name] = ambiguous_counts.get(slot_name, 0) + 1

                # Two consecutive AMBIGUOUS turns → treat as a genuine non-answer
                if ambiguous_counts[slot_name] >= 2:
                    ambiguous_counts[slot_name] = 0  # reset before escalation path
                    self.slot_fail(slot_name, None, is_asr=True)
                    if slot.is_exhausted():
                        return None, self.signal_escalate(
                            state, MSG_SLOT_EXHAUSTED, f"{slot_name} exhausted", initiator="Agent"
                        )
                    msg = await self._generate_slot_retry_response(state, slot_name, ctx, messages)
                    interrupt = self.ask_member_with_context(state, msg, ctx)
                    interrupt["awaiting_slot"] = slot_name
                    interrupt["ambiguous_counts"] = ambiguous_counts
                    return None, interrupt

                # First AMBIGUOUS turn — ask for clarification without counting a failure
                corrections = getattr(decision, "corrections", None) or {}
                if corrections:
                    # Caller signalled correction intent AND provided some data → treat like
                    # a CORRECTED event; recovery guard = "CORRECTION" to produce an ack tone
                    corrected_fields = list(corrections.keys())
                    msg = await self._generate_correction_ack(
                        state, corrected_fields, slot_name, ctx, messages, decision=decision
                    )
                else:
                    # Caller signalled correction intent but gave NO new value (pure ambiguity)
                    # → recovery guard = "CLARIFY"; re-ask gently, no attempt penalty
                    from agent.llm.response_generator import generate_recovery_message

                    _sl_override: str | None = None
                    if slot_name == "relationship" and state.get("relationship"):
                        _sl_override = f"relationship — whether they are the {state['relationship']}"
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
                        guard="CLARIFY",  # no attempt cost; LLM re-asks gently
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

            # ── ANSWERED (default): genuine non-answer — count failure ──────
            # Reset ambiguous counter — this was a real answer attempt, not ambiguous
            ambiguous_counts = dict(state.get("ambiguous_counts") or {})
            ambiguous_counts[slot_name] = 0
            self.slot_fail(slot_name, None, is_asr=True)
            if slot.is_exhausted():
                return None, self.signal_escalate(
                    state, MSG_SLOT_EXHAUSTED, f"{slot_name} exhausted", initiator="Agent"
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
