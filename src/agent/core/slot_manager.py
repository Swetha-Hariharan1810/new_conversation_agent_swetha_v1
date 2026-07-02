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
from agent.responses.grounding import turn_grounding_allowlist
from agent.responses.static import build_slot_exhausted_message
from agent.slots.types import SlotType
from agent.state import State
from agent.utils import _last_user_msg, detect_cannot_provide, detect_stalling

# Max consecutive "give me a moment" turns acknowledged before a stall is treated
# as a normal non-answer (bounds a runaway stall without burning real attempts).
MAX_STALLS = 5

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
    # Shared stalling guard — for hand-coded slot flows outside _collect_slot
    # -------------------------------------------------------------------------

    def check_stalling(
        self,
        state: State,
        messages: list,
        decision: Optional[Any],
        slot_name: str,
        *,
        context: Optional[ConversationContext] = None,
        extra_updates: dict | None = None,
    ) -> Optional[dict]:
        """
        Stalling guard for slot flows that are hand-coded directly in an
        agent's run() (e.g. yes/no confirmations) instead of going through
        _collect_slot. Mirrors _collect_slot's own STALLING branch: when the
        caller is asking for a moment ("give me a few seconds", "hold on"),
        acknowledge ONLY — never re-ask the slot question and never count a
        failed attempt — bounded by MAX_STALLS so a runaway stall cannot loop
        forever.

        Returns an interrupt dict when a stall was acknowledged; returns None
        when the turn is not a stall (or stalls are exhausted), so the caller
        falls through to its normal non-answer handling.
        """
        from agent.llm.schema import EventType
        from agent.responses import turn_acts

        evt = getattr(decision, "event_type", None)
        is_stalling = (evt is not None and evt.value == EventType.STALLING.value) or detect_stalling(
            _last_user_msg(messages)
        )
        if not is_stalling:
            return None

        stall = self.get_slot(f"{slot_name}#stall")
        if stall.attempt_count >= MAX_STALLS:
            return None

        stall.record_attempt(None, success=False)  # bound runaway stalls only
        msg = turn_acts.render_stalling_ack(attempt=stall.attempt_count)
        interrupt = self.ask_member_with_context(state, msg, context) if context else self.ask_member(state, msg)
        interrupt["awaiting_slot"] = slot_name  # still waiting; no real-slot failure
        for k, v in (extra_updates or {}).items():
            interrupt[k] = v
        self.logger.info(
            "check_stalling: stalling acknowledged (no retry counted)",
            extra={"slot": slot_name, "stall_count": stall.attempt_count},
        )
        return interrupt

    # -------------------------------------------------------------------------
    # LLM 2 retry response helper
    # -------------------------------------------------------------------------

    @staticmethod
    def _dynamic_slot_label(state: State, slot_name: str) -> tuple[str | None, str | None]:
        """Runtime ``(label, directive)`` override for slots whose phrasing
        depends on live state (relationship options, the phone number on file).
        The label is a SHORT NOUN PHRASE (safe to interpolate into a spoken
        fallback template); the directive carries the instruction-style guidance
        and is only ever rendered as the generator's ``Guidance:`` context line.
        Returns ``(None, None)`` for fixed slots, which use the static dicts."""
        if slot_name == "relationship" and state.get("relationship"):
            return (
                "relationship to the plan holder",
                "Ask whether they are the plan holder or a dependent.",
            )
        if slot_name in ("phone_confirmed", "phone_confirmation") and state.get("phone_number"):
            digits = "".join(c for c in state["phone_number"] if c.isdigit())
            formatted = (
                f"{digits[:3]}-{digits[3:6]}-{digits[6:]}" if len(digits) == 10 else state["phone_number"]
            )
            return (
                "phone number confirmation",
                f"Ask whether {formatted} is still the number on file (yes or no).",
            )
        return None, None

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
        label_override, directive = self._dynamic_slot_label(state, slot_name)
        sc = session_context or {}
        extracted = extracted_this_turn if extracted_this_turn is not None else sc.get("extracted_val")
        text = await generate_recovery_message(
            slot_name=slot_name,
            attempt=slot_state.attempt_count,
            guard=guard,
            last_messages=messages[-4:],
            slot_label_override=label_override,
            generator_directive=directive,
            caller_name=ctx.caller_first_name,
            confirmed_slots=dict.fromkeys(ctx.confirmed_slots, "confirmed"),
            user_utterance=_last_user_msg(messages),
            extracted_value=extracted,
            pending_slots=sc.get("pending_slots"),
            grounded_values=turn_grounding_allowlist(
                state, ctx, extracted_value=extracted, answered_inline=None, slot_name=slot_name
            ),
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

        # The label stays a plain noun phrase (the slot still being collected);
        # the correction instruction travels on the directive, never the label.
        readback: list[str] = []
        if corrected_fields and corrected_fields[0] in ("first_name", "last_name") and corrected_value:
            # Names: read back explicitly so caller hears their name confirmed
            directive = (
                f"The caller corrected their {corrected_label} to '{corrected_value}' — "
                f"acknowledge by explicitly saying the corrected {corrected_label} "
                f"is '{corrected_value}', "
                f"then ask for their {awaiting_slot.replace('_', ' ')}."
            )
            readback.append(corrected_value)
        elif corrected_value:
            # Sensitive slots (member_id, dob etc.): acknowledge WITHOUT reading
            # the value back out loud — just confirm the update and re-ask
            directive = (
                f"The caller corrected their {corrected_label} — "
                f"acknowledge the correction without repeating the value, "
                f"then ask for their {awaiting_slot.replace('_', ' ')}."
            )
        else:
            # No new value provided: re-ask naturally for awaiting_slot
            directive = (
                f"The caller corrected {corrected_label} — now re-ask for "
                f"their {awaiting_slot.replace('_', ' ')}."
            )

        return await generate_recovery_message(
            slot_name=awaiting_slot,
            attempt=0,
            guard="CORRECTION",
            last_messages=messages[-6:],
            generator_directive=directive,
            caller_name=ctx.caller_first_name,
            confirmed_slots=dict.fromkeys(ctx.confirmed_slots, "confirmed"),
            user_utterance=_last_user_msg(messages[-6:]),
            grounded_values=turn_grounding_allowlist(
                state,
                ctx,
                extracted_value=None,
                answered_inline=None,
                slot_name=awaiting_slot,
                readback_values=readback or None,
            ),
        )

    # -------------------------------------------------------------------------
    # Phase 3: compose a multi-intent turn as ONE generated sentence
    # -------------------------------------------------------------------------

    @staticmethod
    def _next_ask_label(slot: str) -> str:
        """Human ask-label for the next pipeline slot (for the generator's final
        clause). Reuses the response-generator slot labels."""
        from agent.llm.response_generator import _SLOT_LABELS

        return _SLOT_LABELS.get(slot, (slot or "").replace("_", " "))

    def _fallback_multi_intent_text(
        self,
        *,
        speech_act: str,
        correction_field: str,
        slot_value: Optional[str],
        parked_owners: list[str],
        declined: bool,
        next_ask_label: Optional[str],
    ) -> str:
        """Deterministic templated fallback if generation fails/times out — a valid
        (if less fluid) sentence, so a multi-intent turn is never dropped."""
        from agent.orchestration.resolver import CORRECTION_ACK
        from agent.responses import turn_acts

        parts: list[str] = []
        if speech_act == CORRECTION_ACK and correction_field:
            parts.append(turn_acts.render_correction_ack(field=correction_field, slot_value=slot_value))
        elif parked_owners:
            parts.append(turn_acts.render_multi_intent_ack(parked_owners))
        if declined:
            parts.append(turn_acts.render_unsupported_decline())
        if next_ask_label and speech_act != CORRECTION_ACK:
            parts.append(turn_acts.render_re_ask(slot_label=next_ask_label))
        return " ".join(p for p in parts if p) or turn_acts.render_open_redirect()

    async def _compose_multi_intent_via_generator(
        self,
        state: State,
        ctx: ConversationContext,
        slot_name: str,
        slot_value: Optional[str],
        plan,
        outcome,
        *,
        slot_answered: bool,
        slot_label: str,
        pending_slots: Optional[list[str]],
        extra_updates: Optional[dict] = None,
        next_node_when_done: Optional[str] = None,
    ) -> Optional[dict]:
        """Narrate the resolver outcome as ONE generated sentence (Phase 3).

        Handles multi_intent_ack / unsupported_decline / open_redirect and
        correction_ack. Splits surviving in-scope independents into answered-inline
        (grounded, default) vs. parked (PARK_ANSWERABLE=true), folds a brief decline
        for out-of-scope asides, and ends with the next slot ask. Returns None for
        outcomes this path doesn't own (falls back to the templated path). The
        generated text is grounding-checked; on any doubt it uses the deterministic
        template so no unvalidated value is ever emitted."""
        from agent.core import flags
        from agent.llm.response_generator import SPEECH_ACT_MULTI_INTENT, generate_recovery_message
        from agent.orchestration.resolver import (
            CORRECTION_ACK,
            MULTI_INTENT_ACK,
            OPEN_REDIRECT,
            UNSUPPORTED_DECLINE,
        )
        from agent.responses import turn_acts

        if outcome.speech_act not in (CORRECTION_ACK, MULTI_INTENT_ACK, UNSUPPORTED_DECLINE, OPEN_REDIRECT):
            return None
        slot = self.get_slot(slot_name)

        # Correction: acknowledge + rewind to the corrected value's owner.
        correction_field = ""
        rewind = None
        if outcome.speech_act == CORRECTION_ACK:
            if not slot_answered:
                return None  # bare correction → existing CORRECTED path
            rewind = outcome.rewind_target
            correction_field = plan.correction.field if (plan and getattr(plan, "correction", None)) else ""
            if not rewind or not correction_field:
                return None

        # Split independents: answer inline (grounded) vs. park (the dial).
        park_all = flags.park_answerable()
        answered_inline: list[str] = []
        parked_owners: list[str] = []
        for d in getattr(outcome, "independents_detail", []) or []:
            if d.get("answerable") and not park_all:
                answered_inline.append(d["answer"])
            else:
                parked_owners.append(d["owner"])
        declined = bool(outcome.declined)

        # Next ask (deterministic, from the pipeline order).
        next_ask_slot = ""
        next_ask_label: Optional[str] = None
        if not correction_field:
            if slot_answered and pending_slots:
                next_ask_slot = pending_slots[0]
                next_ask_label = self._next_ask_label(next_ask_slot)
            elif not slot_answered:
                next_ask_slot = slot_name
                next_ask_label = slot_label

        # State updates: accept the slot; enqueue ONLY parked owners (inline-answered
        # are handled this turn, not queued); carry dirty flags from the resolver.
        updates = dict(outcome.state_updates or {})
        queue = list(state.get("intent_queue") or [])
        for o in parked_owners:
            if o not in queue:
                queue.append(o)
        if parked_owners:
            updates["intent_queue"] = queue
        else:
            updates.pop("intent_queue", None)

        fallback = self._fallback_multi_intent_text(
            speech_act=outcome.speech_act,
            correction_field=correction_field,
            slot_value=slot_value,
            parked_owners=parked_owners,
            declined=declined,
            next_ask_label=next_ask_label,
        )

        # Grounding guardrail is enforced inside generate_recovery_message (Phase 4):
        # the composed sentence may state only values grounded this turn (accepted
        # answer + inline answers + a known first name + any value deliberately
        # read back for the slot being asked); on any leak it returns the
        # deterministic ``fallback`` template instead.
        allowed = turn_grounding_allowlist(
            state,
            ctx,
            extracted_value=slot_value,
            answered_inline=answered_inline or None,
            slot_name=next_ask_slot or slot_name,
        )
        msg = await generate_recovery_message(
            slot_name=next_ask_slot or slot_name,
            attempt=slot.attempt_count,
            guard=SPEECH_ACT_MULTI_INTENT,
            speech_act=SPEECH_ACT_MULTI_INTENT,
            last_messages=(state.get("messages") or [])[-4:],
            slot_label_override=next_ask_label,
            caller_name=ctx.caller_first_name,
            confirmed_slots=dict.fromkeys(ctx.confirmed_slots, "confirmed"),
            user_utterance=_last_user_msg(state.get("messages") or []),
            extracted_value=slot_value if (slot_answered and slot_value) else None,
            parked=parked_owners or None,
            declined=declined,
            answered_inline=answered_inline or None,
            next_ask=next_ask_label,
            correction_field=(turn_acts.field_label(correction_field) if correction_field else None),
            grounded_values=allowed,
            fallback_text=fallback,
        )

        interrupt = self.ask_member_with_context(state, msg, ctx)
        for k, v in (extra_updates or {}).items():
            interrupt[k] = v
        for k, v in updates.items():
            interrupt[k] = v
        interrupt["is_interrupt"] = True
        if correction_field:
            interrupt["next_node"] = rewind
            interrupt["awaiting_slot"] = correction_field
            if correction_field == "zip_code":
                interrupt["zip_code_used"] = ""
        else:
            interrupt["awaiting_slot"] = next_ask_slot if slot_answered else slot_name
            # Bug 2: a confirmed yes/no slot with parked secondaries completes the
            # current agent's step — the composed sentence already ends with the
            # NEXT step's ask; route the answer to that step's agent.
            if slot_answered and next_node_when_done:
                interrupt["next_node"] = next_node_when_done
        self.logger.info(
            "slot_manager: multi-intent turn composed (one voice)",
            extra={
                "slot": slot_name,
                "speech_act": outcome.speech_act,
                "answered_inline": len(answered_inline),
                "parked": list(parked_owners),
                "declined": declined,
                "next_ask": next_ask_slot,
            },
        )
        return interrupt

    # -------------------------------------------------------------------------
    # Phase 3B/3C: live application of a resolver outcome on a slot-answered turn
    # -------------------------------------------------------------------------

    async def _apply_resolver_outcome(  # noqa: C901
        self,
        state: State,
        ctx: ConversationContext,
        slot_name: str,
        slot_value: Optional[str],
        plan,
        outcome,
        *,
        slot_answered: bool = True,
        slot_label: Optional[str] = None,
        pending_slots: Optional[list[str]] = None,
        extra_updates: Optional[dict] = None,
        next_node_when_done: Optional[str] = None,
        ask_next_on_answered: bool = False,
    ) -> Optional[dict]:
        """Build the live interrupt for an actionable resolver outcome, or None.

        Phase 3 (behind MULTI_INTENT_LIVE): the outcome is narrated by the GENERATOR
        as ONE natural sentence (accept → answer inline → note parked → decline →
        ask next), composed from the resolver's structured decomposition. When the
        flag is off, the deterministic templated path below runs unchanged (so
        single-intent and default flows are byte-identical to Phase 1).

        Runs at the shared chokepoint for every agent (Phase 3D), both when the
        awaiting slot was answered AND when it was not (``slot_answered``). Handles,
        with templated speech (no generative surface):

          * correction_ack — accept the answer (if any), acknowledge the
            correction, and route to the corrected value's owner to re-resolve it.
            Invalidating corrections also flip the dependent artifact dirty (Phase
            1's gate then blocks delivery). Only applied when the slot was answered;
            a bare correction with no answer uses the existing CORRECTED path.
          * multi_intent_ack — enqueue the parked independent(s) for draining and
            speak a templated acknowledgement (no per-parked-intent fan-out).
          * unsupported_decline / open_redirect — give the unanswerable side
            question a spoken outcome; never act on it.
          * cross_slot_accept — the utterance answered a DIFFERENT pending slot of
            this agent; accept that value and re-ask the awaiting slot (no failed
            attempt is counted, because this interrupt returns before slot_fail).

        Phase 2 (hand-coded confirmation flows):
          * ``extra_updates`` — merged into ANY interrupt returned, so a hand-coded
            call site's flow keys (provider_type, zip_code, …) persist.
          * ``ask_next_on_answered`` — when True and the slot WAS answered, the
            spoken outcome ends with the ask for ``pending_slots[0]`` and
            ``awaiting_slot`` moves there (Bug 2: accept → park-ack → next-step
            ask in ONE turn), applying the resolver's ``state_updates`` in full.
          * ``next_node_when_done`` — when the answered slot completes this
            agent's step, route the next turn to that agent.
        Pipeline call sites pass none of these, so their behavior is unchanged.

        When the slot was NOT answered, the spoken outcome is followed by a re-ask
        of the same slot (awaiting kept), so the primary collection continues.
        Returns None for clean answers and re_ask/clarify (existing logic).
        """
        from agent.core import flags
        from agent.orchestration.resolver import (
            CORRECTION_ACK,
            CROSS_SLOT_ACCEPT,
            MULTI_INTENT_ACK,
            OPEN_REDIRECT,
            UNSUPPORTED_DECLINE,
        )
        from agent.responses import turn_acts

        if outcome is None:
            return None
        slot = self.get_slot(slot_name)
        label = slot_label or slot_name.replace("_", " ")
        caller_updates = dict(extra_updates or {})

        # ── Phase 3: narrate the whole turn as ONE generated sentence ──────────
        if flags.multi_intent_live():
            composed = await self._compose_multi_intent_via_generator(
                state,
                ctx,
                slot_name,
                slot_value,
                plan,
                outcome,
                slot_answered=slot_answered,
                slot_label=label,
                pending_slots=pending_slots,
                extra_updates=caller_updates or None,
                next_node_when_done=next_node_when_done,
            )
            if composed is not None:
                return composed
            # Fall through to the deterministic template path as the fallback.

        def _finish(msg: str, *, extra_updates: dict | None = None, keep_slot: bool) -> dict:
            interrupt = self.ask_member_with_context(state, msg, ctx)
            for k, v in caller_updates.items():
                interrupt[k] = v
            for k, v in (extra_updates or {}).items():
                interrupt[k] = v
            interrupt["awaiting_slot"] = slot_name if keep_slot else ""
            interrupt["is_interrupt"] = True
            return interrupt

        def _next_step_ask() -> tuple[str, str]:
            """Bug 2 completion: the yes/no slot WAS answered and the agent's step
            is done — the same sentence must end with the NEXT step's ask.
            Returns (ask_sentence, next_slot), or ("", "") when not applicable."""
            if not (slot_answered and ask_next_on_answered and pending_slots):
                return "", ""
            next_slot = pending_slots[0]
            return turn_acts.render_next_ask(slot_label=self._next_ask_label(next_slot)), next_slot

        def _route_done(interrupt: dict, next_slot: str) -> dict:
            """Move awaiting to the next step's slot and route to its agent."""
            if next_slot:
                interrupt["awaiting_slot"] = next_slot
            if next_node_when_done:
                interrupt["next_node"] = next_node_when_done
            return interrupt

        # ── correction_ack — only when the slot was answered (else CORRECTED path)
        if outcome.speech_act == CORRECTION_ACK:
            if not slot_answered:
                return None
            rewind = outcome.rewind_target
            field = plan.correction.field if (plan and getattr(plan, "correction", None)) else ""
            if not rewind or not field:
                return None
            invalidating = bool(outcome.dirty and any(outcome.dirty.values()))
            msg = turn_acts.render_correction_ack(
                field=field,
                attempt=slot.attempt_count,
                slot_value=slot_value if invalidating else None,
            )
            if outcome.declined:  # co-occurring unsupported request → decline in-line
                msg = f"{msg} {turn_acts.render_unsupported_decline(attempt=slot.attempt_count)}"
            interrupt = self.ask_member_with_context(state, msg, ctx)
            for k, v in caller_updates.items():
                interrupt[k] = v
            for k, v in (outcome.state_updates or {}).items():
                interrupt[k] = v
            interrupt["next_node"] = rewind  # rewind to the corrected value's owner
            interrupt["awaiting_slot"] = field
            interrupt["is_interrupt"] = True
            if field == "zip_code":
                interrupt["zip_code_used"] = ""  # force provider_search to re-resolve
            self.logger.info(
                "slot_manager: correction applied live",
                extra={
                    "slot": slot_name,
                    "corrected_field": field,
                    "rewind": rewind,
                    "invalidating": invalidating,
                },
            )
            return interrupt

        # ── multi_intent_ack — park independent(s) + speak the ack ──────────────
        if outcome.speech_act == MULTI_INTENT_ACK and outcome.parked:
            ack = turn_acts.render_multi_intent_ack(outcome.parked, attempt=slot.attempt_count)
            if outcome.declined:  # co-occurring unsupported request → decline in-line
                ack = f"{ack} {turn_acts.render_unsupported_decline(attempt=slot.attempt_count)}"
            next_ask, next_slot = _next_step_ask()
            if next_ask:
                msg = f"{ack} {next_ask}"
            else:
                msg = ack if slot_answered else f"{ack} {turn_acts.render_re_ask(slot_label=label)}"
            extra = {}
            if slot_answered and ask_next_on_answered:
                # Bug 2: the accept must take effect in full (accepted value +
                # queue), not just the intent_queue delta.
                extra.update(outcome.state_updates or {})
            elif "intent_queue" in (outcome.state_updates or {}):
                extra["intent_queue"] = outcome.state_updates["intent_queue"]
            self.logger.info(
                "slot_manager: multi-intent acknowledged",
                extra={"slot": slot_name, "parked": list(outcome.parked)},
            )
            interrupt = _finish(msg, extra_updates=extra, keep_slot=not slot_answered)
            if slot_answered and ask_next_on_answered:
                interrupt = _route_done(interrupt, next_slot)
            return interrupt

        # ── unsupported / open redirect — spoken outcome, never act ─────────────
        if outcome.speech_act in (UNSUPPORTED_DECLINE, OPEN_REDIRECT):
            base = (
                turn_acts.render_unsupported_decline(attempt=slot.attempt_count)
                if outcome.speech_act == UNSUPPORTED_DECLINE
                else turn_acts.render_open_redirect(attempt=slot.attempt_count)
            )
            next_ask, next_slot = _next_step_ask()
            if next_ask:
                msg = f"{base} {next_ask}"
            else:
                msg = base if slot_answered else f"{base} {turn_acts.render_re_ask(slot_label=label)}"
            self.logger.info(
                "slot_manager: side-question given spoken outcome",
                extra={"slot": slot_name, "speech_act": outcome.speech_act},
            )
            interrupt = _finish(msg, keep_slot=not slot_answered)
            if slot_answered and ask_next_on_answered:
                interrupt = _route_done(interrupt, next_slot)
            return interrupt

        # ── cross_slot_accept — the answer belonged to a DIFFERENT pending slot ──
        if outcome.speech_act == CROSS_SLOT_ACCEPT and outcome.state_updates:
            field, value = next(iter(outcome.state_updates.items()))
            msg = turn_acts.render_cross_slot_accept(
                field=field, value=value, slot_label=label, attempt=slot.attempt_count
            )
            self.logger.info(
                "slot_manager: cross-slot answer accepted",
                extra={"slot": slot_name, "accepted_slot": field},
            )
            # keep_slot: the awaiting slot is still open — re-asked in the same
            # sentence, and NO failed attempt was counted against it.
            return _finish(msg, extra_updates=dict(outcome.state_updates), keep_slot=True)

        return None

    # -------------------------------------------------------------------------
    # Phase 1: one voice — route the happy-path ask/transition through the generator
    # -------------------------------------------------------------------------

    async def _first_ask_message(
        self,
        state: State,
        config: "_InternalSlotConfig",
        ctx: ConversationContext,
        messages: list,
        *,
        is_transition: bool,
        template_text: str,
    ) -> str:
        """Text for a first-ask / transition turn.

        Default (UNIFIED_VOICE off): the template string, exactly as before.
        UNIFIED_VOICE on (and a typed slot): the SAME grounded generator that
        speaks retries/clarifies/corrections, with speech act ``ask`` or
        ``transition`` — so every turn has one voice. The template is passed as
        ``fallback_text`` so a generation failure/timeout can never drop the turn.
        """
        from agent.core import flags

        if not (flags.unified_voice() and config.slot_type):
            return template_text

        from agent.llm.response_generator import (
            SPEECH_ACT_ASK,
            SPEECH_ACT_TRANSITION,
            generate_recovery_message,
        )

        speech_act = SPEECH_ACT_TRANSITION if is_transition else SPEECH_ACT_ASK
        label_override, directive = self._dynamic_slot_label(state, config.slot_name)
        return await generate_recovery_message(
            slot_name=config.slot_name,
            attempt=0,
            guard=speech_act,
            speech_act=speech_act,
            last_messages=messages[-4:],
            slot_label_override=label_override,
            generator_directive=directive,
            caller_name=ctx.caller_first_name,
            confirmed_slots=dict.fromkeys(ctx.confirmed_slots, "confirmed"),
            user_utterance=_last_user_msg(messages),
            grounded_values=turn_grounding_allowlist(
                state, ctx, extracted_value=None, answered_inline=None, slot_name=config.slot_name
            ),
            fallback_text=template_text,
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
        pending_slots: Optional[list[str]] = None,
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
        # Phase 3A/3B → Phase 2 turn gate: shared understanding decode + resolver.
        # Runs the single resolver on every slot-collection turn in every agent
        # through ``understand_turn`` — the ONE per-turn chokepoint (fast path,
        # decode budget, idempotence cache), so an agent that both runs its
        # guards and a pipeline never decodes the same user message twice.
        # Guarded so it can never break a live turn.
        # ------------------------------------------------------------------
        _turn_plan = None
        _turn_outcome = None
        try:
            from agent.orchestration.turn_gate import understand_turn

            _turn_plan, _turn_outcome = await understand_turn(
                state,
                utterance=_last_user_msg(messages),
                awaiting_slot=state.get("awaiting_slot") or slot_name,
                decision=decision,
                agent_name=getattr(self, "AGENT_NAME", ""),
            )
        except Exception:  # observability must never break a turn
            self.logger.debug("understanding decode/resolve failed", exc_info=True)

        # Phase 2: run the log-only TurnPlan observer (the LLM decode in SHADOW).
        # It only logs a turnplan_shadow comparison and never feeds this turn, so
        # the live path above is unchanged. No-op unless TURNPLAN_DECODE=shadow.
        try:
            from agent.orchestration.shadow import run_turnplan_observer

            await run_turnplan_observer(
                state,
                utterance=_last_user_msg(messages),
                awaiting_slot=state.get("awaiting_slot") or slot_name,
                decision=decision,
                agent_name=getattr(self, "AGENT_NAME", ""),
            )
        except Exception:  # observability must never break a turn
            self.logger.debug("turnplan observer failed", exc_info=True)

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

                    # ── Phase 3B/3C: resolver outcome goes LIVE ───────────────
                    # The member answered the awaiting slot AND, in the same
                    # utterance, said something else. Apply the resolver's
                    # templated outcome so nothing is silently dropped:
                    #   * invalidating correction (UAT-007 ZIP) → ack both, mark
                    #     dirty, route to the owner to re-resolve before delivery;
                    #   * in-scope independent → acknowledge + park for draining;
                    #   * out-of-scope / unsupported → spoken decline / redirect.
                    live = await self._apply_resolver_outcome(
                        state,
                        ctx,
                        slot_name,
                        normalized,
                        _turn_plan,
                        _turn_outcome,
                        slot_answered=True,
                        slot_label=slot_label,
                        pending_slots=pending_slots,
                    )
                    if live is not None:
                        return normalized, live

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

            # ── STALLING: caller asked for time — acknowledge ONLY ────────────
            # Pure acknowledgement ("take your time"): do NOT re-prompt the slot
            # and do NOT count a failed attempt. The slot stays pending. See
            # check_stalling() for the shared implementation (also used by
            # hand-coded confirmation flows outside _collect_slot).
            if stall_interrupt := self.check_stalling(state, messages, decision, slot_name, context=ctx):
                return None, stall_interrupt
            # Not a stall, or stalls exhausted — fall through to normal non-answer handling.

            # ── Phase 3D: resolver outcome on a NON-answered slot turn ─────────
            # The member didn't answer the slot but raised a resolver-owned side
            # request (an independent to park, or an out-of-scope/unsupported
            # question). Give it a spoken outcome, then re-ask the slot. Corrections
            # with no answer fall through to the existing CORRECTED path below.
            live = await self._apply_resolver_outcome(
                state,
                ctx,
                slot_name,
                None,
                _turn_plan,
                _turn_outcome,
                slot_answered=False,
                slot_label=slot_label,
                pending_slots=pending_slots,
            )
            if live is not None:
                return None, live

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

                    label_override, directive = self._dynamic_slot_label(state, slot_name)
                    msg = await generate_recovery_message(
                        slot_name=slot_name,
                        attempt=0,
                        guard="CLARIFY",
                        last_messages=messages[-6:],
                        slot_label_override=label_override,
                        generator_directive=directive,
                        caller_name=ctx.caller_first_name,
                        confirmed_slots=dict.fromkeys(ctx.confirmed_slots, "confirmed"),
                        user_utterance=_last_user_msg(messages),
                        grounded_values=turn_grounding_allowlist(
                            state, ctx, extracted_value=None, answered_inline=None, slot_name=slot_name
                        ),
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
        # 3. First ask (or transition). Phase 1: when UNIFIED_VOICE is on this is
        # spoken by the same generator as every recovery act; the template is the
        # guaranteed fallback so no turn is ever dropped.
        # ------------------------------------------------------------------
        first_ask_text = await self._first_ask_message(
            state, config, ctx, messages, is_transition=is_transition, template_text=_ask_first()
        )
        interrupt = self.ask_member_with_context(state, first_ask_text, ctx)
        interrupt["awaiting_slot"] = slot_name
        return None, interrupt
