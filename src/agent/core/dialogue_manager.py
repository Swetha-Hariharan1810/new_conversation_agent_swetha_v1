"""
dialogue_manager.py — DialogueManagerMixin: capture every request in a turn.

Mixed into BaseAgent. capture_and_triage reads the extraction result, finds any
correction of a committed value and any additional intents, classifies each one,
and records them on self._pending_intents. The signal methods persist that list
into LangGraph state, so the orchestrator can drain and rewind on later turns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent.core.pending_intents import (
    IntentKind,
    IntentStatus,
    PendingIntent,
    add_intent,
)
from agent.core.triage import classify_intent
from agent.logger import get_logger

if TYPE_CHECKING:
    from agent.llm.schema import WorkerResult
    from agent.state import State

_logger = get_logger(__name__)

# Non-actionable intents that must be ACKNOWLEDGED OUT LOUD once, without
# derailing the in-progress primary flow.
_SIDE_REQUEST_KINDS = {IntentKind.UNSUPPORTED.value, IntentKind.OFF_TOPIC.value}

# Spoken acknowledgement lines. Short and additive — they prepend to the
# primary flow's next prompt so the member hears their side request was heard.
_ACK_UNSUPPORTED = (
    "By the way, I'm not able to help with that here, but I can point you to the "
    "right team for it before we wrap up."
)
_ACK_OFF_TOPIC = "Also, I'll have to stay focused on your current request for now."


class DialogueManagerMixin:
    """Adds multi intent capture and triage to BaseAgent."""

    def pending_intents_list(self) -> list[dict]:
        return list(getattr(self, "_pending_intents", []) or [])

    def capture_and_triage(self, state: "State", result: "WorkerResult | None") -> None:
        if result is None:
            return

        intents = self.pending_intents_list() or list(state.get("pending_intents") or [])
        guard = getattr(result, "guard", "NONE")
        guard = guard.value if hasattr(guard, "value") else str(guard)

        # 1. A correction of a value the member already committed.
        target = getattr(result, "correction_target", None)
        if target:
            kind = classify_intent(guard="NONE", correction_target=target, intent_label=None, topic=None)
            intents = add_intent(
                intents, PendingIntent(kind=kind.value, raw_text=f"change {target}", target=target)
            )

        # 2. Any extra intents raised in the same turn.
        for label in getattr(result, "secondary_intents", None) or []:
            label = (label or "").strip()
            if not label:
                continue
            kind = classify_intent(guard=guard, correction_target=None, intent_label=label, topic=label)
            intents = add_intent(intents, PendingIntent(kind=kind.value, raw_text=label, target=None))

        self._pending_intents = intents
        if intents:
            _logger.info("dialogue_manager: captured intents", extra={"count": len(intents)})

    def side_request_ack(self) -> str:
        """Return a one-line acknowledgement for non-actionable side requests.

        A captured unsupported (in-domain pharmacy/billing/preauth) or off_topic
        intent must be spoken out loud once — the "never silently drop" promise —
        but it must NOT block or rewind the in-progress primary slot flow. This
        returns a short sentence the agent prepends to its next spoken message and
        flips those intents from OPEN to ACKNOWLEDGED so the line is said exactly
        once. Returns "" when there is nothing fresh to acknowledge (idempotent).

        Only the spoken status changes; kind stays UNSUPPORTED/OFF_TOPIC, so
        all_in_scope_resolved and closure safeguards are unaffected (these kinds
        never block closure, but are voiced before it via this helper).
        """
        intents = self.pending_intents_list()
        fresh = [
            d
            for d in intents
            if d.get("kind") in _SIDE_REQUEST_KINDS and d.get("status") == IntentStatus.OPEN.value
        ]
        if not fresh:
            return ""

        out = []
        for d in intents:
            if d.get("kind") in _SIDE_REQUEST_KINDS and d.get("status") == IntentStatus.OPEN.value:
                d = {**d, "status": IntentStatus.ACKNOWLEDGED.value}
            out.append(d)
        self._pending_intents = out

        parts = []
        if any(d.get("kind") == IntentKind.UNSUPPORTED.value for d in fresh):
            parts.append(_ACK_UNSUPPORTED)
        if any(d.get("kind") == IntentKind.OFF_TOPIC.value for d in fresh):
            parts.append(_ACK_OFF_TOPIC)
        _logger.info("dialogue_manager: acknowledged side requests", extra={"count": len(fresh)})
        return " ".join(parts)
