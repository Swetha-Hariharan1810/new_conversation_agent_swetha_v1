"""
agent.py — BaseAgent: abstract conversational agent.

Composes three mixins into a single base class all agents inherit from:
  ConversationGuardsMixin  (guards.py)    — abuse, ASR, transfer detection
  SlotManagerMixin         (slot_manager.py) — slot state + _collect_slot
  SignalsMixin             (signals.py)   — ask_member, signal_complete/escalate

To build a new agent:
  1. Inherit from BaseAgent
  2. Set AGENT_NAME = "my_agent_name"
  3. Implement async run(self, state) -> dict
  4. Use self._collect_slot(), self.signal_complete(), etc. — they're all inherited

No agent-specific logic lives here. This class is infrastructure only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Set

from agent.core.guards import ConversationGuardsMixin
from agent.core.models import SlotAttempt
from agent.core.signals import SignalsMixin
from agent.core.slot_manager import SlotManagerMixin
from agent.logger import get_logger
from agent.state import State


class BaseAgent(ConversationGuardsMixin, SlotManagerMixin, SignalsMixin, ABC):
    """Abstract base for all conversational agents."""

    AGENT_NAME: str = "base_agent"
    SUPPORTED_TOPICS: Set[str] = set()

    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self._slots: Dict[str, SlotAttempt] = {}
        self._newly_confirmed: Set[str] = set()
        self._pending_ambiguous_resets: Set[str] = set()

    @classmethod
    def from_state(cls, state: State) -> "BaseAgent":
        """Create an instance with slot state restored from LangGraph state."""
        instance = cls()
        instance._slots = {k: cls._restore_slot(k, v) for k, v in (state.get("slot_attempts") or {}).items()}
        instance._pending_ambiguous_resets = set()
        return instance

    async def execute(self, state: State) -> dict:
        result = await self.run(state)
        return self._bridge_drained_intent(state, result)

    def _bridge_drained_intent(self, state: State, result: dict) -> dict:
        """Phase 3: when this agent was routed here by DRAINING a parked side
        request, its first spoken message opens with a one-clause bridge that
        acknowledges the request in the caller's own words ("Now, about the
        other thing you mentioned — a refund on my last bill."). The span was
        stored verbatim at parking time, so the bridge is grounded by
        construction. Spoken only when UNIFIED_VOICE is on; the reason is
        consumed either way so it can never bridge a later, unrelated turn.
        """
        if not state.get("drained_intent_reason"):
            return result
        try:
            from agent.core import flags
            from agent.responses.turn_acts import render_drain_bridge

            message = result.get("messages")
            if flags.unified_voice() and isinstance(message, dict) and (message.get("content") or "").strip():
                bridge = render_drain_bridge(span=state.get("drained_intent_reason"))
                result = {
                    **result,
                    "messages": {**message, "content": f"{bridge} {message['content']}"},
                }
        except Exception:  # a bridge must never break the drained turn
            self.logger.debug("_bridge_drained_intent failed", exc_info=True)
        result["drained_intent_reason"] = ""  # consumed — speak the bridge once
        return result

    @abstractmethod
    async def run(self, state: State) -> dict: ...
