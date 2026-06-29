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

from agent.core.dialogue_manager import DialogueManagerMixin
from agent.core.guards import ConversationGuardsMixin
from agent.core.models import SlotAttempt
from agent.core.signals import SignalsMixin
from agent.core.slot_manager import SlotManagerMixin
from agent.logger import get_logger
from agent.state import State


class BaseAgent(ConversationGuardsMixin, SlotManagerMixin, SignalsMixin, DialogueManagerMixin, ABC):
    """Abstract base for all conversational agents."""

    AGENT_NAME: str = "base_agent"
    SUPPORTED_TOPICS: Set[str] = set()

    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self._slots: Dict[str, SlotAttempt] = {}
        self._newly_confirmed: Set[str] = set()
        self._pending_ambiguous_resets: Set[str] = set()
        self._pending_intents: list[dict] = []

    @classmethod
    def from_state(cls, state: State) -> "BaseAgent":
        """Create an instance with slot state restored from LangGraph state."""
        instance = cls()
        instance._slots = {k: cls._restore_slot(k, v) for k, v in (state.get("slot_attempts") or {}).items()}
        instance._pending_ambiguous_resets = set()
        instance._pending_intents = list(state.get("pending_intents") or [])
        return instance

    async def execute(self, state: State) -> dict:
        return await self.run(state)

    @abstractmethod
    async def run(self, state: State) -> dict: ...
