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

    async def _handle_caller_type_confirm(self, state: State) -> dict:
        """
        Handles yes/no response to the caller_type transfer offer.

        Pattern (same as all yes/no decisions in this codebase):
          1. _quick_yes_no() keyword fast-path — no LLM if clear
          2. LLM extraction only if keyword returns ambiguous
          3. normalize_yes_no() on extracted value
          4. Branch on yes / no / exhausted
        """
        from agent.llm.config import get_extraction_llm
        from agent.llm.extractor import build_worker_input
        from agent.llm.schema import WorkerResult
        from agent.slots.normalizers import normalize_yes_no
        from agent.utils import _last_assistant_msg, _last_user_msg, _quick_yes_no

        messages = list(state.get("messages") or [])
        last_user = _last_user_msg(messages)
        last_agent = _last_assistant_msg(messages)

        # ── Step 1: keyword fast-path ──────────────────────────────────
        response = _quick_yes_no(last_user)

        # ── Step 2: LLM only if keyword gave nothing ───────────────────
        if not response:
            prompt = (
                "Extract whether the caller wants to be transferred "
                "or wants to continue on this line.\n\n"
                "FIELDS\n"
                "transfer_response | 'yes' or 'no'\n\n"
                "yes — caller wants to be transferred\n"
                "no  — caller wants to continue on this line\n\n"
                "Use full semantic understanding. Do not guess."
            )
            msgs = build_worker_input(
                prompt,
                awaiting_slot="transfer_response",
                last_agent_message=last_agent,
                last_user_message=last_user,
                recent_messages=messages[-4:],
            )
            try:
                llm_result: WorkerResult = await (
                    get_extraction_llm().with_structured_output(WorkerResult).ainvoke(msgs)
                )
                raw = (llm_result.extracted or {}).get("transfer_response", "")
                response = normalize_yes_no(raw) if raw else ""
            except Exception:
                response = ""

        # ── Step 3: branch ─────────────────────────────────────────────
        if response == "yes":
            return self.signal_escalate(
                state,
                "Of course — connecting you now. Please hold.",
                reason=f"non_member_transfer:{state.get('caller_type', '')}",
                initiator="Caller",
            )

        if response == "no":
            prior_slot = state.get("slot_before_caller_type_check", "")
            result = self.ask_member(state, "Of course — happy to help. Let's continue.")
            result["awaiting_slot"] = prior_slot
            result["slot_before_caller_type_check"] = ""
            return result

        # ── Step 4: ambiguous — re-ask once then escalate ─────────────
        attempts = (state.get("slot_attempts") or {}).get("transfer_response", {}).get("attempt_count", 0)

        if attempts >= 1:
            return self.signal_escalate(
                state,
                "Let me connect you with someone who can help. Please hold.",
                reason="caller_type_confirm_exhausted",
                initiator="Agent",
            )

        result = self.ask_member(
            state, "Sorry — would you like me to transfer you or would you prefer to continue?"
        )
        result["awaiting_slot"] = "caller_type_transfer_confirm"
        result["slot_attempts"] = {
            **(state.get("slot_attempts") or {}),
            "transfer_response": {
                "attempt_count": attempts + 1,
                "confirmed": False,
                "last_value": None,
            },
        }
        return result

    async def execute(self, state: State) -> dict:
        if state.get("awaiting_slot") == "caller_type_transfer_confirm":
            return await self._handle_caller_type_confirm(state)
        return await self.run(state)

    @abstractmethod
    async def run(self, state: State) -> dict: ...
