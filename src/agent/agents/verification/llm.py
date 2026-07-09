"""
verification_llm.py — LLM extraction for identity verification.

Public API:
    extract_verification_decision(llm, system_prompt, awaiting_slot,
                                  last_agent_message, last_user_message,
                                  confirmed_slots, attempt, recent_messages)
        → WorkerResult

Extracted slot values live in result.extracted:
  first_name, last_name, member_id, dob, relationship, phone_confirmation

Corrections live in result.corrections (dict[str, str]).
"""

from __future__ import annotations

from agent.core.request_detection import reconcile_worker_result
from agent.llm.extractor import build_worker_input
from agent.llm.schema import WorkerResult
from agent.logger import get_logger

logger = get_logger(__name__)


async def extract_verification_decision(
    llm,
    system_prompt: str,
    awaiting_slot: str,
    last_agent_message: str,
    last_user_message: str,
    *,
    confirmed_slots: dict | None = None,
    pending_slots: list[str] | None = None,
    attempt: int = 0,
    recent_messages: list | None = None,
) -> WorkerResult:
    """
    Run one LLM call to extract identity slots from the latest user utterance.

    confirmed_slots: already-confirmed slot values to include as context so
        the LLM can classify corrections for slots it has seen before.
    pending_slots: slots still to be collected later this call, so the LLM
        can classify parkable follow-up questions.
    attempt: how many collection attempts have been made for awaiting_slot.
    recent_messages: recent conversation turns (dicts with "role"/"content")
        passed through to build_worker_input for history context.

    Falls back to an empty WorkerResult on any exception — the slot
    collection loop handles the missing values gracefully.
    """
    messages = build_worker_input(
        system_prompt,
        awaiting_slot=awaiting_slot,
        last_agent_message=last_agent_message,
        last_user_message=last_user_message,
        confirmed_slots=confirmed_slots,
        pending_slots=pending_slots,
        attempt=attempt,
        recent_messages=recent_messages,
    )
    try:
        result: WorkerResult = await llm.with_structured_output(WorkerResult).ainvoke(messages)
        # Regex fallback + veto layer (request_detection): fills a missed
        # update_target/request_kind and clears WAIT on correction turns.
        result = reconcile_worker_result(result, last_user_message)
        return result
    except Exception:
        logger.exception("extract_verification_decision: LLM extraction failed")
        return WorkerResult()


async def extract_name_confirmation(
    llm,
    system_prompt: str,
    *,
    last_agent_message: str,
    last_user_message: str,
    pending_slots: list[str] | None = None,
    attempt: int = 0,
    recent_messages: list | None = None,
) -> WorkerResult:
    """
    Run one LLM call to extract the member's response to the name readback.

    Uses name_confirmation.md which handles three outcomes:
      - name_confirmed="yes"               → member confirmed the spelled name
      - first_name / last_name extracted   → inline correction provided
      - name_confirmed="no", no names      → bare no, correction needed separately

    Falls back to an empty WorkerResult on any exception.
    """
    messages = build_worker_input(
        system_prompt,
        awaiting_slot="name_confirmed",
        last_agent_message=last_agent_message,
        last_user_message=last_user_message,
        confirmed_slots=None,
        pending_slots=pending_slots,
        attempt=attempt,
        recent_messages=recent_messages,
    )
    try:
        result: WorkerResult = await llm.with_structured_output(WorkerResult).ainvoke(messages)
        # Regex fallback + veto layer (request_detection): fills a missed
        # update_target/request_kind and clears WAIT on correction turns.
        result = reconcile_worker_result(result, last_user_message)
        return result
    except Exception:
        logger.exception("extract_name_confirmation: LLM extraction failed")
        return WorkerResult()
