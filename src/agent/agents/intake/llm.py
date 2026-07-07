"""
LLM-based intake intent extraction.

Responsibilities:
- structured output extraction
- extraction fallback handling

IMPORTANT:
This module owns ALL intake LLM logic.

The IntakeAgent should NOT:
- build prompts
- manage structured outputs
- perform LLM orchestration
"""

from __future__ import annotations

from agent.core.request_detection import reconcile_worker_result
from agent.llm.extractor import build_worker_input
from agent.llm.schema import WorkerResult
from agent.logger import get_logger

logger = get_logger(__name__)


async def extract_intake_intent(
    llm,
    system_prompt: str,
    last_agent_message: str,
    last_user_message: str,
    pending_slots: list[str] | None = None,
    attempt: int = 0,
    recent_messages: list | None = None,
) -> WorkerResult:
    """
    Run one LLM call to extract caller intent for the Intake agent.

    Falls back to WorkerResult() on any exception so the caller can use
    keyword-matching fallback logic without crashing.
    """
    messages = build_worker_input(
        system_prompt,
        awaiting_slot="intent",
        last_agent_message=last_agent_message,
        last_user_message=last_user_message,
        pending_slots=pending_slots,
        attempt=0,
        recent_messages=recent_messages,
    )
    try:
        result: WorkerResult = await llm.with_structured_output(WorkerResult).ainvoke(messages)
        # Regex fallback + veto layer (request_detection): fills a missed
        # update_target/request_kind and clears WAIT on correction turns.
        return reconcile_worker_result(result, last_user_message)
    except Exception as exc:
        logger.exception("Intent extraction failed", exc_info=exc)
        return WorkerResult()
