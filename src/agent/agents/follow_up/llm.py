"""
llm.py — Single LLM call for FollowUpAgent.

One call does everything: guard classification, intent routing,
and answer generation from session context.

Previously two calls (extract_follow_up_decision + generate_follow_up_answer)
are now collapsed into one. The session snapshot is injected into the user
content block so the model classifies intent AND writes the answer in a
single forward pass. The answer field on WorkerResult carries it back.

Spoken-form requirement: every email address and website written into the
SESSION SNAPSHOT is rendered in fully spoken words ("at" / "dot") via
speak_email/speak_url, so any answer the LLM generates from the snapshot
speaks them verbatim.
"""

from __future__ import annotations

from agent.llm.extractor import build_worker_input
from agent.llm.schema import FollowUpResult
from agent.llm.snapshot import build_session_snapshot
from agent.logger import get_logger
from agent.state import State

logger = get_logger(__name__)

# Kept as a module-local alias so existing references resolve; the builder now
# lives in agent.llm.snapshot (generalized for the TurnPlan decode, Phase 2).
_build_session_snapshot = build_session_snapshot


async def extract_follow_up_decision(
    llm,
    system_prompt: str,
    *,
    last_agent_message: str,
    last_user_message: str,
    recent_messages: list | None = None,
    state: State | None = None,
) -> FollowUpResult:
    """
    Single LLM call: classifies guard + intent AND generates answer if needed.

    Session snapshot is injected into the user content so the model can answer
    questions directly without a second generation call.

    The answer field on WorkerResult carries the generated response back.
    Falls back to an empty WorkerResult on any exception.
    """
    session_snapshot = _build_session_snapshot(state) if state else ""

    messages = build_worker_input(
        system_prompt,
        awaiting_slot="",
        last_agent_message=last_agent_message,
        last_user_message=last_user_message,
        recent_messages=recent_messages,
    )

    # Inject session snapshot into the user message content so the model
    # has all the information it needs to generate an answer in one pass.
    if session_snapshot and messages:
        messages[-1]["content"] = f"SESSION SNAPSHOT:\n{session_snapshot}\n\n" + messages[-1]["content"]

    try:
        result: FollowUpResult = await llm.with_structured_output(FollowUpResult).ainvoke(messages)
        return result
    except Exception:
        logger.exception("extract_follow_up_decision: LLM call failed")
        return FollowUpResult()
