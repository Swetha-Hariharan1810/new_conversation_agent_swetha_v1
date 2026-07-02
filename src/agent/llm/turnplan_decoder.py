"""
turnplan_decoder.py — the real LLM turn-understanding decode (Phase 2, shadow).

Today understanding is split between GPT's coarse ``event_type`` (WorkerResult)
and the narrow regex ``heuristic_decoder``, so most bundled follow-ups are never
even seen. This installs a real LLM decoder that reads one utterance and emits a
full ``TurnPlan`` (slot answer + secondary intents with verbatim spans +
correction + guard + confidence), and — for an in-scope side question that the
session snapshot can answer — a grounded ``answer`` produced in the SAME decode
call (no extra round-trip). It runs in SHADOW: log only, act never (installed as
a log-only observer via ``shadow.set_turnplan_observer``).

Latency contract:
  * Fast-path — if no secondary cue is present (``detect_secondary_request`` is
    False) AND ``event_type`` is plain ``answered``, skip the LLM decode entirely
    and return the deterministic ``heuristic_decoder`` result. Single-intent turns
    pay nothing.
  * The richer LLM decode runs only when a secondary is plausibly present, and it
    uses the understanding tier (``get_understanding_llm``) — the decode replaces
    an agent's extraction call rather than stacking a second heavy model.
  * Fallback chain is try → LLM → heuristic → None: on any decode failure/timeout
    the deterministic heuristic decoder is used, so the observer never breaks.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional

from agent.llm.extractor import build_worker_input
from agent.llm.schema import EventType, TurnPlan
from agent.llm.snapshot import build_session_snapshot
from agent.logger import get_logger
from agent.utils import _last_assistant_msg, read_prompt

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def build_turnplan_prompt() -> str:
    """System prompt for the TurnPlan decode: the shared grounding rules plus the
    TurnPlan emission contract (reuses ``extraction/header_extraction.md``)."""
    grounding = read_prompt("extraction/header_extraction.md")
    turnplan = read_prompt("extraction/turnplan.md")
    return f"{turnplan}\n\n---\n\n{grounding}"


def _fast_path_single_intent(utterance: str, decision: Any) -> bool:
    """True when this turn is a plain single-intent answer with no secondary cue —
    the cheap path that needs no LLM TurnPlan decode."""
    from agent.orchestration.observability import detect_secondary_request

    if detect_secondary_request(utterance):
        return False
    event_type = getattr(decision, "event_type", None)
    # No decision, or a plain "answered", is single-intent. Anything richer
    # (answered_with_followup / corrected / ambiguous) earns the full decode.
    return event_type is None or event_type == EventType.ANSWERED


async def _llm_decode(state: dict, utterance: str, decision: Any) -> Optional[TurnPlan]:
    """One structured-output call → a TurnPlan, with the session snapshot injected
    so an in-scope side question can be answered from what is already known."""
    from agent.llm.config import get_understanding_llm

    awaiting = state.get("awaiting_slot") or ""
    messages = build_worker_input(
        build_turnplan_prompt(),
        awaiting_slot=awaiting,
        last_agent_message=_last_assistant_msg(state.get("messages") or []),
        last_user_message=utterance,
        recent_messages=(state.get("messages") or [])[-6:],
    )

    # Inject the snapshot into the user block exactly as follow_up does, so the
    # decode can answer a relevant side question in the same forward pass.
    snapshot = build_session_snapshot(state)
    if snapshot and messages:
        messages[-1]["content"] = f"SESSION SNAPSHOT:\n{snapshot}\n\n" + messages[-1]["content"]

    llm = get_understanding_llm()
    plan: TurnPlan = await llm.with_structured_output(TurnPlan).ainvoke(messages)
    return plan


async def llm_turnplan_decoder(state: dict, utterance: str, decision: Any = None) -> Optional[TurnPlan]:
    """Async TurnPlan decoder installed as the shadow observer (Phase 2).

    Fast-paths single-intent turns to the deterministic heuristic (zero LLM cost),
    otherwise runs the LLM decode, falling back to the heuristic on any failure so
    a plan is always available for the shadow log (try → LLM → heuristic → None).
    """
    # Deferred import avoids a cycle: shadow imports resolver/observability, and
    # this module is what shadow later calls — not the reverse at import time.
    from agent.orchestration.shadow import heuristic_decoder

    if _fast_path_single_intent(utterance, decision):
        return heuristic_decoder(state, utterance, decision)

    try:
        plan = await _llm_decode(state, utterance, decision)
        if plan is not None:
            return plan
        logger.warning("llm_turnplan_decoder: empty decode — falling back to heuristic")
    except Exception:
        logger.warning("llm_turnplan_decoder: decode failed — falling back to heuristic", exc_info=True)

    return heuristic_decoder(state, utterance, decision)


def configure_turnplan_decoder() -> None:
    """Install the TurnPlan decoder per ``TURNPLAN_DECODE`` (called at app startup).

      * ``shadow`` — install as a LOG-ONLY observer; the live path is unchanged.
      * ``live``   — install as the acting shadow decoder (a later phase acts on it).
      * ``off``    — clear the observer; leave the existing (deterministic) decoder.
    """
    from agent.core import flags
    from agent.orchestration import shadow

    mode = flags.turnplan_decode()
    if mode == flags.TURNPLAN_SHADOW:
        shadow.set_turnplan_observer(llm_turnplan_decoder)
        logger.info("configure_turnplan_decoder: LLM TurnPlan decode installed in SHADOW (log-only)")
    elif mode == flags.TURNPLAN_LIVE:
        shadow.set_shadow_decoder(llm_turnplan_decoder)
        logger.info("configure_turnplan_decoder: LLM TurnPlan decode installed LIVE")
    else:
        shadow.clear_turnplan_observer()
