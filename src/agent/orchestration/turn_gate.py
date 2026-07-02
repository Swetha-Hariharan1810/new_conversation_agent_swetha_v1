"""
turn_gate.py — ONE understanding decode per turn, at one chokepoint (Phase 2).

``understand_turn`` is the single async entrypoint every per-turn caller uses —
the shared slot collector, the conversation guards, and the hand-coded yes/no
confirmation branches. It guarantees at most one LLM decode per user turn:

  * Fast path — a plain single-intent answer with no secondary cue
    (``_fast_path_single_intent``) is decoded by the deterministic
    ``heuristic_decoder`` at zero LLM cost.
  * Budgeted decode — everything else runs ``decode_and_resolve_async`` under
    ``asyncio.wait_for`` with the ``TURNPLAN_TIMEOUT_MS`` budget; on timeout or
    any exception the gate falls back to the heuristic decoder
    (``metric="turnplan_timeout"``), so a turn is never lost to a slow decode.
  * Idempotence — the result is stamped into the live state dict
    (``_turn_gate_msg_id`` + ``_turn_understanding``); a second call for the
    same user message returns the cached plan without re-decoding. If the
    second caller focuses a different ``awaiting_slot``, only the cheap pure
    resolver re-runs against the cached plan — never the decode.

The gate respects the shadow-decoder kill switch: when no decoder is installed
(``shadow.get_shadow_decoder() is None``) it returns ``(None, None)`` like the
paths it replaces.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Any, Optional

from agent.core import flags
from agent.llm.schema import TurnPlan
from agent.logger import get_logger
from agent.orchestration.resolver import ResolverOutcome
from agent.orchestration.shadow import (
    _resolve_and_log,
    decode_and_resolve,
    decode_and_resolve_async,
    get_shadow_decoder,
    heuristic_decoder,
)

logger = get_logger(__name__)

# Private per-turn state keys (never persisted contract; live only within the turn).
_MSG_ID_KEY = "_turn_gate_msg_id"
_UNDERSTANDING_KEY = "_turn_understanding"


def _msg_id(utterance: str) -> str:
    """Stable id for the last user message (content hash, not PII in logs)."""
    return hashlib.sha1((utterance or "").encode("utf-8")).hexdigest()[:16]


def stashed_turn_understanding(state: dict) -> tuple[Optional[TurnPlan], Optional[ResolverOutcome]]:
    """The (plan, outcome) the gate stashed for this turn, or (None, None)."""
    cached = state.get(_UNDERSTANDING_KEY) or {}
    return cached.get("plan"), cached.get("outcome")


def _log_latency(start: float, *, fast_path: bool, agent_name: str, awaiting_slot: str) -> None:
    """Phase 4 latency proof: one record per decoded turn. Fast-path turns must
    add <1ms; decode turns stay within one LLM round-trip (the idempotence
    cache guarantees a turn never pays a second understanding call)."""
    latency_ms = (time.perf_counter() - start) * 1000.0
    logger.info(
        "turn_gate_latency",
        extra={
            "metric": "turn_gate_latency_ms",
            "latency_ms": round(latency_ms, 3),
            "fast_path": fast_path,
            "agent": agent_name,
            "awaiting_slot": awaiting_slot,
        },
    )


def _stash(
    state: dict,
    msg_id: str,
    awaiting_slot: str,
    plan: Optional[TurnPlan],
    outcome: Optional[ResolverOutcome],
) -> None:
    try:
        state[_MSG_ID_KEY] = msg_id
        state[_UNDERSTANDING_KEY] = {
            "plan": plan,
            "outcome": outcome,
            "awaiting_slot": awaiting_slot,
        }
    except Exception:  # a read-only state must never break the turn
        logger.debug("turn_gate: could not stash understanding in state", exc_info=True)


async def understand_turn(
    state: dict,
    *,
    utterance: str,
    awaiting_slot: str,
    decision: Any = None,
    agent_name: str = "",
) -> tuple[Optional[TurnPlan], Optional[ResolverOutcome]]:
    """Decode + resolve this user turn exactly once. See module docstring.

    Same contract as ``decode_and_resolve_async``: returns ``(plan, outcome)``,
    or ``(None, None)`` when no decoder is installed or there is nothing to plan.
    """
    if get_shadow_decoder() is None:
        return (None, None)

    from agent.llm.turnplan_decoder import _fast_path_single_intent

    start = time.perf_counter()
    msg_id = _msg_id(utterance)

    # ── Idempotence: already decoded this user message this turn ───────────────
    if state.get(_MSG_ID_KEY) == msg_id and state.get(_UNDERSTANDING_KEY):
        cached = state[_UNDERSTANDING_KEY]
        plan = cached.get("plan")
        if cached.get("awaiting_slot") == awaiting_slot:
            return plan, cached.get("outcome")
        # Same utterance, different slot focus: re-run only the pure resolver on
        # the cached plan (no second decode).
        outcome = (
            _resolve_and_log(
                plan, state, utterance=utterance, awaiting_slot=awaiting_slot, agent_name=agent_name
            )
            if plan is not None
            else None
        )
        _stash(state, msg_id, awaiting_slot, plan, outcome)
        return plan, outcome

    # ── Fast path: plain single-intent answer — zero LLM cost ─────────────────
    # decode_and_resolve (the sync path) runs the installed decoder when it is
    # synchronous (deterministic, free) and substitutes the heuristic when the
    # installed decoder is the async LLM decode — either way, no LLM call.
    if _fast_path_single_intent(utterance, decision):
        plan, outcome = decode_and_resolve(
            state,
            utterance=utterance,
            awaiting_slot=awaiting_slot,
            decision=decision,
            agent_name=agent_name,
        )
        _stash(state, msg_id, awaiting_slot, plan, outcome)
        _log_latency(start, fast_path=True, agent_name=agent_name, awaiting_slot=awaiting_slot)
        return plan, outcome

    # ── Budgeted full decode; heuristic fallback on timeout/failure ───────────
    try:
        plan, outcome = await asyncio.wait_for(
            decode_and_resolve_async(
                state,
                utterance=utterance,
                awaiting_slot=awaiting_slot,
                decision=decision,
                agent_name=agent_name,
            ),
            timeout=flags.turnplan_timeout_ms() / 1000.0,
        )
    except Exception:
        logger.warning(
            "understand_turn: decode timed out/failed — using heuristic decoder",
            extra={"metric": "turnplan_timeout", "agent": agent_name, "awaiting_slot": awaiting_slot},
        )
        plan = heuristic_decoder(state, utterance, decision)
        outcome = (
            _resolve_and_log(
                plan, state, utterance=utterance, awaiting_slot=awaiting_slot, agent_name=agent_name
            )
            if plan is not None
            else None
        )

    _stash(state, msg_id, awaiting_slot, plan, outcome)
    _log_latency(start, fast_path=False, agent_name=agent_name, awaiting_slot=awaiting_slot)
    return plan, outcome
