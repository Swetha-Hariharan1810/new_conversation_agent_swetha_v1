"""
observability.py — Phase 2 dropped-request metric (observability only).

The proposal's "make it visible" step: establish the baseline of how often
multi-intent turns occur and how often a second request is dropped. This module
adds NO behavior change — it only:

  * detects, deterministically and PII-safely, whether a member's utterance
    carried a *secondary* request alongside the primary slot answer
    (a conjunction/clause + redirect/imperative heuristic — no LLM call), and
  * emits one structured ``dropped_request`` metric event per multi-intent turn
    recording whether that secondary request was actioned vs. parked vs. dropped,
    and increments ``State.dropped_request_count`` when it was dropped.

Wiring is a thin decorator around an agent node (``observe_dropped_requests``):
it inspects the incoming utterance and the node's returned update dict, logs the
metric, and — only when the secondary was dropped — adds the counter delta to the
returned updates. It never alters routing, messages, or any other field, so it is
safe to layer on without touching agent logic. Phase 3 uses the counter to prove
the drop rate goes to zero on UAT-007.
"""

from __future__ import annotations

import functools
import re
from typing import Awaitable, Callable

from agent.logger import get_logger
from agent.utils import _last_user_msg

logger = get_logger(__name__)

# ── Heuristic vocabulary (measurement only — never routes or responds) ─────────

# Pivot/conjunction tokens that introduce a second clause. Bare "and" is
# deliberately excluded (too noisy in spoken slot answers like dates); only the
# explicit "and also" form counts.
_PIVOTS = [
    "but",
    "also",
    "and also",
    "by the way",
    "btw",
    "as well",
    "plus",
    "in addition",
    "additionally",
    "one more thing",
    "another thing",
    "too",
]
_PIVOT_RE = re.compile(r"\b(" + "|".join(re.escape(p) for p in _PIVOTS) + r")\b", re.IGNORECASE)

# Redirect / change imperatives that signal a distinct actionable request,
# even with no pivot word (e.g. "send it to a different fax number").
_REDIRECT_RE = re.compile(
    r"(?:"
    r"\b(?:another|different|a new|new)\s+(?:fax|email|e-?mail|number|address|zip|phone)\b"
    r"|\b(?:update|change|correct|fix)\s+(?:my|the)\s+"
    r"(?:zip|fax|email|e-?mail|number|address|phone)\b"
    r"|\bsend\s+it\s+(?:somewhere\s+else|to\s+another|to\s+a\s+different)\b"
    r"|\buse\s+(?:a\s+)?(?:different|another)\b"
    r")",
    re.IGNORECASE,
)

# Map a mentioned contact/field word → the State field it concerns.
_TARGET_TOKENS: list[tuple[str, str]] = [
    (r"\bzip\b", "zip_code"),
    (r"\bfax\b", "fax"),
    (r"\be-?mail\b", "email"),
    (r"\bphone\b", "phone_number"),
]


def detect_secondary_request(utterance: str | None) -> bool:
    """True when the utterance plausibly carries a second request alongside the
    primary answer. Deterministic, PII-safe, zero model cost. Measurement only."""
    if not utterance:
        return False
    text = utterance.strip()
    if not text:
        return False
    return bool(_PIVOT_RE.search(text) or _REDIRECT_RE.search(text))


def matched_span(utterance: str | None) -> str:
    """Return a verbatim substring of the utterance that triggered secondary
    detection (redirect cue preferred, else the pivot word), or "". Guaranteed
    to be a substring of ``utterance`` — used to build TurnPlan.verbatim_span
    deterministically. PII-safe: a short cue phrase, never the whole utterance."""
    text = (utterance or "").strip()
    m = _REDIRECT_RE.search(text)
    if m:
        return m.group(0)
    m = _PIVOT_RE.search(text)
    if m:
        return m.group(1)
    return ""


def secondary_target(utterance: str | None) -> str:
    """Best-effort guess of which field the secondary request concerns (or "")."""
    text = (utterance or "").lower()
    for pattern, field in _TARGET_TOKENS:
        if re.search(pattern, text):
            return field
    return ""


def secondary_request_shape(utterance: str | None) -> dict:
    """A PII-safe descriptor of the utterance shape (no raw text)."""
    text = (utterance or "").strip()
    pivot_match = _PIVOT_RE.search(text)
    return {
        "pivot": pivot_match.group(1).lower() if pivot_match else "",
        "redirect": bool(_REDIRECT_RE.search(text)),
        "target": secondary_target(text),
        "n_tokens": len(text.split()),
        "has_question": "?" in text,
    }


def classify_secondary_outcome(state: dict, updates: dict, *, target: str) -> str:
    """Classify what happened to a detected secondary request this turn.

    Pure function over the pre-turn state and the node's update dict:
      * "parked"   — the turn newly marked a dependent artifact stale (deferring
                     a dependent action until the value is re-resolved).
      * "actioned" — an owner field the secondary referenced was changed to a new
                     value this turn (the request was handled).
      * "dropped"  — neither; the secondary request produced no effect.

    Today (Phase 2) the multi-intent turns in scope resolve to "dropped"; the
    other branches are real and will light up as later phases handle/park
    secondaries.
    """
    prev_dirty = state.get("dirty_artifacts") or {}
    new_dirty = updates.get("dirty_artifacts") or {}
    newly_parked = [k for k, v in new_dirty.items() if v and not prev_dirty.get(k)]
    if newly_parked:
        return "parked"

    # A secondary enqueued for later (resolver multi_intent_ack) is parked, not dropped.
    new_queue = updates.get("intent_queue")
    if new_queue is not None and len(new_queue) > len(state.get("intent_queue") or []):
        return "parked"

    if target and target in updates:
        new_val = (updates.get(target) or "")
        old_val = (state.get(target) or "")
        if new_val and new_val != old_val:
            return "actioned"

    return "dropped"


def observe_turn(state: dict, updates: dict, *, agent_name: str) -> dict:
    """Emit the dropped_request metric for one turn and return updates, augmented
    with the counter delta ONLY when a secondary request was dropped. No other
    field is touched (observability only)."""
    utterance = _last_user_msg(state.get("messages") or [])
    if not detect_secondary_request(utterance):
        return updates

    target = secondary_target(utterance)
    outcome = classify_secondary_outcome(state, updates, target=target)
    shape = secondary_request_shape(utterance)

    logger.info(
        "multi_intent_turn",
        extra={
            "metric": "dropped_request",
            "agent": agent_name,
            "awaiting_slot": state.get("awaiting_slot", ""),
            "secondary_request": True,
            "outcome": outcome,
            **shape,
        },
    )

    if outcome == "dropped":
        prev = state.get("dropped_request_count") or 0
        return {**updates, "dropped_request_count": prev + 1}
    return updates


def observe_dropped_requests(
    node: Callable[[dict], Awaitable[dict]],
) -> Callable[[dict], Awaitable[dict]]:
    """Decorator for an agent node callable. Runs the node unchanged, then emits
    the dropped-request metric and (only on a dropped secondary) adds the counter
    delta to the returned updates. Failures here never affect the node's result."""
    agent_name = getattr(node, "__name__", "agent")

    @functools.wraps(node)
    async def wrapper(state: dict) -> dict:
        updates = await node(state)
        try:
            return observe_turn(state, updates, agent_name=agent_name)
        except Exception:  # observability must never break a turn
            logger.exception("observe_dropped_requests: metric emission failed")
            return updates

    return wrapper
