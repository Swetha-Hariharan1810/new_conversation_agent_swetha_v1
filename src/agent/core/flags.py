"""
flags.py — centralized feature flags for the Context-Retention rebuild (Phase 0).

Single source of truth for the rebuild's rollout switches. Every later phase
that gates new behavior MUST read its switch from here — never re-read the raw
environment variable at the call site — so the whole rollout has one auditable
surface and the baseline dashboard can snapshot the live configuration in one
call (see ``snapshot()``).

Phase 0 contract — these are the OLD defaults, chosen so the flags reproduce
today's behavior when nothing is set:

    UNIFIED_VOICE      = false           # generation vs. template voice unified
    TURNPLAN_DECODE    = off             # off | shadow | live
    MULTI_INTENT_LIVE  = false           # resolver acts live on multi-intent turns
    STREAM_GENERATION  = false           # stream LLM-2 generation token-by-token
    PARK_ANSWERABLE    = false           # park answerable follow-ups for draining

Phase 0 only *establishes* the module (plus the baseline + characterization
tests); it deliberately does NOT wire these switches into any runtime branch, so
adding this file is a zero-output-change change. Wiring each switch onto its
behavior — and flipping a default — is the job of the later phase that owns it.

Reads are live (each getter reads ``os.environ`` on call) so tests can toggle a
flag with ``monkeypatch.setenv`` without reimporting anything. Values are parsed
leniently and clamped to a known-good default, so a malformed env var can never
crash a turn.
"""

from __future__ import annotations

import os
from typing import Final

# ── Environment variable names (the only place these strings are spelled) ───────
ENV_UNIFIED_VOICE: Final[str] = "UNIFIED_VOICE"
ENV_TURNPLAN_DECODE: Final[str] = "TURNPLAN_DECODE"
ENV_MULTI_INTENT_LIVE: Final[str] = "MULTI_INTENT_LIVE"
ENV_STREAM_GENERATION: Final[str] = "STREAM_GENERATION"
ENV_PARK_ANSWERABLE: Final[str] = "PARK_ANSWERABLE"

# ── TurnPlan decode modes ───────────────────────────────────────────────────────
TURNPLAN_OFF: Final[str] = "off"
TURNPLAN_SHADOW: Final[str] = "shadow"
TURNPLAN_LIVE: Final[str] = "live"
TURNPLAN_MODES: Final[tuple[str, ...]] = (TURNPLAN_OFF, TURNPLAN_SHADOW, TURNPLAN_LIVE)

# ── Defaults (the "OLD" baseline behavior) ──────────────────────────────────────
DEFAULT_UNIFIED_VOICE: Final[bool] = False
DEFAULT_TURNPLAN_DECODE: Final[str] = TURNPLAN_OFF
DEFAULT_MULTI_INTENT_LIVE: Final[bool] = False
DEFAULT_STREAM_GENERATION: Final[bool] = False
DEFAULT_PARK_ANSWERABLE: Final[bool] = False

_TRUE_TOKENS: Final[frozenset[str]] = frozenset({"1", "true", "t", "yes", "y", "on"})
_FALSE_TOKENS: Final[frozenset[str]] = frozenset({"0", "false", "f", "no", "n", "off", ""})


def _read_bool(name: str, default: bool) -> bool:
    """Parse an env-backed boolean leniently; unknown values fall back to default."""
    raw = os.getenv(name)
    if raw is None:
        return default
    token = raw.strip().lower()
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    return default


# ── Public getters — read these, never os.getenv, at every call site ────────────


def unified_voice() -> bool:
    """Unify the generation and template voices (Phase: response unification)."""
    return _read_bool(ENV_UNIFIED_VOICE, DEFAULT_UNIFIED_VOICE)


def turnplan_decode() -> str:
    """TurnPlan understanding-decode mode: ``off`` | ``shadow`` | ``live``.

    ``off`` runs no decode, ``shadow`` decodes and logs only, ``live`` lets the
    resolver act. Any unrecognized value clamps to the default (``off``).
    """
    raw = (os.getenv(ENV_TURNPLAN_DECODE) or DEFAULT_TURNPLAN_DECODE).strip().lower()
    return raw if raw in TURNPLAN_MODES else DEFAULT_TURNPLAN_DECODE


def multi_intent_live() -> bool:
    """Let the resolver ACT live on multi-intent slot turns."""
    return _read_bool(ENV_MULTI_INTENT_LIVE, DEFAULT_MULTI_INTENT_LIVE)


def stream_generation() -> bool:
    """Stream LLM-2 (recovery/utterance) generation token-by-token."""
    return _read_bool(ENV_STREAM_GENERATION, DEFAULT_STREAM_GENERATION)


def park_answerable() -> bool:
    """Park an answerable follow-up as a secondary intent for later draining."""
    return _read_bool(ENV_PARK_ANSWERABLE, DEFAULT_PARK_ANSWERABLE)


def snapshot() -> dict[str, object]:
    """Return the live value of every flag — one call for logs/dashboards."""
    return {
        ENV_UNIFIED_VOICE: unified_voice(),
        ENV_TURNPLAN_DECODE: turnplan_decode(),
        ENV_MULTI_INTENT_LIVE: multi_intent_live(),
        ENV_STREAM_GENERATION: stream_generation(),
        ENV_PARK_ANSWERABLE: park_answerable(),
    }
