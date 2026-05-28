"""
constants.py — Shared detection patterns and limits for all agents.

Single source of truth for:
  - Abuse / toxicity patterns (ABUSE_PATTERNS)
  - Self-harm detection patterns (SELF_HARM_PATTERNS)
  - Escalation and interruption keywords
  - Retry and execution limits
  - Conversational semantic patterns

Previously duplicated across agent-specific and component modules.
Those files have been deleted; this is the authoritative source.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Retry / execution limits
# ---------------------------------------------------------------------------
MAX_SLOT_ATTEMPTS = 2
MAX_TOOL_ITERATIONS = 5
MAX_ROUTER_LOOPS = 25

HISTORY_WINDOW_SIZE = 6

# ---------------------------------------------------------------------------
# Abuse / toxicity detection — compiled once at import time
# ---------------------------------------------------------------------------
ABUSE_PATTERNS: list = [
    re.compile(p, re.IGNORECASE)
    for p in [
        # Keyword fallback only — primary abuse detection is LLM-based (guard == "ABUSE").
        # These patterns catch unambiguous explicit abuse the LLM might miss under low confidence.
        r"\bidiot\b",
        r"\bshut up\b",
        r"\bfuck\b",
        r"\basshole\b",
        r"\bbitch\b",
        r"\bcunt\b",
        r"\bmoron\b",
        r"\bnigger\b",
        r"\bslut\b",
        r"\bdick\b",
        r"\bpussy\b",
    ]
]

# ---------------------------------------------------------------------------
# Self-harm detection — compiled once at import time
# Keep separate from ABUSE_PATTERNS for independent logging and future routing
# ---------------------------------------------------------------------------
SELF_HARM_PATTERNS: list = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bkill myself\b",
        r"\bkilling myself\b",
        r"\bend my life\b",
        r"\bending my life\b",
        r"\bwant to die\b",
        r"\bwanting to die\b",
        r"\bsuicid",  # catches suicide, suicidal, suiciding
        r"\bhurt myself\b",
        r"\bharming myself\b",
        r"\bself.harm\b",
        r"\bno reason to live\b",
        r"\bnot worth living\b",
        r"\bcan't go on\b",
        r"\bcant go on\b",
    ]
]

# ---------------------------------------------------------------------------
# Timeout defaults
# ---------------------------------------------------------------------------
DEFAULT_LLM_TIMEOUT_SECONDS = 30
DEFAULT_TOOL_TIMEOUT_SECONDS = 15

# ---------------------------------------------------------------------------
# Conversational semantic patterns
# ---------------------------------------------------------------------------

# Regex fallback — used by SemanticSignalDetector when LLM guard confidence is below threshold.
HESITATION_PATTERNS = [
    r"\buh\b",
    r"\bum\b",
    r"\bhmm\b",
    r"\bah\b",
    r"\ber\b",
    r"\bone second\b",
    r"\bhold on\b",
    r"\bwait\b",
]

# Regex fallback — used by SemanticSignalDetector when LLM guard confidence is below threshold.
UNCERTAIN_PATTERNS = [
    r"\bi think\b",
    r"\bprobably\b",
    r"\bmaybe\b",
    r"\bnot sure\b",
    r"\bi guess\b",
]

# Regex fallback — used by SemanticSignalDetector when LLM guard confidence is below threshold.
CONFUSED_PATTERNS = [
    r"\bwhat do you mean\b",
    r"\bcan you repeat\b",
    r"\bi dont understand\b",
    r"\bi didn't understand\b",
    r"\bcould you explain\b",
]

# Regex fallback — used by SemanticSignalDetector when LLM guard confidence is below threshold.
FRUSTRATED_PATTERNS = [
    r"\bthis is ridiculous\b",
    r"\bive already said\b",
    r"\bthis is taking too long\b",
    r"\bwhy do you need that\b",
]

# Regex fallback — used by SemanticSignalDetector when LLM guard confidence is below threshold.
SPELLING_PATTERNS = [
    r"\bas in\b",
    r"\bfor\b",
]

# Keyword fallback for INTERRUPTION guard — used by guards.py when LLM confidence < 0.7.
# Only unambiguous phrases are included; broader interruption detection is LLM-based.
INTERRUPTION_PATTERNS: list[str] = [
    "one more thing",
    "before you continue",
    "hold on a second",
]
