"""
speech.py — TTS-friendly normalization of assistant messages.

spokenize_text() rewrites emails and web URLs embedded in a message into
their spoken form so the voice channel reads them naturally:

    jane.doe@example.com      →  jane dot doe at example dot com
    www.mysagilityhealth.com  →  www dot mysagilityhealth dot com

The transform is idempotent: spoken output contains no "@" and no "www."
token, so re-applying it is a no-op. This lets the emission layer
(SignalsMixin / orchestrator message_override) call it unconditionally,
even on messages a call site has already partially converted.
"""

from __future__ import annotations

import re

# Email: local part must contain at least one word char; domain needs a TLD.
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9][A-Za-z0-9._%+-]*@[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?\.[A-Za-z]{2,}\b")

# URL: scheme-prefixed or www.-prefixed hostnames, with optional path.
# Bare domains (example.com) are deliberately NOT matched — too many false
# positives in ordinary prose (e.g. file names, "5 p.m.").
_URL_RE = re.compile(
    r"\b(?:https?://)?www\.[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+(?:/[^\s,;!?]*)?",
    re.IGNORECASE,
)


def _speak_email(match: re.Match) -> str:
    return match.group(0).replace("@", " at ").replace(".", " dot ")


def _speak_url(match: re.Match) -> str:
    url = re.sub(r"^https?://", "", match.group(0))
    return url.replace(".", " dot ").replace("/", " slash ")


def spokenize_text(text: str) -> str:
    """Return `text` with any embedded emails and URLs in spoken form."""
    if not text:
        return text
    text = _EMAIL_RE.sub(_speak_email, text)
    text = _URL_RE.sub(_speak_url, text)
    return text
