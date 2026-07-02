"""
grounding.py — guardrails on generated (spoken) text.

Phase 1 unifies every turn's voice onto one grounded generator. Two guardrails
keep that voice safe, and both are pure functions so they can run as assertions
now (tests) and as live checks later (the Phase-4 grounding gate):

  * ``find_ungrounded_values`` — the Phase-4 grounding check, added early: a
    generated turn must not state a concrete value (Member ID, ZIP, date, phone,
    email, reference number) that wasn't grounded this turn — i.e. not in
    ``confirmed_slots ∪ validated_answer`` (plus a known first name). Anything
    value-shaped in the text that isn't in the allowed set is returned.
  * ``has_false_accept_opener`` — on a wrong-format RETRY the reply must never
    open as if the invalid value was accepted ("Thank you", "Got it", …).

``turn_grounding_allowlist`` builds the complete per-turn allowed set for the
first check — including any value the agent itself deliberately reads back this
turn — so the guard can never veto a read-back the prompt legitimately asked for.

None of these change any behavior on their own; they only inspect text/state.
"""

from __future__ import annotations

import re

# ── Value shapes we consider "concrete" and therefore must be grounded ──────────
# Deliberately conservative: only clearly identifier-shaped tokens, so ordinary
# words and small counts ("one moment", "30 seconds") are never flagged.
_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bM\d{6,}\b", re.IGNORECASE),  # Member ID (M + 6+ digits)
    re.compile(r"\b\d{5}(?:-\d{4})?\b"),  # ZIP / ZIP+4
    re.compile(r"\b\d{8,}\b"),  # reference numbers, long digit runs
    re.compile(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"),  # phone
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),  # dates
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),  # email
    # Spoken digit read-backs: 5+ single digits separated one-by-one ("1 2 1 3 9").
    # _normalize collapses these to the contiguous form ("12139") for the
    # allowed-set comparison, so a spaced read-back matches the value on file.
    re.compile(r"\b\d(?:[ .\-]\d){4,}\b"),
)

# Openers that imply the value was accepted — forbidden on a wrong-format retry.
FALSE_ACCEPT_OPENERS: tuple[str, ...] = (
    "thank you",
    "thanks",
    "got it",
    "i see",
    "i understand",
    "understood",
    "okay",
    "ok",
    "sure",
    "great",
    "perfect",
    "appreciate that",
    "appreciate it",
    "of course",
    "certainly",
    "absolutely",
    "wonderful",
    "excellent",
)


def _normalize(value: str) -> str:
    """Collapse a value to comparable form: lowercase alphanumerics only."""
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


def _extract_values(text: str) -> list[str]:
    found: list[str] = []
    for pattern in _VALUE_PATTERNS:
        found.extend(m.group(0) for m in pattern.finditer(text or ""))
    return found


def find_ungrounded_values(text: str, allowed: object) -> list[str]:
    """Return concrete values in ``text`` that are NOT in ``allowed``.

    ``allowed`` is any iterable of grounded values (confirmed slot values, the
    value validated this turn, a known first name). Comparison is normalized
    (case- and punctuation-insensitive) so "M123456" matches "m 123 456". An
    empty result means the text is grounded.
    """
    allowed_norm = {_normalize(str(a)) for a in (allowed or []) if str(a).strip()}
    ungrounded: list[str] = []
    for value in _extract_values(text):
        norm = _normalize(value)
        if not norm:
            continue
        # Grounded if it equals, or is a substring of / superstring of, an allowed
        # value (handles "M123456" vs a spoken "123456", ZIP vs ZIP+4, etc.).
        if any(norm == a or norm in a or a in norm for a in allowed_norm):
            continue
        ungrounded.append(value)
    return ungrounded


def is_grounded(text: str, allowed: object) -> bool:
    return not find_ungrounded_values(text, allowed)


# ── Per-turn allow-list ──────────────────────────────────────────────────────────

# State keys the agent deliberately reads back to the caller while collecting a
# given slot (the ZIP on file during a zip-confirmation act, the phone on file
# during phone_confirmed, the email being confirmed). The prompt asks the model
# to SPEAK these, so the grounding guard must accept them.
_READBACK_STATE_KEYS: dict[str, tuple[str, ...]] = {
    "zip_confirmed": ("zip_code",),
    "zip_code": ("zip_code",),
    "phone_confirmed": ("phone_number",),
    "phone_confirmation": ("phone_number",),
    "phone": ("phone_number",),
    "email_confirmed": ("pending_email", "email"),
    "email": ("pending_email", "email"),
}


def _readback_variants(value: object) -> list[str]:
    """A read-back value plus its formatted variants — contiguous digits and the
    spaced spoken form ("12139" and "1 2 1 3 9") — so the guard accepts the value
    however the generator reads it out."""
    text = str(value or "").strip()
    if not text:
        return []
    variants = [text]
    digits = re.sub(r"\D", "", text)
    if digits:
        variants.append(digits)
        variants.append(" ".join(digits))
    return variants


def turn_grounding_allowlist(
    state,
    ctx,
    *,
    extracted_value: str | None = None,
    answered_inline: list[str] | None = None,
    slot_name: str | None = None,
    readback_values: list[str] | None = None,
) -> list[str]:
    """The COMPLETE allow-list of concrete values one generated turn may speak.

    Returns the union of:
      * the value validated this turn (``extracted_value``);
      * inline answers already written for the model (``answered_inline``);
      * the caller's first name (from ``ctx``);
      * any value the agent itself is deliberately reading back this turn —
        derived from ``state`` for the slot being collected (``slot_name``,
        defaulting to ``state["awaiting_slot"]``), plus any explicit
        ``readback_values`` the call site is putting in the prompt/directive —
        each with its formatted variants (spaced digits and contiguous).

    Pass the result as ``grounded_values=`` to ``generate_recovery_message`` at
    every call site that reads a value back, so the grounding guard can never
    contradict what the prompt legitimately asked the model to say.
    """
    allowed: list[str] = []

    def _add(value: object) -> None:
        for variant in _readback_variants(value):
            if variant not in allowed:
                allowed.append(variant)

    _add(extracted_value)
    for ans in answered_inline or []:
        _add(ans)
    if ctx is not None:
        _add(getattr(ctx, "caller_first_name", None))
    for value in readback_values or []:
        _add(value)

    getter = state.get if hasattr(state, "get") else (lambda _k: None)
    slot = slot_name or (getter("awaiting_slot") or "")
    for key in _READBACK_STATE_KEYS.get(slot, ()):
        _add(getter(key))
    return allowed


def has_false_accept_opener(text: str) -> bool:
    """True when ``text`` opens with a phrase implying a value was accepted."""
    lowered = (text or "").lstrip().lower()
    for opener in FALSE_ACCEPT_OPENERS:
        if lowered.startswith(opener):
            # Must be a word boundary (avoid matching "understanding" via "understand").
            rest = lowered[len(opener) :]
            if not rest or not rest[0].isalnum():
                return True
    return False
