"""
slot_normalizer.py — Reusable conversational slot normalizers.

Normalizers transform raw conversational input into canonical structured values.
  - spoken DOB  → "10/18/1990"
  - noisy phone → digits only
  - "m 123456"  → "M123456"

Rules:
  - Deterministic and stateless
  - Reusable across all agents
  - Must NOT validate, mutate state, or emit signals
  - Name normalizers return "" for empty/blank input
"""

from __future__ import annotations

import re
from typing import Callable, Optional

__all__ = [
    "normalize_name",
    "normalize_member_id",
    "normalize_dob",
    "normalize_zip_code",
    "normalize_phone_number",
    "normalize_fax_number",
    "normalize_email",
    "normalize_yes_no",
    "normalize_caller_role",
    "normalize_provider_type",
    "normalize_delivery_method",
    "NORMALIZER_REGISTRY",
    "get_normalizer",
    "normalize_slot_value",
]

# ---------------------------------------------------------------------------
# Shared cleanup
# ---------------------------------------------------------------------------


def _clean(value: str | None) -> str:
    return (value or "").strip().replace("\n", " ").replace("\t", " ")


# ---------------------------------------------------------------------------
# Spoken-word → character mappings
# ---------------------------------------------------------------------------

# For zip / phone: "oh" and "zero" both map to digit "0"
_DIGIT_WORDS: dict[str, str] = {
    "zero": "0",
    "oh": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}

# For member IDs: "oh" → letter "O", "zero" → digit "0"
_MEMBER_ID_WORD_MAP: dict[str, str] = {
    **_DIGIT_WORDS,
    "oh": "O",  # spoken letter O (overrides the "0" from _DIGIT_WORDS)
}

# ---------------------------------------------------------------------------
# Spoken date helpers
# ---------------------------------------------------------------------------

_ORDINAL_TO_NUM: dict[str, int] = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
    "thirtieth": 30,
    "thirtyth": 30,
    "twenty-first": 21,
    "twenty-second": 22,
    "twenty-third": 23,
    "twenty-fourth": 24,
    "twenty-fifth": 25,
    "twenty-sixth": 26,
    "twenty-seventh": 27,
    "twenty-eighth": 28,
    "twenty-ninth": 29,
    "thirty-first": 31,
}

_TENS_WORDS: dict[str, int] = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

_BASIC_NUMS: dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}

_MONTH_NAMES: dict[str, int] = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def _parse_spoken_year(tokens: list[str]) -> int | None:
    """Convert spoken year tokens like ['nineteen', 'eighty', 'eight'] → 1988."""
    if not tokens:
        return None
    first_val = _BASIC_NUMS.get(tokens[0])
    if first_val is None:
        return None
    if len(tokens) == 1:
        return first_val
    century = first_val * 100
    rest = tokens[1:]
    suffix = 0
    if len(rest) == 1:
        suffix = _TENS_WORDS.get(rest[0]) or _BASIC_NUMS.get(rest[0]) or 0
    elif len(rest) >= 2:
        suffix = _TENS_WORDS.get(rest[0], 0) + _BASIC_NUMS.get(rest[1], 0)
    return century + suffix


def _spoken_date_to_mdy(text: str) -> str | None:
    """Try to parse a fully spoken date and return MM/DD/YYYY, or None."""
    words = text.lower().replace("-", " ").split()
    words = [w for w in words if w not in {"of", "the"}]

    # Find month word
    month_idx: int | None = None
    month_num: int | None = None
    for i, w in enumerate(words):
        if w in _MONTH_NAMES:
            month_idx = i
            month_num = _MONTH_NAMES[w]
            break

    if month_num is None:
        return None

    assert month_idx is not None
    words_without_month = words[:month_idx] + words[month_idx + 1 :]

    # Find ordinal day word
    day_num: int | None = None
    remaining: list[str] = []
    for w in words_without_month:
        if w in _ORDINAL_TO_NUM and day_num is None:
            day_num = _ORDINAL_TO_NUM[w]
        else:
            remaining.append(w)

    if day_num is None:
        return None

    year = _parse_spoken_year(remaining)
    if year is None:
        return None

    return f"{month_num:02d}/{day_num:02d}/{year}"


def _convert_spoken_digits(text: str) -> str:
    """Replace spoken digit words with their numeric equivalents: 'one six' → '1 6'."""
    words = text.lower().split()
    return " ".join(_DIGIT_WORDS.get(w, w) for w in words)


# ---------------------------------------------------------------------------
# Name
# ---------------------------------------------------------------------------


def normalize_name(value: str | None) -> str:
    """Normalize a personal name to Title Case."""
    cleaned = _clean(value)
    if not cleaned:
        return ""
    # Use title() so apostrophes in names like "O'Brien" are handled correctly
    return " ".join(word.title() for word in cleaned.split())


# ---------------------------------------------------------------------------
# Member ID
# ---------------------------------------------------------------------------


def normalize_member_id(value: str | None) -> str:
    """Normalize member ID: uppercase, strip non-alphanumeric characters.

    Handles spoken words: "m nine zero seven five oh three" → "M9075O3"
    "oh" maps to the letter O; digit words map to their digit.
    """
    cleaned = _clean(value)
    words = cleaned.lower().split()
    parts = [_MEMBER_ID_WORD_MAP.get(w, w) for w in words]
    joined = "".join(parts).upper()
    return re.sub(r"[^A-Z0-9]", "", joined)


# ---------------------------------------------------------------------------
# Date of Birth
# ---------------------------------------------------------------------------


def normalize_dob(value: str | None) -> str:
    """Normalize DOB to MM/DD/YYYY.

    Handles: MM/DD/YYYY, YYYY-MM-DD, spoken dates ("October 18th 1990"),
    ordinal suffixes ("18th" → "18"), and common separators.
    Also handles fully spoken dates like "April twelfth nineteen eighty eight".
    """
    cleaned = _clean(value)
    if not cleaned:
        return ""

    # Try spoken-date form first (e.g. "April twelfth nineteen eighty eight")
    spoken_result = _spoken_date_to_mdy(cleaned)
    if spoken_result:
        return spoken_result

    # Strip ordinal suffixes: "18th" → "18"
    cleaned = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", cleaned, flags=re.IGNORECASE)
    # Normalise separators to "/"
    cleaned = cleaned.replace("-", "/").replace(".", "/").replace(",", " ")

    formats = [
        "%m/%d/%Y",  # 10/18/1990  (canonical output + MM/DD/YYYY input)
        "%m/%d/%y",  # 10/18/90
        "%Y/%m/%d",  # 1990/10/18  (LLM YYYY-MM-DD after sep-replace)
        "%d/%m/%Y",  # 18/10/1990  (DD/MM/YYYY)
        "%B %d %Y",  # October 18 1990
        "%b %d %Y",  # Oct 18 1990
        "%d %B %Y",  # 18 October 1990
        "%d %b %Y",  # 18 Oct 1990
    ]
    from datetime import datetime

    for fmt in formats:
        try:
            parsed = datetime.strptime(cleaned.strip(), fmt)
            if fmt == "%m/%d/%y" and parsed.year > datetime.today().year:
                parsed = parsed.replace(year=parsed.year - 100)
            return parsed.strftime("%m/%d/%Y")
        except ValueError:
            continue
    return cleaned


# ---------------------------------------------------------------------------
# ZIP Code
# ---------------------------------------------------------------------------


def normalize_zip_code(value: str | None) -> str:
    """Normalize to 5-digit ZIP string. Handles spoken digits: 'one six seven eight three' → '16783'."""
    converted = _convert_spoken_digits(_clean(value))
    return re.sub(r"\D", "", converted)[:5]


# ---------------------------------------------------------------------------
# Phone Number
# ---------------------------------------------------------------------------


def normalize_phone_number(value: str | None) -> str:
    """Normalize to 10-digit string (strips US country code if present).

    Handles spoken digits: 'four one five five five five three two one one' → '4155553211'.
    """
    converted = _convert_spoken_digits(_clean(value))
    digits = re.sub(r"\D", "", converted)
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    return digits


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------


def normalize_email(value: str | None) -> str:
    """Normalize email to lowercase."""
    return _clean(value).lower()


# ---------------------------------------------------------------------------
# Yes / No
# ---------------------------------------------------------------------------


def normalize_yes_no(value: str | None) -> str:
    """
    Safety net for non-canonical outputs. Primary yes/no mapping is
    handled by the extraction LLM via semantic prompts. This function
    catches residual cases where the LLM returns a colloquial variant
    instead of the canonical "yes" or "no".
    """
    if not value:
        return ""
    cleaned = _clean(value).lower().strip()

    if cleaned in ("yes", "no"):
        return cleaned

    if cleaned in {
        "yeah",
        "yep",
        "yup",
        "ye",
        "sure",
        "ok",
        "okay",
        "correct",
        "right",
        "true",
        "uh huh",
        "mm hmm",
        "absolutely",
        "definitely",
        "affirmative",
    }:
        return "yes"

    if cleaned in {
        "nope",
        "nah",
        "wrong",
        "incorrect",
        "false",
        "negative",
    }:
        return "no"

    if cleaned.startswith(("yes ", "yeah ", "yep ", "yup ", "sure ", "correct ")):
        return "yes"
    if cleaned.startswith(("no ", "nope ", "nah ")):
        return "no"

    return cleaned


# ---------------------------------------------------------------------------
# Caller Role / Relationship
# ---------------------------------------------------------------------------

_PLAN_HOLDER_TERMS = {
    "plan holder",
    "planholder",
    "plan_holder",
    "myself",
    "me",
    "primary",
    "account holder",
}
_SUBSCRIBER_TERMS = {"subscriber", "insured", "policy holder", "policyholder"}
_DEPENDENT_TERMS = {"spouse", "dependent", "child", "family member", "my wife", "my husband", "my partner"}


def normalize_caller_role(value: str | None) -> str:
    """Normalize caller relationship to 'plan_holder' | 'subscriber' | 'dependent' | ''."""
    if not value:
        return ""
    cleaned = value.strip().lower()
    if any(term in cleaned for term in _PLAN_HOLDER_TERMS):
        return "plan_holder"
    if any(term in cleaned for term in _SUBSCRIBER_TERMS):
        return "subscriber"
    if any(term in cleaned for term in _DEPENDENT_TERMS):
        return "dependent"
    return ""


# ---------------------------------------------------------------------------
# Provider type / delivery method / fax normalizers
# ---------------------------------------------------------------------------

_PROVIDER_TYPE_MAP: dict[str, str] = {
    "pcp": "Primary Care Physician",
    "primary care": "Primary Care Physician",
    "primary care physician": "Primary Care Physician",
    "family doctor": "Primary Care Physician",
    "family physician": "Primary Care Physician",
    "general practitioner": "Primary Care Physician",
    "my doctor": "Primary Care Physician",
    "regular doctor": "Primary Care Physician",
    "pediatrician": "Pediatrician",
    "kids doctor": "Pediatrician",
    "children's doctor": "Pediatrician",
    "child doctor": "Pediatrician",
    "cardiologist": "Cardiologist",
    "heart doctor": "Cardiologist",
    "heart specialist": "Cardiologist",
    "dermatologist": "Dermatologist",
    "skin doctor": "Dermatologist",
    "orthopedic": "Orthopedic Specialist",
    "orthopedist": "Orthopedic Specialist",
    "bone doctor": "Orthopedic Specialist",
    "joint doctor": "Orthopedic Specialist",
    "joints": "Orthopedic Specialist",
}

_DELIVERY_METHOD_MAP: dict[str, str] = {
    # Canonical pass-through
    "fax": "fax",
    "email": "email",
    # Residual non-canonical values the LLM may occasionally return
    "e-mail": "email",
    "mail": "email",
    "electronic": "email",
}


def normalize_provider_type(value: str | None) -> str:
    """Normalize spoken provider type to canonical name."""
    if not value:
        return ""
    cleaned = _clean(value).lower().strip()
    # Exact / substring match in priority order (longest first avoids partial shadowing)
    for key in sorted(_PROVIDER_TYPE_MAP, key=len, reverse=True):
        if key in cleaned:
            return _PROVIDER_TYPE_MAP[key]
    return ""


def normalize_delivery_method(value: str | None) -> str:
    """Normalize delivery method to 'fax' or 'email'."""
    if not value:
        return ""
    cleaned = _clean(value).lower()
    for key in sorted(_DELIVERY_METHOD_MAP, key=len, reverse=True):
        if key in cleaned:
            return _DELIVERY_METHOD_MAP[key]
    return ""


def normalize_fax_number(value: str | None) -> str:
    """Normalize fax number — identical to phone_number normalization."""
    return normalize_phone_number(value)


def normalize_reference_number(value: str | None) -> str:
    """
    Normalize adjustment reference number to digits only.
    Handles spoken digits: "one two four nine one five eight four" -> "12491584"
    """
    converted = _convert_spoken_digits(_clean(value))
    return re.sub(r"\D", "", converted)


def normalize_notification_method(value: str | None) -> str:
    """Normalize notification method to 'sms' or 'email'."""
    if not value:
        return ""
    cleaned = _clean(value).lower()
    _NOTIFICATION_MAP = {
        "sms": "sms",
        "text": "sms",
        "phone": "sms",
        "my phone": "sms",
        "call": "sms",
        "email": "email",
        "e-mail": "email",
        "mail": "email",
    }
    for key in sorted(_NOTIFICATION_MAP, key=len, reverse=True):
        if key in cleaned:
            return _NOTIFICATION_MAP[key]
    return ""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

NORMALIZER_REGISTRY: dict[str, Callable[[Optional[str]], str]] = {
    "normalize_name": normalize_name,
    "normalize_member_id": normalize_member_id,
    "normalize_dob": normalize_dob,
    "normalize_zip_code": normalize_zip_code,
    "normalize_phone_number": normalize_phone_number,
    "normalize_fax_number": normalize_fax_number,
    "normalize_reference_number": normalize_reference_number,
    "normalize_notification_method": normalize_notification_method,
    "normalize_email": normalize_email,
    "normalize_yes_no": normalize_yes_no,
    "normalize_caller_role": normalize_caller_role,
    "normalize_provider_type": normalize_provider_type,
    "normalize_delivery_method": normalize_delivery_method,
}


def get_normalizer(name: str) -> Callable[[Optional[str]], str]:
    if name not in NORMALIZER_REGISTRY:
        raise ValueError(f"Unknown normalizer: {name}")
    return NORMALIZER_REGISTRY[name]


def normalize_slot_value(normalizer_name: str | None, value: str | None) -> str:
    if not normalizer_name:
        return value or ""
    return get_normalizer(normalizer_name)(value)
