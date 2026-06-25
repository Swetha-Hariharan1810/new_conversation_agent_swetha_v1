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

import difflib
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

# Unit ordinals used as the second word in space-separated compound ordinals
# such as "twenty fifth" (LLM returns without hyphen).
# Maps the unit word to its numeric value for combinations like:
#   "twenty" + "fifth" -> 20 + 5 = 25
#   "thirty" + "first" -> 30 + 1 = 31
_UNIT_ORDINALS: dict[str, int] = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
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


# ---------------------------------------------------------------------------
# Fuzzy fallback for misspelled spoken number-words (defense in depth)
# ---------------------------------------------------------------------------

# The cutoff is deliberately TIGHT. Obvious DOB typos are corrected upstream in
# the extraction prompt; this is only a safety net for the rare misspelling that
# still reaches normalization. A tight threshold means only near-exact typos (a
# doubled/dropped/transposed letter or two) are accepted — genuinely ambiguous or
# garbled tokens score below it and return None, so real ambiguous input still
# fails cleanly and neither the 8-digit ambiguity guard nor month matching is
# loosened. 0.75 (rather than 0.80) is used because common double-letter typos
# such as "twlevee" -> "twelve" only score ~0.77; anything materially looser
# would start admitting unrelated words.
_FUZZY_CUTOFF = 0.75


def _closest_number_word(token: str, mapping: dict[str, int]) -> str | None:
    """Return the closest valid number-word key for a misspelled token, or None.

    Candidates come from difflib.get_close_matches with the tight _FUZZY_CUTOFF.
    Among near-ties we prefer the candidate that shares the longest suffix with
    the token: number-word typos almost always preserve the ordinal/cardinal
    ending, so this resolves same-length neighbours correctly (e.g.
    "thirtheeth" -> "thirtieth", not the raw-similarity winner "thirteenth").
    """
    candidates = difflib.get_close_matches(token, list(mapping), n=3, cutoff=_FUZZY_CUTOFF)
    if not candidates:
        return None

    def _suffix_len(a: str, b: str) -> int:
        n = 0
        for ca, cb in zip(reversed(a), reversed(b)):
            if ca != cb:
                break
            n += 1
        return n

    return max(
        candidates,
        key=lambda c: (_suffix_len(token, c), difflib.SequenceMatcher(None, token, c).ratio()),
    )


def _resolve_number_word(mapping: dict[str, int], token: str) -> int | None:
    """Exact dictionary lookup with the tight fuzzy fallback above. Returns int or None."""
    if token in mapping:
        return mapping[token]
    match = _closest_number_word(token, mapping)
    return mapping[match] if match is not None else None


def _parse_spoken_year(tokens: list[str]) -> int | None:
    """Convert spoken year tokens like ['nineteen', 'eighty', 'eight'] → 1988.

    Also accepts a single numeric 4-digit token ('1989') for mixed
    spoken/numeric input like 'november second 1989'.
    """
    if not tokens:
        return None
    # Numeric year token, e.g. "1989" — common in mixed spoken/numeric input
    # like "november second 1989"
    if len(tokens) == 1 and re.fullmatch(r"\d{4}", tokens[0]):
        return int(tokens[0])
    first_val = _BASIC_NUMS.get(tokens[0])
    if first_val is None:
        return None
    if len(tokens) == 1:
        return first_val
    century = first_val * 100
    rest = tokens[1:]
    suffix = 0
    # Tens and the leading century word stay EXACT on purpose: a cardinal unit
    # like "eight" fuzzy-matches the tens word "eighty" (~0.91), so fuzzing the
    # tens slot would corrupt years (e.g. "nineteen eight" -> 1980). Only the
    # cardinal unit position gets the fuzzy fallback (e.g. "eigh" -> "eight").
    if len(rest) == 1:
        suffix = _TENS_WORDS.get(rest[0]) or _resolve_number_word(_BASIC_NUMS, rest[0]) or 0
    elif len(rest) >= 2:
        suffix = _TENS_WORDS.get(rest[0], 0) + (_resolve_number_word(_BASIC_NUMS, rest[1]) or 0)
    return century + suffix


def _spoken_date_to_mdy(text: str) -> str | None:
    """Try to parse a fully spoken date and return MM/DD/YYYY, or None.

    Handles hyphenated compound ordinals ('twenty-fifth'), space-separated
    compound ordinals ('twenty fifth'), and cardinal day words ('twelve',
    'two', 'twenty five'), so that dates like 'April twelve nineteen
    eighty eight' parse correctly as 04/12/1988.

    Also handles mixed spoken/numeric forms (only entered when a month
    word is present, so purely numeric dates never reach this path):
      - numeric year:  'november second 1989'              → 11/02/1989
      - numeric day:   'november 2 nineteen eighty nine'   → 11/02/1989
      - suffixed day:  'november 2nd 1989'                 → 11/02/1989
      - cardinal day:  'november two nineteen eighty nine' → 11/02/1989
    """
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

    # Find day word.
    #
    # Priority (order matters — compounds before singles, ordinals before
    # cardinals, so all pre-existing behavior is unchanged):
    #   1. compound ordinal:  tens + unit-ordinal   "twenty fifth"  → 25
    #   2. compound cardinal: tens + unit-cardinal  "twenty five"   → 25
    #   3. single ordinal:    _ORDINAL_TO_NUM       "twelfth"       → 12
    #   4. numeric:           "2", "2nd", "12th"    → 2 / 2 / 12
    #      The 1-2 digit bound means a 4-digit year token can never be
    #      consumed as the day; it falls through to `remaining` and is
    #      parsed as the year.
    #   5. single cardinal:   _BASIC_NUMS           "twelve"        → 12
    #
    # Without the compound check coming first, "twenty" falls through to
    # `remaining` and "fifth" is then matched as the standalone ordinal 5 —
    # yielding day=5 instead of the correct day=25, and leaving "twenty" in
    # the year tokens which makes year parsing fail entirely.
    #
    # Cardinal rules (2 and 5) require at least one token remaining AFTER
    # the day so the year can still be parsed. This prevents misparses:
    #   - "January nineteen twenty five" (year 1925, no day): without the
    #     guard, rule 2 would steal "twenty five" as day 25. Spoken years
    #     end in cardinals, never ordinals — which is why the ordinal
    #     rules need no such guard.
    #   - "June nineteen" (no day, no parseable year): cardinal "nineteen"
    #     is the last token → guard blocks it → day stays None → clean fail.
    #
    # When a cardinal IS wrongly consumed as the day (e.g. "June nineteen
    # eighty eight" spoken with no day: day=19, remaining "eighty eight"),
    # the year parse fails (_parse_spoken_year requires a _BASIC_NUMS
    # leading token) and the function returns None — the same retry
    # outcome as before this change.
    day_num: int | None = None
    remaining: list[str] = []
    i = 0
    while i < len(words_without_month):
        w = words_without_month[i]
        if day_num is None:
            # 1 & 2: compound forms — tens word + unit
            if w in _TENS_WORDS and i + 1 < len(words_without_month):
                nxt = words_without_month[i + 1]
                unit = _UNIT_ORDINALS.get(nxt)  # "fifth" → 5
                if unit is None:
                    # compound cardinal "twenty five" → 25; requires year
                    # tokens to remain after BOTH consumed words so a
                    # year-ending "twenty five" is never taken as the day
                    unit_card = _BASIC_NUMS.get(nxt)
                    if unit_card is not None and 1 <= unit_card <= 9 and i + 2 < len(words_without_month):
                        unit = unit_card
                if unit is None:
                    # fuzzy fallback for a misspelled unit-ordinal, e.g.
                    # "twenty fith" → "twenty fifth" → 25
                    fz_unit = _closest_number_word(nxt, _UNIT_ORDINALS)
                    if fz_unit is not None:
                        unit = _UNIT_ORDINALS[fz_unit]
                if unit is not None:
                    candidate = _TENS_WORDS[w] + unit
                    if 1 <= candidate <= 31:
                        day_num = candidate
                        i += 2
                        continue
            # 3: single ordinal — "twelfth", "twentieth", "thirtieth"
            if w in _ORDINAL_TO_NUM:
                day_num = _ORDINAL_TO_NUM[w]
                i += 1
                continue
            # 4: numeric day, with optional ordinal suffix: "2", "2nd"
            m = re.fullmatch(r"(\d{1,2})(?:st|nd|rd|th)?", w, flags=re.IGNORECASE)
            if m and 1 <= int(m.group(1)) <= 31:
                day_num = int(m.group(1))
                i += 1
                continue
            # 5: single cardinal day word — "twelve" → 12, "two" → 2.
            #    Guard: year tokens must remain after the day.
            card = _BASIC_NUMS.get(w)
            if card is not None and 1 <= card <= 31 and i + 1 < len(words_without_month):
                day_num = card
                i += 1
                continue
            # Fuzzy fallback (defense in depth): a single token may be a
            # misspelled number-word that exact lookup missed. Try ordinal
            # then cardinal with the tight cutoff before treating it as a
            # non-day token. Ordinal is tried first so an ordinal typo never
            # collapses onto a cardinal neighbour.
            fz_ord = _resolve_number_word(_ORDINAL_TO_NUM, w)
            if fz_ord is not None and 1 <= fz_ord <= 31:
                day_num = fz_ord
                i += 1
                continue
            fz_card = _resolve_number_word(_BASIC_NUMS, w)
            if fz_card is not None and 1 <= fz_card <= 31 and i + 1 < len(words_without_month):
                day_num = fz_card
                i += 1
                continue
        remaining.append(w)
        i += 1

    if day_num is None:
        return None

    year = _parse_spoken_year(remaining)
    if year is None:
        return None

    return f"{month_num:02d}/{day_num:02d}/{year}"


def _digit_string_to_mdy(text: str) -> str | None:
    """
    Handle DOB spoken as raw digit sequence e.g.
    'one nine eight five zero five zero three' → '05/03/1985'

    Tries YYYYMMDD first, then DDMMYYYY.
    Returns None if ambiguous, invalid, or not exactly 8 digits.
    """
    words = text.lower().split()

    # Only proceed if every token is a digit word — no month names, no ordinals
    if not all(w in _DIGIT_WORDS for w in words):
        return None

    if len(words) != 8:
        return None

    digits = "".join(_DIGIT_WORDS[w] for w in words)

    def try_yyyymmdd(d: str):
        year, month, day = int(d[0:4]), int(d[4:6]), int(d[6:8])
        from datetime import datetime

        try:
            dt = datetime(year, month, day)
            if 1900 <= year <= datetime.today().year:
                return dt.strftime("%m/%d/%Y")
        except ValueError:
            pass
        return None

    def try_ddmmyyyy(d: str):
        day, month, year = int(d[0:2]), int(d[2:4]), int(d[4:8])
        from datetime import datetime

        try:
            dt = datetime(year, month, day)
            if 1900 <= year <= datetime.today().year:
                return dt.strftime("%m/%d/%Y")
        except ValueError:
            pass
        return None

    yyyymmdd = try_yyyymmdd(digits)
    ddmmyyyy = try_ddmmyyyy(digits)

    if yyyymmdd and ddmmyyyy and yyyymmdd != ddmmyyyy:
        # Ambiguous — e.g. 01021985 could be Jan 2 or Feb 1
        return None

    return yyyymmdd or ddmmyyyy


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

    Handles:
      - Digit string: "one nine eight five zero five zero three" → "05/03/1985"
      - Spoken date (ordinal day):
          "April twelfth nineteen eighty eight"                 → "04/12/1988"
      - Spoken date (cardinal day):
          "April twelve nineteen eighty eight"                  → "04/12/1988"
      - Spoken date with space-separated compound day:
          "February twenty fifth nineteen ninety two"           → "02/25/1992"
          "February twenty five nineteen ninety two"            → "02/25/1992"
      - Mixed spoken/numeric forms:
          "november second 1989"                                → "11/02/1989"
          "november 2 nineteen eighty nine"                     → "11/02/1989"
      - MM/DD/YYYY, YYYY-MM-DD, and common separator variants.
    """
    cleaned = _clean(value)
    if not cleaned:
        return ""

    # 1. Try digit-string form first (all tokens are digit words, exactly 8)
    digit_result = _digit_string_to_mdy(cleaned)
    if digit_result:
        return digit_result

    # 2. Try spoken-date form (e.g. "April twelfth nineteen eighty eight",
    #    "April twelve nineteen eighty eight", "november second 1989")
    spoken_result = _spoken_date_to_mdy(cleaned)
    if spoken_result:
        return spoken_result

    # 3. Strip ordinal suffixes: "18th" → "18"
    cleaned = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", cleaned, flags=re.IGNORECASE)
    # Normalise separators to "/"
    cleaned = cleaned.replace("-", "/").replace(".", "/").replace(",", " ")

    formats = [
        "%m/%d/%Y",
        "%m/%d/%y",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%B %d %Y",
        "%b %d %Y",
        "%d %B %Y",
        "%d %b %Y",
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
    """Normalize caller relationship to 'plan_holder' | 'dependent' | ''."""
    if not value:
        return ""
    cleaned = value.strip().lower()
    if any(term in cleaned for term in _PLAN_HOLDER_TERMS) or any(
        term in cleaned for term in _SUBSCRIBER_TERMS
    ):
        return "plan_holder"
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
