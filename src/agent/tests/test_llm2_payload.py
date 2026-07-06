"""
LLM-2 payload hygiene tests (Phase 1, fixes Bug D).

Covers:
  - mask_confirmed: slot-name masking + value-shape belt-and-braces pass
  - _is_reportable_slot: counter/flag pseudo-slots never reach Confirmed:
  - _tone_hint: attempt count → coarse tone label mapping
  - _render_payload: no raw member_id/dob leak (Phase-0 payload-leak test,
    formerly xfail), no Attempt:/Pending: lines, FOLLOWUP Collecting: semantics
"""

import pytest

from agent.llm.redaction import MASKED_SLOTS, _is_reportable_slot, mask_confirmed
from agent.llm.response_generator import _COLLECTING_NOTHING, _render_payload, _tone_hint

_HISTORY = [
    {"role": "assistant", "content": "Could I get your date of birth?"},
    {"role": "user", "content": "It's March first, nineteen ninety."},
]


# ── mask_confirmed ───────────────────────────────────────────────────────────


def test_mask_confirmed_masks_by_slot_name():
    values = {"member_id": "M451982", "dob": "03/01/1990", "first_name": "Ana"}
    assert mask_confirmed(values) == {
        "member_id": "on file",
        "dob": "on file",
        "first_name": "Ana",
    }


def test_mask_confirmed_masks_sensitive_shapes_under_any_name():
    # Belt-and-braces second pass: member_id/dob-shaped values are masked even
    # when they arrive under a slot name outside MASKED_SLOTS.
    values = {"mystery": "M123456", "other": "01/02/1990"}
    assert mask_confirmed(values) == {"mystery": "on file", "other": "on file"}


def test_mask_confirmed_stringifies_and_passes_through_safe_values():
    values = {"zip_code": 90210, "email": "ana@example.com", "relationship": "self"}
    assert mask_confirmed(values) == {
        "zip_code": "90210",
        "email": "ana@example.com",
        "relationship": "self",
    }


def test_mask_confirmed_handles_none_and_empty():
    assert mask_confirmed(None) == {}
    assert mask_confirmed({}) == {}


def test_masked_slots_single_source_of_truth():
    assert MASKED_SLOTS == frozenset({"member_id", "dob"})


# ── _is_reportable_slot ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name", ["name_confirmed", "phone_confirmed", "email_confirmed", "update_email", "intent_cycles", ""]
)
def test_pseudo_slots_are_not_reportable(name):
    assert not _is_reportable_slot(name)


@pytest.mark.parametrize("name", ["first_name", "last_name", "member_id", "dob", "zip_code", "email"])
def test_real_slots_are_reportable(name):
    assert _is_reportable_slot(name)


# ── _tone_hint ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("attempt", "expected"),
    [
        (0, "first ask"),
        (1, "first ask"),
        (2, "gentle retry"),
        (3, "patient retry"),
        (7, "patient retry"),
    ],
)
def test_tone_hint_mapping(attempt, expected):
    assert _tone_hint(attempt) == expected


# ── _render_payload — Phase-0 payload-leak test (un-xfailed) ────────────────


def test_payload_never_leaks_raw_sensitive_values():
    payload = _render_payload(
        slot_name="relationship",
        attempt=1,
        guard="RETRY",
        last_messages=_HISTORY,
        confirmed_slots={"member_id": "M451982", "dob": "03/01/1990", "first_name": "Ana"},
        user_utterance="what was that?",
    )
    assert "M451982" not in payload
    assert "03/01/1990" not in payload
    assert "member_id=on file" in payload
    assert "dob=on file" in payload
    assert "first_name=Ana" in payload


def test_payload_has_tone_but_never_attempt_or_pending():
    payload = _render_payload(
        slot_name="dob",
        attempt=2,
        guard="RETRY",
        last_messages=_HISTORY,
        confirmed_slots={"first_name": "Ana"},
    )
    assert "Tone:" in payload
    assert "gentle retry" in payload
    assert "Attempt:" not in payload
    assert "Pending:" not in payload


@pytest.mark.parametrize("guard", ["FOLLOWUP_ANSWER", "FOLLOWUP_PARK", "FOLLOWUP_DECLINE"])
def test_followup_payload_has_neither_attempt_nor_pending(guard):
    payload = _render_payload(
        slot_name="dob",
        attempt=0,
        guard=guard,
        last_messages=_HISTORY,
        confirmed_slots={"first_name": "Ana", "dob": "03/01/1990"},
        extracted_value="03/01/1990",
        followup_query="will I get a text about this?",
    )
    assert "Attempt:" not in payload
    assert "Pending:" not in payload
    assert f"Event:      {guard}" in payload


@pytest.mark.parametrize("guard", ["FOLLOWUP_ANSWER", "FOLLOWUP_PARK", "FOLLOWUP_DECLINE"])
def test_followup_with_captured_value_collects_nothing(guard):
    payload = _render_payload(
        slot_name="dob",
        attempt=0,
        guard=guard,
        last_messages=_HISTORY,
        extracted_value="03/01/1990",
        followup_query="how long does verification take?",
    )
    assert f"Collecting: {_COLLECTING_NOTHING}" in payload


def test_followup_decline_without_extraction_keeps_real_label():
    # No-extraction decline (e.g. non-updatable bare update request) still
    # re-asks the awaiting slot, so the real label must be rendered.
    payload = _render_payload(
        slot_name="dob",
        attempt=0,
        guard="FOLLOWUP_DECLINE",
        last_messages=_HISTORY,
        followup_query="update my mailing address",
    )
    assert _COLLECTING_NOTHING not in payload
    assert "Collecting: date of birth" in payload


def test_retry_with_extraction_keeps_real_label():
    # The Collecting: override applies to post-confirmation FOLLOWUP guards
    # only — a RETRY with a rejected extraction still re-asks the slot.
    payload = _render_payload(
        slot_name="member_id",
        attempt=1,
        guard="RETRY",
        last_messages=_HISTORY,
        extracted_value="451982",
    )
    assert _COLLECTING_NOTHING not in payload
    assert "Collecting: Member ID" in payload
