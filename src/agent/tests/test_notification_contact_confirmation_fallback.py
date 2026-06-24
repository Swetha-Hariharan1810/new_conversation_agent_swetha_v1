"""
Unit tests for the deterministic yes/no fallback in the phone/email
confirmation phases of NotificationSetupAgent (Sub-Agent 6b).

Regression target: when the extraction LLM returns an empty
``contact_confirmed`` for a clearly affirmative reply (it is biased to treat
the reply as a redundant acknowledgment because notification_method is passed
in as an already-confirmed slot), the confirmation phase used to fall through
to a non-advancing slot retry. The caller had to repeat "yes" several times.

The fix adds a deterministic ``normalize_yes_no(last_user)`` fallback, gated on
the absence of a replacement phone/email this turn so inline corrections still
route through the replacement path.
"""

from __future__ import annotations

import pytest

import agent.agents.notification_setup.agent as agent_mod
from agent.agents.notification_setup.agent import NotificationSetupAgent
from agent.llm.schema import WorkerResult


@pytest.fixture(autouse=True)
def _patch_llm(monkeypatch):
    """Stub out the LLM plumbing so tests drive extraction deterministically.

    ``extract_notification_decision`` is replaced per-test (see ``_extract``).
    ``get_extraction_llm`` / ``build_extraction_prompt_extraction`` are stubbed
    so no real LLM client or prompt file is required.
    """
    monkeypatch.setattr(agent_mod, "get_extraction_llm", lambda: object())
    monkeypatch.setattr(
        agent_mod, "build_extraction_prompt_extraction", lambda _path: "PROMPT"
    )


def _extract(monkeypatch, worker_result: WorkerResult) -> None:
    """Force ``extract_notification_decision`` to return ``worker_result``."""

    async def _fake_extract(*_args, **_kwargs):
        return worker_result

    monkeypatch.setattr(agent_mod, "extract_notification_decision", _fake_extract)


def _msgs(ai: str, human: str) -> list[dict]:
    return [
        {"role": "assistant", "content": ai},
        {"role": "user", "content": human},
    ]


# ── phone_confirmed ────────────────────────────────────────────────────────


@pytest.mark.parametrize("affirmative", ["yes thats correct", "yes", "yes please"])
async def test_phone_empty_extraction_affirmative_advances_first_turn(monkeypatch, affirmative):
    """Empty contact_confirmed + affirmative reply advances to save on turn 1."""
    _extract(monkeypatch, WorkerResult(extracted={}))

    state = {
        "messages": _msgs(
            "I'll send updates to 512-555-6101. Is that still the correct number?",
            affirmative,
        ),
        "awaiting_slot": "phone_confirmed",
        "notification_channel": "sms",
        "phone_number": "512-555-6101",
    }

    agent = NotificationSetupAgent.from_state(state)
    result = await agent.run(state)

    # Advanced straight to the timeline bridge (the _save_and_complete output),
    # NOT a re-ask of phone_confirmed.
    assert result["awaiting_slot"] == "timeline_question"
    assert result["notification_channel"] == "sms"
    assert result["claim_notification_contact"] == "512-555-6101"


async def test_phone_none_extraction_affirmative_advances(monkeypatch):
    """A fully empty WorkerResult (extracted=None) still advances on a 'yes'."""
    _extract(monkeypatch, WorkerResult())

    state = {
        "messages": _msgs(
            "I'll send updates to 512-555-6101. Is that still the correct number?",
            "yes thats correct",
        ),
        "awaiting_slot": "phone_confirmed",
        "notification_channel": "sms",
        "phone_number": "512-555-6101",
    }

    result = await NotificationSetupAgent.from_state(state).run(state)
    assert result["awaiting_slot"] == "timeline_question"
    assert result["claim_notification_contact"] == "512-555-6101"


async def test_phone_inline_replacement_still_routes_to_confirm(monkeypatch):
    """An inline correction ('no, use 415-555-0000') is NOT swallowed by the fallback.

    The extraction supplies the replacement phone; because a replacement was
    extracted this turn, the deterministic fallback must not fire and the turn
    must route through the phone read-back/confirm path.
    """
    _extract(monkeypatch, WorkerResult(extracted={"phone": "4155550000"}))

    state = {
        "messages": _msgs(
            "I'll send updates to 512-555-6101. Is that still the correct number?",
            "no, use 415-555-0000",
        ),
        "awaiting_slot": "phone_confirmed",
        "notification_channel": "sms",
        "phone_number": "512-555-6101",
    }

    result = await NotificationSetupAgent.from_state(state).run(state)

    # Re-confirms the NEW number — stays in phone_confirmed with pending_phone set.
    assert result["awaiting_slot"] == "phone_confirmed"
    assert result["pending_phone"] == "4155550000"
    assert "415-555-0000" in result["messages"]["content"]


async def test_phone_genuine_no_routes_to_update(monkeypatch):
    """An extracted 'no' routes to the phone-update prompt, not the fallback."""
    _extract(monkeypatch, WorkerResult(extracted={"contact_confirmed": "no"}))

    state = {
        "messages": _msgs(
            "I'll send updates to 512-555-6101. Is that still the correct number?",
            "no thats wrong",
        ),
        "awaiting_slot": "phone_confirmed",
        "notification_channel": "sms",
        "phone_number": "512-555-6101",
    }

    result = await NotificationSetupAgent.from_state(state).run(state)
    assert result["awaiting_slot"] == "phone"
    assert result["pending_phone"] == ""


# ── email_confirmed ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("affirmative", ["yes thats correct", "yes", "yes please"])
async def test_email_empty_extraction_affirmative_advances_first_turn(monkeypatch, affirmative):
    """Empty contact_confirmed + affirmative reply advances to save on turn 1."""
    _extract(monkeypatch, WorkerResult(extracted={}))

    state = {
        "messages": _msgs(
            "I'll send updates to jane at example dot com. Is that still correct?",
            affirmative,
        ),
        "awaiting_slot": "email_confirmed",
        "notification_channel": "email",
        "email": "jane@example.com",
    }

    result = await NotificationSetupAgent.from_state(state).run(state)

    assert result["awaiting_slot"] == "timeline_question"
    assert result["notification_channel"] == "email"
    assert result["claim_notification_contact"] == "jane@example.com"


async def test_email_inline_replacement_still_routes_to_confirm(monkeypatch):
    """An inline email correction is NOT swallowed by the fallback."""
    _extract(monkeypatch, WorkerResult(extracted={"email": "new@example.org"}))

    state = {
        "messages": _msgs(
            "I'll send updates to jane at example dot com. Is that still correct?",
            "no, use new@example.org",
        ),
        "awaiting_slot": "email_confirmed",
        "notification_channel": "email",
        "email": "jane@example.com",
    }

    result = await NotificationSetupAgent.from_state(state).run(state)

    assert result["awaiting_slot"] == "email_confirmed"
    assert result["pending_email"] == "new@example.org"


async def test_email_genuine_no_routes_to_update(monkeypatch):
    """An extracted 'no' routes to the email-update prompt, not the fallback."""
    _extract(monkeypatch, WorkerResult(extracted={"contact_confirmed": "no"}))

    state = {
        "messages": _msgs(
            "I'll send updates to jane at example dot com. Is that still correct?",
            "no thats wrong",
        ),
        "awaiting_slot": "email_confirmed",
        "notification_channel": "email",
        "email": "jane@example.com",
    }

    result = await NotificationSetupAgent.from_state(state).run(state)
    assert result["awaiting_slot"] == "email"
    assert result["pending_email"] == ""
