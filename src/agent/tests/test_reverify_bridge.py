"""
Unit tests for the mid-call re-verification first-name bridge (one-shot).

When a fresh intent is detected mid-call, ``reset_for_new_intent`` stages the
new intent and sets ``reverify_bridge_pending=True``. On the next verification
entry, VerificationAgent must deliver the same deterministic first-name bridge
intake uses and pause — WITHOUT firing the extraction LLM call (the trigger
utterance carries no identity data). The flag is one-shot: it is cleared on that
turn so the following turn extracts normally.

These tests patch ``extract_verification_decision`` so it raises if awaited,
proving the bridge turn skips extraction, and confirm extraction IS reached on a
turn where the flag is absent (the normal/first-ever verification path).
"""

from __future__ import annotations

import pytest

import agent.agents.verification.agent as agent_mod
from agent.agents.intake.constants import INTENT_BRIDGE_MSGS
from agent.agents.verification.agent import VerificationAgent
from agent.state import reset_for_new_intent


class _ExtractionCalled(Exception):
    """Sentinel raised by the patched extractor to prove it was reached."""


@pytest.fixture(autouse=True)
def _forbid_extraction(monkeypatch):
    """Default: any call to the extraction LLM is a failure.

    ``get_extraction_llm`` is stubbed so no Azure credentials are needed; it is
    evaluated as an argument to ``extract_verification_decision`` and would
    otherwise raise before our sentinel.
    """

    async def _raise(*_args, **_kwargs):
        raise _ExtractionCalled("extract_verification_decision must not be awaited on the bridge turn")

    monkeypatch.setattr(agent_mod, "get_extraction_llm", lambda: object())
    monkeypatch.setattr(agent_mod, "extract_verification_decision", _raise)


def _reset_state(intent: str = "claim_services") -> dict:
    """A fresh post-reset state, as produced when follow_up reroutes mid-call."""
    state = reset_for_new_intent({}, intent)
    # The trigger utterance is already in the transcript when verification runs.
    state["messages"] = [
        {"role": "user", "content": "Actually, can you check a claim reprocessing for me?"},
    ]
    return state


async def test_bridge_fires_and_skips_extraction(monkeypatch):
    """First re-verification entry: emit the first-name bridge, skip extraction."""
    state = _reset_state()
    assert state["reverify_bridge_pending"] is True

    result = await VerificationAgent.from_state(state).run(state)

    # Bridge return shape.
    assert result["is_interrupt"] is True
    assert result["reverify_bridge_pending"] is False  # one-shot cleared
    assert result["awaiting_slot"] == "first_name"
    assert result["next_node"] == VerificationAgent.AGENT_NAME
    assert result["messages"]["content"] in INTENT_BRIDGE_MSGS
    # Every bridge message ends with the first-name ask.
    assert "your first name?" in result["messages"]["content"]


async def test_bridge_does_not_fire_when_flag_absent(monkeypatch):
    """No reverify_bridge_pending (first-ever / intake verification path) → the
    extraction LLM IS reached. The forbidding fixture's sentinel proves it."""
    state = _reset_state()
    state["reverify_bridge_pending"] = False  # simulate the normal/first-ever path

    with pytest.raises(_ExtractionCalled):
        await VerificationAgent.from_state(state).run(state)


async def test_following_turn_extracts_normally(monkeypatch):
    """After the bridge clears the flag, the member's name reply on the next turn
    routes into extraction (no longer short-circuited by the bridge)."""
    state = _reset_state()
    # Carry the bridge's one-shot clear forward, mimicking the next turn's state.
    state["reverify_bridge_pending"] = False
    state["awaiting_slot"] = "first_name"
    state["messages"] = [
        {"role": "assistant", "content": INTENT_BRIDGE_MSGS[0]},
        {"role": "user", "content": "emily"},
    ]

    with pytest.raises(_ExtractionCalled):
        await VerificationAgent.from_state(state).run(state)
