"""
test_signals_spoken.py — central spoken-text enforcement at message emission.

Every assistant message enters LangGraph state through SignalsMixin
(ask_member / _build / _emergency) or the orchestrator's message_override.
These tests pin the contract: outgoing content is spokenized at those choke
points, and nothing else in the result dict is transformed.
"""

from __future__ import annotations

from agent.agents.escalation.agent import escalation_agent
from agent.core.agent import BaseAgent
from agent.speech import spokenize_text


class _StubAgent(BaseAgent):
    AGENT_NAME = "stub_agent"

    async def run(self, state):  # pragma: no cover — never exercised
        return {}


RAW = (
    "I'll send it to jane.doe@example.com. You can also track rewards at "
    "www.mysagilityhealth.com under the My Wellness section."
)
SPOKEN = (
    "I'll send it to jane dot doe at example dot com. You can also track rewards at "
    "www dot mysagilityhealth dot com under the My Wellness section."
)


def test_spokenize_text_converts_email_and_url():
    assert spokenize_text(RAW) == SPOKEN


def test_spokenize_text_is_idempotent():
    assert spokenize_text(SPOKEN) == SPOKEN


def test_ask_member_emits_spoken_message_and_keeps_other_keys():
    agent = _StubAgent()
    result = agent.ask_member({"app_run_id": "run-123"}, RAW)
    assert result == {
        "messages": {"role": "assistant", "content": SPOKEN},
        "next_node": "stub_agent",
        "is_interrupt": True,
        "active_agent": "stub_agent",
        "slot_attempts": {},
        "metadata_events": [],
        "app_run_id": "run-123",
    }


def test_signal_complete_emits_spoken_message():
    agent = _StubAgent()
    result = agent.signal_complete({"app_run_id": ""}, RAW)
    assert result["messages"] == {"role": "assistant", "content": SPOKEN}


def test_signal_complete_leaves_context_updates_raw():
    agent = _StubAgent()
    result = agent.signal_complete(
        {"app_run_id": ""}, RAW, context_updates={"email": "jane.doe@example.com"}
    )
    assert result["email"] == "jane.doe@example.com"


def test_emergency_message_passes_through_spokenize():
    agent = _StubAgent()
    result = agent._emergency({"ref_no": "REF123456"}, "boom")
    # MSG_EMERGENCY contains no email/URL — spokenize must be a no-op here.
    assert result["messages"]["content"] == "Technical issue. Reference: REF123456."


async def test_escalation_pre_message_is_covered_by_build():
    # escalation_agent folds escalation_pre_message into the message it hands
    # to signal_complete → _build, so the central wrap covers it — no second
    # transform needed in the escalation agent itself.
    state = {
        "escalation_pre_message": "I'll have a specialist email you at jane.doe@example.com",
        "ref_no": "REF999000111",
        "escalation_reason": "no_delivery_contact",
        "metadata_events": [],
        "app_run_id": "run-esc",
    }
    result = await escalation_agent(state)
    content = result["messages"]["content"]
    assert "jane dot doe at example dot com" in content
    assert "@" not in content
    assert "REF999000111" in content
