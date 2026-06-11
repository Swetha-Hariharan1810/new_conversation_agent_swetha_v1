"""
test_email_readback_spoken.py — Phase 3 regression: ad-hoc @ replacements are
gone, the raw email feeds the message text, and the central sanitizer in
signals.py produces the fully spoken form (at + dot) on emission, while state
keys (email / pending_email) keep the raw address.
"""

from __future__ import annotations

import agent.agents.delivery_management.agent as dm_module
from agent.agents.delivery_management.agent import DeliveryManagementAgent
from agent.llm.schema import WorkerResult


async def test_delivery_email_readback_spoken_but_state_raw(monkeypatch):
    async def fake_extract(*args, **kwargs):
        return WorkerResult(extracted={"email": "jane.doe@example.com"})

    monkeypatch.setattr(dm_module, "extract_delivery_management_decision", fake_extract)
    monkeypatch.setattr(dm_module, "get_extraction_llm", lambda: None)

    state = {
        "messages": [
            {"role": "assistant", "content": "Is the email on file correct?"},
            {"role": "user", "content": "No, use my new one"},
        ],
        "awaiting_slot": "email_confirmed",
        "delivery_method": "email",
        "email": "old.address@example.com",
        "slot_attempts": {},
        "app_run_id": "run-dm",
    }

    agent = DeliveryManagementAgent.from_state(state)
    result = await agent.run(state)

    content = result["messages"]["content"]
    # Outgoing message is fully spoken: "at" and "dot", no "@", no raw dotted email.
    assert "jane dot doe at example dot com" in content
    assert "@" not in content
    assert "jane.doe" not in content
    assert "example.com" not in content
    # State keeps the raw addresses.
    assert result["pending_email"] == "jane.doe@example.com"
    assert result["email"] == "old.address@example.com"
