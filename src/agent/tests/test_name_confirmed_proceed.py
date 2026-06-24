"""Regression test for name persistence on the name-only partial re-ask path.

When a name field mismatches, _partial_reask resets name_confirmed. The corrected
name is then re-read-back and, on confirmation, _name_confirmed_proceed finds every
identity slot already populated and proceeds straight to the lookup via
_finish_after_identity. The dict it RETURNS (the post-lookup interrupt) must carry
name_confirmed=True, or the name gate re-fires the read-back on the next turn —
the loop the live E2E suite surfaced for verification_last_name_only_mismatch.

_finish_after_identity is stubbed so no live Salesforce / LLM calls occur.
"""

from agent.agents.verification.agent import VerificationAgent

ALL_SLOTS_STATE = {
    "first_name": "Emily",
    "last_name": "Carter",  # corrected
    "member_id": "M907503",
    "dob": "04/12/1988",
    "name_confirmed": False,  # was reset by the last-name partial re-ask
    "name_confirm_attempts": 1,
    "call_intent": "provider_services",
    "messages": [],
}


async def test_proceed_sets_name_confirmed_on_success(mocker):
    agent = VerificationAgent()

    # Stub the lookup tail: success path returns a post-lookup (relationship)
    # interrupt that does NOT carry name_confirmed on its own.
    async def fake_finish(state, collected, messages, call_intent, decision):
        return {
            "messages": {"role": "assistant", "content": "Are you the plan holder or dependent?"},
            "is_interrupt": True,
            "awaiting_slot": "relationship",
            "member_status_verify": True,
        }

    mocker.patch.object(agent, "_finish_after_identity", side_effect=fake_finish)

    result = await agent._name_confirmed_proceed(dict(ALL_SLOTS_STATE), [])

    # The just-confirmed name must be persisted so the gate never re-fires.
    assert result["name_confirmed"] is True
    assert result["name_confirm_attempts"] == 0
    # And it still routes to the post-lookup question (no Member-ID re-ask).
    assert result["awaiting_slot"] == "relationship"


async def test_proceed_respects_deliberate_name_confirmed_false(mocker):
    # If the re-run lookup itself returned a re-ask that deliberately reset
    # name_confirmed (e.g. a fresh name mismatch, or a full restart), that value
    # must be respected — not overwritten back to True.
    agent = VerificationAgent()

    async def fake_finish(state, collected, messages, call_intent, decision):
        return {
            "messages": {"role": "assistant", "content": "Let's try once more — your first name?"},
            "is_interrupt": True,
            "name_confirmed": False,
        }

    mocker.patch.object(agent, "_finish_after_identity", side_effect=fake_finish)

    result = await agent._name_confirmed_proceed(dict(ALL_SLOTS_STATE), [])

    assert result["name_confirmed"] is False
