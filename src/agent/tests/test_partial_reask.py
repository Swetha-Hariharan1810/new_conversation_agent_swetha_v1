"""Unit tests for the partial-re-ask handlers (Phase 3).

These exercise the extracted helpers in agents/verification/handlers.py directly,
with a minimal fake agent that mirrors signals.ask_member's behavior: it persists
confirmed (non-reset) slot values into the returned state dict, and skips slots
whose attempt counter was reset. This lets us assert the core acceptance —
matched fields + Member ID are preserved, only mismatches are cleared.
"""

from collections import defaultdict

from agent.agents.verification import handlers
from agent.agents.verification.constants import (
    MSG_REASK_DOB,
    MSG_REASK_FIRST_NAME,
    MSG_REASK_GENERIC,
    MSG_REASK_LAST_NAME,
)
from agent.agents.verification.handlers import MSG_RESTART

# Slot values that were confirmed earlier in the identity pipeline and would be
# persisted by ask_member for any slot whose counter was NOT reset.
CONFIRMED = {
    "member_id": "M123456",
    "first_name": "James",
    "last_name": "Anderson",
    "dob": "1977-07-13",
}


class _FakeSlot:
    def __init__(self):
        self.reset_called = False

    def reset(self):
        self.reset_called = True


class _FakeAgent:
    AGENT_NAME = "verification_agent"

    def __init__(self):
        self._slots = defaultdict(_FakeSlot)

    def get_slot(self, name):
        return self._slots[name]

    def ask_member(self, state, message):
        # Mirror signals.ask_member: base dict + persisted confirmed slots.
        # A reset slot is no longer confirmed, so it is not written back.
        result = {
            "messages": {"role": "assistant", "content": message},
            "is_interrupt": True,
            "next_node": self.AGENT_NAME,
        }
        for slot_name, value in CONFIRMED.items():
            slot = self._slots.get(slot_name)
            if slot is None or not slot.reset_called:
                result[slot_name] = value
        return result


def _state():
    return {
        # two prior messages → verification_restart_index should become 2
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "..."},
        ],
        "conversation_context": {
            "confirmed_slots": ["first_name", "last_name", "member_id", "dob"],
            "caller_first_name": "James",
        },
        "first_name": "James",
    }


# ── _reask_message selection ──────────────────────────────────────────────────


def test_reask_message_single_field_is_disclosing():
    assert handlers._reask_message(["dob"]) in MSG_REASK_DOB
    assert handlers._reask_message(["last_name"]) in MSG_REASK_LAST_NAME
    assert handlers._reask_message(["first_name"]) in MSG_REASK_FIRST_NAME


def test_reask_message_multi_field_is_generic():
    assert handlers._reask_message(["first_name", "dob"]) in MSG_REASK_GENERIC
    assert handlers._reask_message(["last_name", "dob"]) in MSG_REASK_GENERIC


# ── _partial_reask: DOB-only mismatch ─────────────────────────────────────────


def test_partial_reask_dob_only_clears_only_dob():
    agent = _FakeAgent()
    result = handlers._partial_reask(agent, _state(), ["dob"])

    # Only dob cleared; Member ID and matched name fields preserved.
    assert result["dob"] == ""
    assert result["member_id"] == "M123456"
    assert result["first_name"] == "James"
    assert result["last_name"] == "Anderson"

    # Only the dob counter was reset.
    assert agent._slots["dob"].reset_called is True
    assert agent._slots.get("first_name") is None
    assert agent._slots.get("member_id") is None

    # No name field mismatched → name confirmation untouched.
    assert "name_confirmed" not in result
    assert "name_confirm_attempts" not in result

    ctx = result["conversation_context"]
    assert ctx["confirmed_slots"] == ["first_name", "last_name", "member_id"]
    assert ctx["caller_first_name"] == "James"

    assert result["verification_restart_index"] == 2
    assert result["messages"]["content"] in MSG_REASK_DOB


# ── _partial_reask: last-name mismatch resets name confirmation ───────────────


def test_partial_reask_last_name_resets_name_confirmation_only():
    agent = _FakeAgent()
    result = handlers._partial_reask(agent, _state(), ["last_name"])

    assert result["last_name"] == ""
    assert result["first_name"] == "James"  # matched name field preserved
    assert result["member_id"] == "M123456"
    assert result["dob"] == "1977-07-13"

    # A name field mismatched → name confirmation reset.
    assert result["name_confirmed"] is False
    assert result["name_confirm_attempts"] == 0

    ctx = result["conversation_context"]
    assert ctx["confirmed_slots"] == ["first_name", "member_id", "dob"]
    # first_name matched, so the cached caller name is preserved.
    assert ctx["caller_first_name"] == "James"
    assert result["messages"]["content"] in MSG_REASK_LAST_NAME


# ── _partial_reask: first-name mismatch clears cached caller name ─────────────


def test_partial_reask_first_name_clears_cached_name():
    agent = _FakeAgent()
    result = handlers._partial_reask(agent, _state(), ["first_name"])

    assert result["first_name"] == ""
    assert result["last_name"] == "Anderson"
    assert result["member_id"] == "M123456"
    assert result["name_confirmed"] is False

    ctx = result["conversation_context"]
    assert ctx["caller_first_name"] == ""
    assert ctx["confirmed_slots"] == ["last_name", "member_id", "dob"]
    assert result["messages"]["content"] in MSG_REASK_FIRST_NAME


# ── _partial_reask: multi-field mismatch uses generic, keeps matched ─────────


def test_partial_reask_multi_field_uses_generic():
    agent = _FakeAgent()
    result = handlers._partial_reask(agent, _state(), ["last_name", "dob"])

    assert result["last_name"] == ""
    assert result["dob"] == ""
    # first_name + member_id matched → preserved.
    assert result["first_name"] == "James"
    assert result["member_id"] == "M123456"

    assert result["name_confirmed"] is False  # last_name is a name field
    ctx = result["conversation_context"]
    assert set(ctx["confirmed_slots"]) == {"first_name", "member_id"}
    assert result["messages"]["content"] in MSG_REASK_GENERIC


# ── _full_restart: wipes everything ──────────────────────────────────────────


def test_full_restart_wipes_all_identity_fields():
    agent = _FakeAgent()
    result = handlers._full_restart(agent, _state())

    assert result["first_name"] == ""
    assert result["last_name"] == ""
    assert result["member_id"] == ""
    assert result["dob"] == ""
    assert result["name_confirmed"] is False
    assert result["name_confirm_attempts"] == 0

    ctx = result["conversation_context"]
    assert ctx["caller_first_name"] == ""
    assert "first_name" not in ctx["confirmed_slots"]
    assert "last_name" not in ctx["confirmed_slots"]

    assert result["verification_restart_index"] == 2
    assert result["messages"]["content"] in MSG_RESTART
