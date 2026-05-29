"""
test_verification_agent_mock.py — Comprehensive mock test suite for VerificationAgent.

No external credentials required. Covers all event_type branches, guard triggers,
correction + ambiguous counter logic, retry exhaustion, bonus extraction, and
response content checks.

Run all:    pytest src/agent/tests/test_verification_agent_mock.py -v
By marker:  pytest src/agent/tests/test_verification_agent_mock.py -v -m happy
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.agents.verification.agent import VerificationAgent
from agent.llm.schema import EventType, GuardType, WorkerResult
from agent.tests.fixtures import (
    VERIFIED_MEMBER,
    advance,
    get_ambiguous,
    get_attempt,
    get_awaiting,
    get_response,
    is_ask,
    is_complete,
    is_escalation,
    make_state,
    make_verified_state,
)

# ---------------------------------------------------------------------------
# Preset WorkerResult factories
# ---------------------------------------------------------------------------

SLOT_ANSWERS: dict[str, WorkerResult] = {
    "first_name": WorkerResult(
        extracted={"first_name": "Emily"},
        event_type=EventType.ANSWERED,
        guard=GuardType.NONE,
        guard_confidence=0.0,
    ),
    "last_name": WorkerResult(
        extracted={"last_name": "Carter"},
        event_type=EventType.ANSWERED,
        guard=GuardType.NONE,
        guard_confidence=0.0,
    ),
    "member_id": WorkerResult(
        extracted={"member_id": "M907503"},
        event_type=EventType.ANSWERED,
        guard=GuardType.NONE,
        guard_confidence=0.0,
    ),
    "dob": WorkerResult(
        extracted={"dob": "04/12/1988"},
        event_type=EventType.ANSWERED,
        guard=GuardType.NONE,
        guard_confidence=0.0,
    ),
    "relationship": WorkerResult(
        extracted={"relationship": "plan holder"},
        event_type=EventType.ANSWERED,
        guard=GuardType.NONE,
        guard_confidence=0.0,
    ),
    "phone_confirmed": WorkerResult(
        extracted={"phone_confirmed": "true"},
        event_type=EventType.ANSWERED,
        guard=GuardType.NONE,
        guard_confidence=0.0,
    ),
    "phone_confirmation": WorkerResult(
        extracted={"phone_confirmed": "true"},
        event_type=EventType.ANSWERED,
        guard=GuardType.NONE,
        guard_confidence=0.0,
    ),
}

EMPTY_ANSWERED = WorkerResult(event_type=EventType.ANSWERED)


def make_correction(slot: str, value: str) -> WorkerResult:
    return WorkerResult(
        corrections={slot: value}, event_type=EventType.CORRECTED, guard=GuardType.NONE, guard_confidence=0.0
    )


def make_ambiguous() -> WorkerResult:
    return WorkerResult(
        corrections={}, event_type=EventType.AMBIGUOUS, guard=GuardType.NONE, guard_confidence=0.0
    )


def make_guard(guard: GuardType) -> WorkerResult:
    return WorkerResult(extracted=None, event_type=EventType.NONE, guard=guard, guard_confidence=0.95)


# ---------------------------------------------------------------------------
# Mock helper
# ---------------------------------------------------------------------------


class _MockHelper:
    def __init__(self, mock: AsyncMock) -> None:
        self._mock = mock

    def set_responses(self, responses: list) -> "_MockHelper":
        counter = [0]
        _resp = list(responses)

        async def _se(*args, **kwargs):
            i = counter[0]
            counter[0] += 1
            return _resp[i] if i < len(_resp) else EMPTY_ANSWERED

        self._mock.side_effect = _se
        return self

    def set_single(self, response: WorkerResult) -> "_MockHelper":
        async def _se(*args, **kwargs):
            return response

        self._mock.side_effect = _se
        return self

    def set_slot_answers(self) -> "_MockHelper":
        async def _se(*args, **kwargs):
            return SLOT_ANSWERS.get(kwargs.get("awaiting_slot", ""), EMPTY_ANSWERED)

        self._mock.side_effect = _se
        return self


async def _run(state: dict) -> dict:
    return await VerificationAgent.from_state(state).execute(state)


def _msg(role: str, text: str) -> dict:
    return {"role": role, "content": text}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_extraction(monkeypatch) -> _MockHelper:
    fake_llm = MagicMock()
    monkeypatch.setattr("agent.agents.verification.agent.get_extraction_llm", lambda: fake_llm)
    raw = AsyncMock()
    monkeypatch.setattr("agent.agents.verification.agent.extract_verification_decision", raw)
    helper = _MockHelper(raw)
    helper.set_slot_answers()
    return helper


@pytest.fixture
def mock_sf(monkeypatch) -> MagicMock:
    tool = MagicMock()
    tool.ainvoke = AsyncMock(return_value=VERIFIED_MEMBER)
    monkeypatch.setattr("agent.storage.tools.lookup_member", tool)
    return tool


@pytest.fixture
def mock_recovery(monkeypatch) -> AsyncMock:
    async def _fn(*, slot_name, attempt, guard, last_messages, **kwargs):
        return f"[RECOVERY:{slot_name}:attempt{attempt}:guard{guard}]"

    mock = AsyncMock(side_effect=_fn)
    monkeypatch.setattr("agent.llm.response_generator.generate_recovery_message", mock)
    return mock


@pytest.fixture(autouse=True)
def _mocks(mock_extraction, mock_sf, mock_recovery):
    pass


# ---------------------------------------------------------------------------
# SECTION 1 — Happy path (marker: happy)
# ---------------------------------------------------------------------------


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_provider_step_by_step(mock_extraction) -> None:
    mock_extraction.set_slot_answers()
    state = make_state()
    result = await _run(state)
    assert is_ask(result), "Turn 0 should ask for name"

    for turn, user_text in enumerate(["Emily", "Carter", "M907503", "04/12/1988", "plan holder"], 1):
        state = advance(state, result, user_text)
        result = await _run(state)
        assert not is_escalation(result), f"Unexpected escalation on turn {turn}"

    assert is_complete(result)


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_claims_step_by_step(mock_extraction) -> None:
    mock_extraction.set_slot_answers()
    state = make_state(call_intent="claim_services")
    result = await _run(state)
    assert not is_escalation(result)

    for user_text in ["Emily", "Carter", "M907503", "04/12/1988"]:
        state = advance(state, result, user_text)
        result = await _run(state)
        assert not is_escalation(result)

    state = advance(state, result, "yes")
    result = await _run(state)
    assert is_complete(result)
    assert result.get("phone_confirmed") is True


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_claims_phone_no(mock_extraction) -> None:
    mock_extraction.set_single(
        WorkerResult(
            extracted={"phone_confirmed": "false"},
            event_type=EventType.ANSWERED,
            guard=GuardType.NONE,
            guard_confidence=0.0,
        )
    )
    state = make_state(
        call_intent="claim_services",
        first_name="Emily",
        last_name="Carter",
        member_id="M907503",
        dob="04/12/1988",
        awaiting_slot="phone_confirmation",
        messages=[_msg("assistant", "Is 617-555-4101 your number?"), _msg("user", "no")],
    )
    result = await _run(state)
    assert is_complete(result) or not is_escalation(result)
    assert result.get("phone_update_requested") is True or is_complete(result)


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_reentry_already_verified(mock_extraction) -> None:
    state = make_verified_state()
    result = await _run(state)
    assert is_complete(result)
    mock_extraction._mock.assert_not_called()


# ---------------------------------------------------------------------------
# SECTION 2 — Multi-slot extraction (marker: multi_slot)
# ---------------------------------------------------------------------------


@pytest.mark.multi_slot
@pytest.mark.asyncio
async def test_all_four_slots_one_turn(mock_extraction) -> None:
    mock_extraction.set_single(
        WorkerResult(
            extracted={
                "first_name": "Emily",
                "last_name": "Carter",
                "member_id": "M907503",
                "dob": "04/12/1988",
            },
            event_type=EventType.ANSWERED,
            guard=GuardType.NONE,
            guard_confidence=0.0,
        )
    )
    state = make_state(
        awaiting_slot="first_name",
        messages=[
            _msg("assistant", "Please provide your details."),
            _msg("user", "Emily Carter M907503 04/12/1988"),
        ],
    )
    result = await _run(state)
    assert not is_escalation(result)
    assert get_awaiting(result) in ("relationship", "phone_confirmation", "phone_confirmed") or is_complete(
        result
    )


@pytest.mark.multi_slot
@pytest.mark.asyncio
async def test_first_last_one_turn(mock_extraction) -> None:
    mock_extraction.set_single(
        WorkerResult(
            extracted={"first_name": "Emily", "last_name": "Carter"},
            event_type=EventType.ANSWERED,
            guard=GuardType.NONE,
            guard_confidence=0.0,
        )
    )
    state = make_state(awaiting_slot="first_name", messages=[_msg("user", "Emily Carter")])
    result = await _run(state)
    assert not is_escalation(result)
    assert result.get("first_name") == "Emily"
    assert get_awaiting(result) in ("last_name", "member_id")


@pytest.mark.multi_slot
@pytest.mark.asyncio
async def test_bonus_dob_saved_when_member_id_fails(mock_extraction) -> None:
    mock_extraction.set_single(
        WorkerResult(
            extracted={"dob": "04/12/1988"},
            event_type=EventType.ANSWERED,
            guard=GuardType.NONE,
            guard_confidence=0.0,
        )
    )
    state = make_state(
        first_name="Emily",
        last_name="Carter",
        member_id="",
        dob="",
        awaiting_slot="member_id",
        messages=[
            _msg("assistant", "May I have your member ID?"),
            _msg("user", "April twelfth nineteen eighty eight"),
        ],
    )
    result = await _run(state)
    assert is_ask(result)
    assert get_awaiting(result) == "member_id"
    assert result.get("dob") == "04/12/1988", "Bonus DOB must be saved"
    assert get_attempt(result, "member_id") == 1
    assert get_attempt(result, "dob") == 0


@pytest.mark.multi_slot
@pytest.mark.asyncio
async def test_bonus_dob_not_reasked_after_member_id_provided(mock_extraction) -> None:
    mock_extraction.set_responses(
        [
            WorkerResult(
                extracted={"dob": "04/12/1988"},
                event_type=EventType.ANSWERED,
                guard=GuardType.NONE,
                guard_confidence=0.0,
            ),
            WorkerResult(
                extracted={"member_id": "M907503"},
                event_type=EventType.ANSWERED,
                guard=GuardType.NONE,
                guard_confidence=0.0,
            ),
        ]
    )
    state = make_state(
        first_name="Emily",
        last_name="Carter",
        member_id="",
        dob="",
        awaiting_slot="member_id",
        messages=[_msg("assistant", "Member ID?"), _msg("user", "April twelfth 1988")],
    )
    result1 = await _run(state)
    assert result1.get("dob") == "04/12/1988"

    state2 = advance(state, result1, "M907503")
    result2 = await _run(state2)
    assert result2.get("member_id") == "M907503"
    assert result2.get("dob") == "04/12/1988"
    assert get_awaiting(result2) != "dob", "Pipeline must not re-ask for dob"


# ---------------------------------------------------------------------------
# SECTION 3 — Slot retry exhaustion (marker: slot_retry)
# ---------------------------------------------------------------------------


def _exhausted_state(slot: str, **extra) -> dict:
    # attempt_count=1: one more slot_fail → count 1→2 >= MAX_SLOT_ATTEMPTS(2)
    # → is_exhausted()=True → ESCALATE
    return make_state(
        awaiting_slot=slot,
        slot_attempts={slot: {"attempt_count": 1, "confirmed": False, "last_value": None}},
        messages=[_msg("assistant", f"What is your {slot}?"), _msg("user", "???")],
        **extra,
    )


@pytest.mark.slot_retry
@pytest.mark.asyncio
async def test_exhaust_first_name(mock_extraction) -> None:
    mock_extraction.set_single(EMPTY_ANSWERED)
    assert is_escalation(await _run(_exhausted_state("first_name")))


@pytest.mark.slot_retry
@pytest.mark.asyncio
async def test_exhaust_last_name(mock_extraction) -> None:
    mock_extraction.set_single(EMPTY_ANSWERED)
    assert is_escalation(await _run(_exhausted_state("last_name", first_name="Emily")))


@pytest.mark.slot_retry
@pytest.mark.asyncio
async def test_exhaust_member_id(mock_extraction) -> None:
    mock_extraction.set_single(EMPTY_ANSWERED)
    assert is_escalation(await _run(_exhausted_state("member_id", first_name="Emily", last_name="Carter")))


@pytest.mark.slot_retry
@pytest.mark.asyncio
async def test_exhaust_dob(mock_extraction) -> None:
    mock_extraction.set_single(EMPTY_ANSWERED)
    assert is_escalation(
        await _run(_exhausted_state("dob", first_name="Emily", last_name="Carter", member_id="M907503"))
    )


@pytest.mark.slot_retry
@pytest.mark.asyncio
async def test_exhaust_relationship(mock_extraction) -> None:
    mock_extraction.set_single(EMPTY_ANSWERED)
    assert is_escalation(
        await _run(
            _exhausted_state(
                "relationship",
                first_name="Emily",
                last_name="Carter",
                member_id="M907503",
                dob="04/12/1988",
                member_status_verify=True,
            )
        )
    )


@pytest.mark.slot_retry
@pytest.mark.asyncio
async def test_one_failure_retries_not_escalates(mock_extraction) -> None:
    """
    With MAX_SLOT_ATTEMPTS=2, the first failure must produce a retry message,
    not an escalation. Escalation only fires on the second failure.
    """
    mock_extraction.set_single(EMPTY_ANSWERED)
    state = make_state(
        awaiting_slot="member_id",
        slot_attempts={"member_id": {"attempt_count": 0, "confirmed": False, "last_value": None}},
        messages=[_msg("assistant", "Member ID?"), _msg("user", "???")],
        first_name="Emily",
        last_name="Carter",
    )
    result = await _run(state)
    assert is_ask(result), "First failure must produce a retry ask"
    assert not is_escalation(result), "First failure must NOT escalate"
    assert get_attempt(result, "member_id") == 1, "attempt_count must be 1 after first failure"


@pytest.mark.slot_retry
@pytest.mark.asyncio
async def test_second_failure_escalates(mock_extraction) -> None:
    """
    The second failure must escalate.
    attempt_count=1 (one failure already recorded) + one more EMPTY_ANSWERED
    → count 1→2 >= MAX(2) → ESCALATE.
    """
    mock_extraction.set_single(EMPTY_ANSWERED)
    state = make_state(
        awaiting_slot="member_id",
        slot_attempts={"member_id": {"attempt_count": 1, "confirmed": False, "last_value": None}},
        messages=[_msg("assistant", "Member ID?"), _msg("user", "???")],
        first_name="Emily",
        last_name="Carter",
    )
    result = await _run(state)
    assert is_escalation(result), "Second failure must escalate"


@pytest.mark.slot_retry
@pytest.mark.asyncio
async def test_ambiguous_then_failure_escalates(mock_extraction) -> None:
    """
    Turn 1: CLARIFY (first AMBIGUOUS, no penalty) — attempt_count stays 0
    Turn 2: ANSWERED failure — attempt_count 0→1, not exhausted → retry
    Turn 3: ANSWERED failure — attempt_count 1→2 >= MAX(2) → ESCALATE
    """
    # Turn 1: first AMBIGUOUS — clarify, no penalty
    mock_extraction.set_single(make_ambiguous())
    state = make_state(
        awaiting_slot="member_id",
        slot_attempts={"member_id": {"attempt_count": 0, "confirmed": False, "last_value": None}},
        messages=[_msg("assistant", "Member ID?"), _msg("user", "no no wait")],
        first_name="Emily",
        last_name="Carter",
    )
    result1 = await _run(state)
    assert is_ask(result1)
    assert not is_escalation(result1)
    assert get_attempt(result1, "member_id") == 0
    assert get_ambiguous(result1, "member_id") == 1

    # Turn 2: ANSWERED failure — attempt_count 0→1
    mock_extraction.set_single(EMPTY_ANSWERED)
    state2 = advance(state, result1, "I don't know it")
    result2 = await _run(state2)
    assert is_ask(result2)
    assert not is_escalation(result2)
    assert get_attempt(result2, "member_id") == 1

    # Turn 3: ANSWERED failure — attempt_count 1→2 >= MAX(2) → ESCALATE
    mock_extraction.set_single(EMPTY_ANSWERED)
    state3 = advance(state2, result2, "still don't know")
    result3 = await _run(state3)
    assert is_escalation(result3), "Second genuine failure must escalate"


@pytest.mark.slot_retry
@pytest.mark.asyncio
async def test_lookup_one_restart_then_escalate(mock_sf, mock_extraction) -> None:
    """
    MAX_LOOKUP_ATTEMPTS=2: exactly one identity restart, then ESCALATE.
    Total SF calls = 2.
    """
    mock_sf.ainvoke = AsyncMock(return_value={"verified": False})
    state = make_state(
        first_name="Emily",
        last_name="Carter",
        member_id="M907503",
        dob="04/12/1988",
    )
    # First lookup fails → restart (ask for first_name again)
    result1 = await _run(state)
    assert is_ask(result1), "First lookup fail must restart collection"
    assert not is_escalation(result1)
    assert result1.get("first_name") == "", "Identity slots must be cleared on restart"

    # Second lookup fails → ESCALATE
    state2 = make_state(
        first_name="Emily",
        last_name="Carter",
        member_id="M907503",
        dob="04/12/1988",
        slot_attempts=result1.get("slot_attempts", {}),
    )
    result2 = await _run(state2)
    assert is_escalation(result2), "Second lookup fail must escalate"
    assert mock_sf.ainvoke.call_count == 2, f"Expected exactly 2 SF calls, got {mock_sf.ainvoke.call_count}"


# ---------------------------------------------------------------------------
# SECTION 4 — Guard triggers (marker: guards)
# ---------------------------------------------------------------------------


@pytest.mark.guards
@pytest.mark.asyncio
async def test_guard_transfer_request(mock_extraction) -> None:
    mock_extraction.set_single(make_guard(GuardType.TRANSFER_REQUEST))
    assert is_escalation(
        await _run(make_state(awaiting_slot="member_id", messages=[_msg("user", "transfer me")]))
    )


@pytest.mark.guards
@pytest.mark.asyncio
async def test_guard_abuse(mock_extraction) -> None:
    mock_extraction.set_single(make_guard(GuardType.ABUSE))
    assert is_escalation(
        await _run(make_state(awaiting_slot="member_id", messages=[_msg("user", "you're useless")]))
    )


@pytest.mark.guards
@pytest.mark.asyncio
async def test_guard_self_harm(mock_extraction) -> None:
    mock_extraction.set_single(make_guard(GuardType.SELF_HARM))
    assert is_escalation(
        await _run(make_state(awaiting_slot="first_name", messages=[_msg("user", "I want to end my life")]))
    )


@pytest.mark.guards
@pytest.mark.asyncio
async def test_guard_interruption(mock_extraction) -> None:
    mock_extraction.set_single(make_guard(GuardType.INTERRUPTION))
    assert is_ask(
        await _run(
            make_state(awaiting_slot="member_id", messages=[_msg("user", "can you help with something else")])
        )
    )


@pytest.mark.guards
@pytest.mark.asyncio
async def test_guard_offtopic_agent_redirects(mock_extraction) -> None:
    mock_extraction.set_single(make_guard(GuardType.OFFTOPIC_AGENT))
    state = make_state(
        first_name="Emily",
        last_name="Carter",
        awaiting_slot="member_id",
        messages=[_msg("user", "tell me about my benefits")],
    )
    result = await _run(state)
    assert is_ask(result)
    assert result.get("member_id", "") == ""


@pytest.mark.guards
@pytest.mark.asyncio
async def test_offtopic_global_escalates_at_max(mock_extraction) -> None:
    # offtopic_global_count=1: guard increments to 2 >= MAX_SLOT_ATTEMPTS(2) → ESCALATE
    mock_extraction.set_single(make_guard(GuardType.OFFTOPIC_GLOBAL))
    assert is_escalation(
        await _run(
            make_state(
                awaiting_slot="member_id", offtopic_global_count=1, messages=[_msg("user", "sports question")]
            )
        )
    )


# ---------------------------------------------------------------------------
# SECTION 5 — Corrections (marker: corrections)
# ---------------------------------------------------------------------------


@pytest.mark.corrections
@pytest.mark.asyncio
async def test_correction_last_name_while_awaiting_member_id(mock_extraction) -> None:
    mock_extraction.set_single(make_correction("last_name", "Johnson"))
    state = make_state(
        first_name="Emily",
        last_name="Carter",
        awaiting_slot="member_id",
        messages=[_msg("assistant", "Member ID?"), _msg("user", "no my last name is Johnson")],
    )
    result = await _run(state)
    assert is_ask(result)
    assert result.get("last_name") == "Johnson"
    assert get_attempt(result, "member_id") == 0
    assert get_awaiting(result) == "member_id"


@pytest.mark.corrections
@pytest.mark.asyncio
async def test_correction_first_name_clears_last_name(mock_extraction) -> None:
    mock_extraction.set_single(make_correction("first_name", "James"))
    state = make_state(
        first_name="John",
        last_name="Carter",
        awaiting_slot="member_id",
        messages=[_msg("assistant", "Member ID?"), _msg("user", "actually my name is James")],
    )
    result = await _run(state)
    assert result.get("first_name") == "James"
    assert result.get("last_name") in ("", None)
    assert get_attempt(result, "member_id") == 0


@pytest.mark.corrections
@pytest.mark.asyncio
async def test_correction_member_id_clears_dob(mock_extraction) -> None:
    mock_extraction.set_single(make_correction("member_id", "M907503"))
    state = make_state(
        first_name="Emily",
        last_name="Carter",
        member_id="U0000001",
        dob="01/01/1980",
        messages=[_msg("assistant", "Verifying."), _msg("user", "sorry wrong member ID")],
    )
    result = await _run(state)
    assert result.get("member_id") == "M907503"
    assert result.get("dob") in ("", None)


@pytest.mark.corrections
@pytest.mark.asyncio
async def test_locked_slot_not_acknowledged(mock_extraction) -> None:
    mock_extraction.set_single(EMPTY_ANSWERED)
    state = make_state(
        first_name="Emily",
        last_name="Carter",
        member_id="M907503",
        awaiting_slot="dob",
        messages=[_msg("assistant", "Date of birth?"), _msg("user", "that zip code is wrong")],
    )
    result = await _run(state)
    assert get_attempt(result, "dob") == 1
    assert "zip" not in get_response(result).lower()


@pytest.mark.corrections
@pytest.mark.asyncio
async def test_corrected_event_empty_corrections_treated_as_answered(mock_extraction) -> None:
    mock_extraction.set_single(
        WorkerResult(
            corrections={}, event_type=EventType.CORRECTED, guard=GuardType.NONE, guard_confidence=0.0
        )
    )
    state = make_state(
        first_name="Emily",
        last_name="Carter",
        awaiting_slot="member_id",
        messages=[_msg("assistant", "Member ID?"), _msg("user", "uh I'm not sure")],
    )
    result = await _run(state)
    assert is_ask(result)
    assert get_attempt(result, "member_id") == 1


# ---------------------------------------------------------------------------
# SECTION 6 — Ambiguous events (marker: ambiguous)
# ---------------------------------------------------------------------------


@pytest.mark.ambiguous
@pytest.mark.asyncio
async def test_ambiguous_first_turn_no_penalty(mock_extraction) -> None:
    mock_extraction.set_single(make_ambiguous())
    state = make_state(
        first_name="Emily",
        last_name="Carter",
        awaiting_slot="member_id",
        messages=[_msg("assistant", "Member ID?"), _msg("user", "no no my last name")],
    )
    result = await _run(state)
    assert is_ask(result)
    assert get_attempt(result, "member_id") == 0
    assert get_ambiguous(result, "member_id") == 1


@pytest.mark.ambiguous
@pytest.mark.asyncio
async def test_ambiguous_second_turn_counts_as_failure(mock_extraction) -> None:
    mock_extraction.set_single(make_ambiguous())
    state = make_state(
        first_name="Emily",
        last_name="Carter",
        awaiting_slot="member_id",
        ambiguous_counts={"member_id": 1},
        messages=[_msg("assistant", "Member ID?"), _msg("user", "no no my last name")],
    )
    result = await _run(state)
    assert is_ask(result)
    assert get_attempt(result, "member_id") == 1
    assert get_ambiguous(result, "member_id") == 0


@pytest.mark.ambiguous
@pytest.mark.asyncio
async def test_ambiguous_then_exhaust(mock_extraction) -> None:
    mock_extraction.set_single(make_ambiguous())
    state = make_state(
        first_name="Emily",
        last_name="Carter",
        awaiting_slot="member_id",
        ambiguous_counts={"member_id": 1},
        slot_attempts={"member_id": {"attempt_count": 2, "confirmed": False, "last_value": None}},
        messages=[_msg("assistant", "Member ID?"), _msg("user", "no no my last name")],
    )
    assert is_escalation(await _run(state))


# ---------------------------------------------------------------------------
# SECTION 7 — Lookup failures (marker: lookup)
# ---------------------------------------------------------------------------


@pytest.mark.lookup
@pytest.mark.asyncio
async def test_lookup_three_failures_escalate(mock_sf, mock_extraction) -> None:
    # MAX_LOOKUP_ATTEMPTS=2: first failure restarts, second failure escalates.
    # Total of 2 SF calls before escalation.
    mock_sf.ainvoke = AsyncMock(return_value={"verified": False})
    state = make_state(first_name="Emily", last_name="Carter", member_id="M907503", dob="04/12/1988")

    # Only 1 restart before escalation (was 2 with MAX=3)
    for _ in range(1):
        result = await _run(state)
        assert is_ask(result)
        state = make_state(
            first_name="Emily",
            last_name="Carter",
            member_id="M907503",
            dob="04/12/1988",
            slot_attempts=result.get("slot_attempts", {}),
        )

    assert is_escalation(await _run(state))


@pytest.mark.lookup
@pytest.mark.asyncio
async def test_lookup_fail_then_success(mock_sf, mock_extraction) -> None:
    call_count = [0]

    async def _sf(args):
        call_count[0] += 1
        return {"verified": False} if call_count[0] == 1 else VERIFIED_MEMBER

    mock_sf.ainvoke = _sf

    state = make_state(first_name="Emily", last_name="Carter", member_id="M907503", dob="04/12/1988")
    result1 = await _run(state)
    assert is_ask(result1)

    state2 = make_state(
        first_name="Emily",
        last_name="Carter",
        member_id="M907503",
        dob="04/12/1988",
        slot_attempts=result1.get("slot_attempts", {}),
    )
    assert not is_escalation(await _run(state2))
    assert call_count[0] == 2


@pytest.mark.lookup
@pytest.mark.asyncio
async def test_lookup_restart_resets_slot_attempts(mock_sf, mock_extraction) -> None:
    mock_sf.ainvoke = AsyncMock(return_value={"verified": False})
    state = make_state(
        first_name="Emily",
        last_name="Carter",
        member_id="M907503",
        dob="04/12/1988",
        slot_attempts={
            "first_name": {"attempt_count": 2, "confirmed": False, "last_value": None},
            "last_name": {"attempt_count": 2, "confirmed": False, "last_value": None},
        },
    )
    result = await _run(state)
    assert is_ask(result)
    assert get_attempt(result, "first_name") == 0
    assert get_attempt(result, "last_name") == 0
    assert get_attempt(result, "member_id") == 0
    assert get_attempt(result, "dob") == 0
    assert result.get("first_name") == ""


# ---------------------------------------------------------------------------
# SECTION 8 — Post-lookup slot collection (marker: post_lookup)
# ---------------------------------------------------------------------------


@pytest.mark.post_lookup
@pytest.mark.asyncio
async def test_phone_confirmed_yes(mock_extraction) -> None:
    mock_extraction.set_single(
        WorkerResult(
            extracted={"phone_confirmed": "true"},
            event_type=EventType.ANSWERED,
            guard=GuardType.NONE,
            guard_confidence=0.0,
        )
    )
    state = make_verified_state(
        call_intent="claim_services",
        awaiting_slot="phone_confirmed",
        messages=[_msg("assistant", "Is this your number?"), _msg("user", "yes")],
    )
    result = await _run(state)
    assert is_complete(result)
    assert result.get("phone_confirmed") is True


@pytest.mark.post_lookup
@pytest.mark.asyncio
async def test_phone_confirmed_no(mock_extraction) -> None:
    mock_extraction.set_single(
        WorkerResult(
            extracted={"phone_confirmed": "false"},
            event_type=EventType.ANSWERED,
            guard=GuardType.NONE,
            guard_confidence=0.0,
        )
    )
    state = make_verified_state(
        call_intent="claim_services",
        awaiting_slot="phone_confirmed",
        messages=[_msg("assistant", "Is this your number?"), _msg("user", "no")],
    )
    result = await _run(state)
    assert is_complete(result)
    assert result.get("phone_update_requested") is True


@pytest.mark.post_lookup
@pytest.mark.asyncio
async def test_relationship_plan_holder(mock_extraction) -> None:
    mock_extraction.set_single(
        WorkerResult(
            extracted={"relationship": "plan holder"},
            event_type=EventType.ANSWERED,
            guard=GuardType.NONE,
            guard_confidence=0.0,
        )
    )
    state = make_verified_state(
        awaiting_slot="relationship",
        messages=[_msg("assistant", "Relationship?"), _msg("user", "I'm the plan holder")],
    )
    assert not is_escalation(await _run(state))


@pytest.mark.post_lookup
@pytest.mark.asyncio
async def test_relationship_exhausted(mock_extraction) -> None:
    mock_extraction.set_single(EMPTY_ANSWERED)
    state = make_verified_state(
        awaiting_slot="relationship",
        slot_attempts={"relationship": {"attempt_count": 2, "confirmed": False, "last_value": None}},
        messages=[_msg("assistant", "Relationship?"), _msg("user", "I have no idea")],
    )
    assert is_escalation(await _run(state))


# ---------------------------------------------------------------------------
# SECTION 9 — Response content checks (marker: response_check)
# ---------------------------------------------------------------------------


@pytest.mark.response_check
@pytest.mark.asyncio
async def test_response_first_ask(mock_extraction) -> None:
    mock_extraction.set_single(EMPTY_ANSWERED)
    result = await _run(make_state())
    response = get_response(result)
    assert len(response) > 0
    assert "name" in response.lower()


@pytest.mark.response_check
@pytest.mark.asyncio
async def test_response_retry_member_id(mock_extraction) -> None:
    mock_extraction.set_single(EMPTY_ANSWERED)
    state = make_state(
        first_name="Emily",
        last_name="Carter",
        awaiting_slot="member_id",
        slot_attempts={"member_id": {"attempt_count": 0, "confirmed": False, "last_value": None}},
        messages=[_msg("assistant", "Member ID?"), _msg("user", "???")],
    )
    assert "[RECOVERY:member_id:attempt1:guardRETRY]" in get_response(await _run(state))


@pytest.mark.response_check
@pytest.mark.asyncio
async def test_response_correction_ack(mock_extraction) -> None:
    mock_extraction.set_single(make_correction("last_name", "Johnson"))
    state = make_state(
        first_name="Emily",
        last_name="Carter",
        awaiting_slot="member_id",
        messages=[_msg("assistant", "Member ID?"), _msg("user", "no my last name is Johnson")],
    )
    assert "[RECOVERY:member_id:attempt0:guardCORRECTION]" in get_response(await _run(state))


# ---------------------------------------------------------------------------
# SECTION 10 — Regression tests (marker: regression)
# ---------------------------------------------------------------------------


@pytest.mark.regression
@pytest.mark.asyncio
async def test_recovery_does_not_pivot_to_alternative_slot(mock_sf, mock_extraction, monkeypatch) -> None:
    captured: dict = {}

    async def _capture(*, slot_name, attempt, guard, last_messages, **kwargs):
        captured["slot_name"] = slot_name
        return f"[RECOVERY:{slot_name}:attempt{attempt}:guard{guard}]"

    monkeypatch.setattr(
        "agent.llm.response_generator.generate_recovery_message", AsyncMock(side_effect=_capture)
    )
    mock_extraction.set_single(EMPTY_ANSWERED)

    state = make_state(
        first_name="Emily",
        last_name="Carter",
        awaiting_slot="member_id",
        slot_attempts={"member_id": {"attempt_count": 0, "confirmed": False, "last_value": None}},
        messages=[_msg("assistant", "Member ID?"), _msg("user", "I don't remember")],
    )
    result = await _run(state)
    assert get_awaiting(result) == "member_id"
    assert "date of birth" not in get_response(result).lower()


@pytest.mark.regression
@pytest.mark.asyncio
async def test_spelling_confirmation_not_stored_as_last_name(mock_sf, mock_extraction) -> None:
    mock_extraction.set_single(
        WorkerResult(
            extracted={"first_name": "Emma"},
            event_type=EventType.ANSWERED,
            guard=GuardType.NONE,
            guard_confidence=0.0,
        )
    )
    state = make_state(
        awaiting_slot="first_name",
        messages=[_msg("assistant", "First name?"), _msg("user", "its emma , E M M A")],
    )
    result = await _run(state)
    assert result.get("first_name") == "Emma"
    assert result.get("last_name", "") == ""


@pytest.mark.regression
@pytest.mark.asyncio
async def test_phone_confirmed_recovery_no_false_pivot_warning(
    mock_sf, mock_extraction, monkeypatch, caplog
) -> None:
    """phone_confirmed mentioning 'phone' must NOT trigger a pivot hallucination WARNING."""
    import logging

    async def _phone_recovery(*, slot_name, attempt, guard, last_messages, **kwargs):
        return "Could you confirm whether that phone number is still correct?"

    monkeypatch.setattr(
        "agent.llm.response_generator.generate_recovery_message",
        AsyncMock(side_effect=_phone_recovery),
    )
    mock_extraction.set_single(EMPTY_ANSWERED)

    state = make_verified_state(
        call_intent="claim_services",
        awaiting_slot="phone_confirmed",
        messages=[_msg("assistant", "Is 617-555-4101 still your number?"), _msg("user", "um")],
    )
    with caplog.at_level(logging.WARNING):
        await _run(state)

    assert not any("pivot hallucination" in r.message for r in caplog.records), (
        "phone_confirmed mentioning 'phone' must NOT be logged as a pivot hallucination"
    )


@pytest.mark.regression
@pytest.mark.asyncio
async def test_dob_reextracted_after_lookup_restart(mock_sf, monkeypatch) -> None:
    DOB_MSG = "April twelfth nineteen eighty eight"
    messages = [_msg("user", f"msg {i}") for i in range(8)]
    messages += [
        _msg("user", DOB_MSG),
        _msg("assistant", "Let me verify that"),
        _msg("assistant", "Let me start fresh — what's your first name?"),
        _msg("user", "Emily"),
    ]

    async def _extraction(llm, system_prompt, *, awaiting_slot, recent_messages=None, **kwargs):
        if awaiting_slot == "dob":
            for m in recent_messages or []:
                if "twelfth" in (m.get("content") or "").lower():
                    return WorkerResult(
                        extracted={"dob": "04/12/1988"},
                        event_type=EventType.ANSWERED,
                        guard=GuardType.NONE,
                        guard_confidence=0.0,
                    )
        return SLOT_ANSWERS.get(awaiting_slot, EMPTY_ANSWERED)

    monkeypatch.setattr("agent.agents.verification.agent.get_extraction_llm", lambda: MagicMock())
    monkeypatch.setattr(
        "agent.agents.verification.agent.extract_verification_decision", AsyncMock(side_effect=_extraction)
    )

    state = make_state(
        first_name="Emily",
        last_name="Carter",
        member_id="M907503",
        dob="",
        awaiting_slot="dob",
        verification_restart_index=10,
        messages=messages,
    )
    result = await _run(state)
    assert not is_escalation(result)
    assert result.get("dob") == "04/12/1988" or get_awaiting(result) != "dob"
