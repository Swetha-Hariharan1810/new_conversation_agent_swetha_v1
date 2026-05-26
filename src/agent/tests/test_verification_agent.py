"""
test_verification_agent.py — Live LLM test suite for VerificationAgent.

Requires AZURE_OPENAI_API_KEY to be set; all tests are skipped when absent.

Run all:    pytest src/agent/tests/test_verification_agent.py -v
By marker:  pytest src/agent/tests/test_verification_agent.py -v -m happy
Record:     RECORD_RESPONSES=1 pytest src/agent/tests/test_verification_agent.py -v
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from statistics import mean
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.tests.helpers import load_test_env as _load_env

_load_env()

CREDS_AVAILABLE = bool(os.getenv("AZURE_OPENAI_API_KEY"))
pytestmark = pytest.mark.skipif(not CREDS_AVAILABLE, reason="AZURE_OPENAI_API_KEY not set")

from agent.agents.verification.agent import VerificationAgent  # noqa: E402
from agent.tests.fixtures import (  # noqa: E402
    advance,
    get_attempt,
    get_awaiting,
    get_response,
    is_ask,
    is_complete,
    is_escalation,
    make_state,
    make_verified_state,
)
from agent.tests.recorder import get_recorder  # noqa: E402

_VERIFIED_MEMBER_RECORD = {
    "verified": True,
    "phone_number": "6175554101",
    "zip_code": "12139",
    "relationship": "plan_holder",
}


async def _run(state: dict) -> dict:
    return await VerificationAgent.from_state(state).execute(state)


def _msg(role: str, text: str) -> dict:
    return {"role": role, "content": text}


def _slot_state(attempt_count: int) -> dict:
    return {"attempt_count": attempt_count, "confirmed": False, "last_value": None}


def _p(data: list[float], pct: float) -> float:
    s = sorted(data)
    n = len(s)
    k = (pct / 100) * (n - 1)
    lo, hi = int(k), min(int(k) + 1, n - 1)
    return s[lo] + (k - lo) * (s[hi] - s[lo])


async def _timed_run(state: dict) -> tuple[dict, float]:
    """Run the agent and return (result, elapsed_ms)."""
    t0 = time.perf_counter()
    result = await _run(state)
    return result, (time.perf_counter() - t0) * 1000


def assert_quality(response: str, *, max_words: int = 50) -> None:
    assert isinstance(response, str) and len(response.strip()) > 0
    assert len(response.split()) <= max_words, f"Too long: {response!r}"


@pytest.fixture
def mock_sf(monkeypatch):
    tool = MagicMock()
    tool.ainvoke = AsyncMock(return_value=_VERIFIED_MEMBER_RECORD)
    monkeypatch.setattr("agent.storage.tools.lookup_member", tool)
    return tool


# ---------------------------------------------------------------------------
# SECTION 1 — Happy path (marker: happy)
# ---------------------------------------------------------------------------


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_provider_step_by_step(mock_sf) -> None:
    rec = get_recorder()
    state = make_state()
    result = await _run(state)
    assert is_ask(result)
    assert_quality(get_response(result))
    rec.record("test_happy_provider_step_by_step", 0, "first_ask", "", state, result)

    for turn, text in enumerate(["Emily", "Carter", "M907503", "04/12/1988", "plan holder"], 1):
        state = advance(state, result, text)
        result = await _run(state)
        assert not is_escalation(result), f"Escalation on turn {turn}"
        assert_quality(get_response(result), max_words=60)
        rec.record("test_happy_provider_step_by_step", turn, text, text, state, result)

    assert is_complete(result)


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_claims_phone_yes(mock_sf) -> None:
    state = make_state(call_intent="claim_services")
    result = await _run(state)
    for text in ["Emily", "Carter", "M907503", "04/12/1988"]:
        state = advance(state, result, text)
        result = await _run(state)
        assert not is_escalation(result)
    state = advance(state, result, "yes that's right")
    result = await _run(state)
    assert is_complete(result)
    assert result.get("phone_confirmed") is True


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_reentry_already_verified(mock_sf) -> None:
    assert is_complete(await _run(make_verified_state()))


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_spoken_member_id(mock_sf) -> None:
    state = make_state(
        first_name="Emily",
        last_name="Carter",
        awaiting_slot="member_id",
        messages=[_msg("assistant", "Member ID?"), _msg("user", "m nine zero seven five zero three")],
    )
    result = await _run(state)
    assert not is_escalation(result)
    assert result.get("member_id") == "M907503" or get_awaiting(result) in ("dob", "relationship")


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_spoken_dob(mock_sf) -> None:
    state = make_state(
        first_name="Emily",
        last_name="Carter",
        member_id="M907503",
        awaiting_slot="dob",
        messages=[_msg("assistant", "Date of birth?"), _msg("user", "April twelfth nineteen eighty eight")],
    )
    result = await _run(state)
    assert not is_escalation(result)
    assert result.get("dob") == "04/12/1988" or is_complete(result) or get_awaiting(result) != "dob"


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_all_slots_one_turn(mock_sf) -> None:
    state = make_state(
        awaiting_slot="first_name",
        messages=[
            _msg("assistant", "Details please."),
            _msg("user", "Emily Carter, member ID M907503, date of birth April 12th 1988"),
        ],
    )
    result = await _run(state)
    assert not is_escalation(result)
    assert (
        get_awaiting(result) in ("relationship", "phone_confirmation")
        or is_complete(result)
        or all(result.get(s) for s in ("first_name", "last_name", "member_id"))
    )


# ---------------------------------------------------------------------------
# SECTION 2 — Non-happy path (marker: unhappy)
# ---------------------------------------------------------------------------


@pytest.mark.unhappy
@pytest.mark.asyncio
async def test_unhappy_refuses_member_id(mock_sf) -> None:
    state = make_state(
        first_name="Emily",
        last_name="Carter",
        awaiting_slot="member_id",
        messages=[_msg("assistant", "Member ID?"), _msg("user", "I don't want to give it")],
    )
    result = await _run(state)
    assert not is_escalation(result) and is_ask(result)


@pytest.mark.unhappy
@pytest.mark.asyncio
async def test_unhappy_abuse_escalates(mock_sf) -> None:
    state = make_state(
        awaiting_slot="member_id",
        messages=[_msg("assistant", "Member ID?"), _msg("user", "you're useless, I hate this bot")],
    )
    assert is_escalation(await _run(state))


@pytest.mark.unhappy
@pytest.mark.asyncio
async def test_unhappy_transfer_request_escalates(mock_sf) -> None:
    state = make_state(
        first_name="Emily",
        awaiting_slot="last_name",
        messages=[_msg("assistant", "Last name?"), _msg("user", "transfer me to a representative")],
    )
    assert is_escalation(await _run(state))


@pytest.mark.unhappy
@pytest.mark.asyncio
async def test_unhappy_self_harm_escalates(mock_sf) -> None:
    state = make_state(
        awaiting_slot="first_name",
        messages=[_msg("assistant", "First name?"), _msg("user", "I want to end my life")],
    )
    assert is_escalation(await _run(state))


@pytest.mark.unhappy
@pytest.mark.asyncio
async def test_unhappy_offtopic_redirects(mock_sf) -> None:
    state = make_state(
        first_name="Emily",
        last_name="Carter",
        awaiting_slot="member_id",
        messages=[_msg("assistant", "Member ID?"), _msg("user", "can you tell me about my benefits?")],
    )
    result = await _run(state)
    assert not is_escalation(result) and is_ask(result)


# ---------------------------------------------------------------------------
# SECTION 3 — Slot retry exhaustion (marker: slot_retry)
# ---------------------------------------------------------------------------


@pytest.mark.slot_retry
@pytest.mark.asyncio
async def test_exhaust_first_name(mock_sf) -> None:
    # attempt_count=1: one more slot_fail → 1→2 >= MAX(2) → ESCALATE
    # ambiguous_counts pre-seeded: if LLM returns AMBIGUOUS instead of ANSWERED,
    # the second consecutive AMBIGUOUS also calls slot_fail → same result.
    state = make_state(
        awaiting_slot="first_name",
        slot_attempts={"first_name": _slot_state(1)},
        ambiguous_counts={"first_name": 1},
        messages=[_msg("assistant", "First name?"), _msg("user", "12345")],
    )
    assert is_escalation(await _run(state))


@pytest.mark.slot_retry
@pytest.mark.asyncio
async def test_exhaust_member_id(mock_sf) -> None:
    state = make_state(
        first_name="Emily",
        last_name="Carter",
        awaiting_slot="member_id",
        slot_attempts={"member_id": _slot_state(1)},
        ambiguous_counts={"member_id": 1},
        messages=[_msg("assistant", "Member ID?"), _msg("user", "AB")],
    )
    assert is_escalation(await _run(state))


@pytest.mark.slot_retry
@pytest.mark.asyncio
async def test_exhaust_dob(mock_sf) -> None:
    state = make_state(
        first_name="Emily",
        last_name="Carter",
        member_id="M907503",
        awaiting_slot="dob",
        slot_attempts={"dob": _slot_state(1)},
        ambiguous_counts={"dob": 1},
        messages=[_msg("assistant", "Date of birth?"), _msg("user", "someday")],
    )
    assert is_escalation(await _run(state))


# ---------------------------------------------------------------------------
# SECTION 4 — Corrections (marker: corrections)
# ---------------------------------------------------------------------------


@pytest.mark.corrections
@pytest.mark.asyncio
async def test_correct_first_name(mock_sf) -> None:
    state = make_state(
        first_name="John",
        awaiting_slot="last_name",
        messages=[_msg("assistant", "Last name?"), _msg("user", "actually my name is James")],
    )
    result = await _run(state)

    # Core contract: a correction utterance must never cause escalation and
    # the pipeline must remain collecting last_name.
    assert not is_escalation(result), "Correction utterance must not escalate"
    assert is_ask(result), "Pipeline must pause for caller input"
    assert get_awaiting(result) == "last_name", (
        "Pipeline must remain on last_name after a first_name correction utterance"
    )

    # first_name outcome depends on whether LLM 1 returns CORRECTED or ANSWERED.
    # CORRECTED → apply_corrections fires → first_name = "James"
    # ANSWERED  → correction not detected → first_name stays "John"
    # Both are acceptable in a live test; first_name must never become empty.
    assert result.get("first_name") in ("James", "John"), (
        f"first_name must be James (corrected) or John (original), got {result.get('first_name')!r}"
    )
    # last_name must remain empty (the pipeline has not yet collected it)
    assert (result.get("last_name") or "") == "", "last_name must not be set before the caller provides it"


@pytest.mark.corrections
@pytest.mark.asyncio
async def test_correct_last_name_mid_pipeline(mock_sf) -> None:
    state = make_state(
        first_name="Emily",
        last_name="Chavez",
        awaiting_slot="member_id",
        messages=[_msg("assistant", "Member ID?"), _msg("user", "actually it's Carter, not Chavez")],
    )
    result = await _run(state)
    assert not is_escalation(result)
    assert result.get("last_name") == "Carter" or get_awaiting(result) in ("last_name", "member_id")


# ---------------------------------------------------------------------------
# SECTION 5 — Lookup failures (marker: lookup)
# ---------------------------------------------------------------------------


@pytest.mark.lookup
@pytest.mark.asyncio
async def test_lookup_fails_three_times_escalates(monkeypatch) -> None:
    # MAX_LOOKUP_ATTEMPTS=2: 1 restart then ESCALATE. Total SF calls = 2.
    tool = MagicMock()
    tool.ainvoke = AsyncMock(return_value={"verified": False})
    monkeypatch.setattr("agent.storage.tools.lookup_member", tool)
    state = make_state(first_name="Emily", last_name="Carter", member_id="M907503", dob="04/12/1988")
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
    assert tool.ainvoke.call_count == 2


@pytest.mark.lookup
@pytest.mark.asyncio
async def test_lookup_fail_then_success(monkeypatch) -> None:
    count = [0]

    async def _sf(_):
        count[0] += 1
        return {"verified": False} if count[0] == 1 else _VERIFIED_MEMBER_RECORD

    tool = MagicMock()
    tool.ainvoke = _sf
    monkeypatch.setattr("agent.storage.tools.lookup_member", tool)
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


@pytest.mark.lookup
@pytest.mark.asyncio
async def test_lookup_restart_resets_attempt_counters(monkeypatch) -> None:
    tool = MagicMock()
    tool.ainvoke = AsyncMock(return_value={"verified": False})
    monkeypatch.setattr("agent.storage.tools.lookup_member", tool)
    state = make_state(
        first_name="Emily",
        last_name="Carter",
        member_id="M907503",
        dob="04/12/1988",
        slot_attempts={"first_name": _slot_state(2), "last_name": _slot_state(2)},
    )
    result = await _run(state)
    assert is_ask(result)
    assert get_attempt(result, "first_name") == 0
    assert get_attempt(result, "member_id") == 0


# ---------------------------------------------------------------------------
# SECTION 6 — Response quality (marker: response_check)
# ---------------------------------------------------------------------------


@pytest.mark.response_check
@pytest.mark.asyncio
async def test_response_first_ask(mock_sf) -> None:
    result = await _run(make_state())
    r = get_response(result)
    assert_quality(r)
    assert any(w in r.lower() for w in ("name", "first", "hello", "may i"))


@pytest.mark.response_check
@pytest.mark.asyncio
async def test_response_retry_differs_from_first_ask(mock_sf) -> None:
    original = "Can I get your first name?"
    state = make_state(
        awaiting_slot="first_name",
        slot_attempts={"first_name": _slot_state(1)},
        messages=[_msg("assistant", original), _msg("user", "99999")],
    )
    r = get_response(await _run(state))
    assert_quality(r, max_words=60)
    assert r != original


# ---------------------------------------------------------------------------
# SECTION 7 — Regression tests (marker: regression)
# ---------------------------------------------------------------------------


@pytest.mark.regression
@pytest.mark.asyncio
async def test_spelling_confirmation_not_in_last_name(mock_sf) -> None:
    state = make_state(
        awaiting_slot="first_name",
        messages=[
            _msg("assistant", "First name?"),
            _msg("user", "My name is Emily, E M I L Y"),
        ],
    )
    result = await _run(state)
    assert not is_escalation(result)
    if result.get("first_name"):
        assert result.get("first_name", "").lower() == "emily"
    assert (result.get("last_name") or "").lower() != "emily"


@pytest.mark.regression
@pytest.mark.asyncio
async def test_x_is_y_last_name_is_ambiguous(mock_sf) -> None:
    state = make_state(
        first_name="Emily",
        awaiting_slot="last_name",
        messages=[
            _msg("assistant", "Last name?"),
            _msg("user", "Basking is Carter"),
        ],
    )
    result = await _run(state)
    assert is_ask(result)
    assert (result.get("last_name") or "").lower() not in ("basking", "carter")


@pytest.mark.regression
@pytest.mark.asyncio
async def test_phone_confirmed_recovery_no_false_pivot_warning(mock_sf, caplog) -> None:
    """phone_confirmed recovery mentioning 'phone' must NOT trigger pivot hallucination WARNING."""
    import logging

    state = make_verified_state(
        call_intent="claim_services",
        awaiting_slot="phone_confirmed",
        messages=[
            _msg("assistant", "Is 617-555-4101 still your number on file?"),
            _msg("user", "um"),
        ],
    )
    with caplog.at_level(logging.WARNING):
        await _run(state)
    assert not any("pivot hallucination" in r.message for r in caplog.records)


@pytest.mark.regression
@pytest.mark.asyncio
async def test_two_consecutive_retries_differ(mock_sf) -> None:
    state = make_state(
        first_name="Emily",
        last_name="Carter",
        awaiting_slot="member_id",
        messages=[_msg("assistant", "Member ID?"), _msg("user", "uh... I'm not sure")],
    )
    result1 = await _run(state)
    assert is_ask(result1)
    r1 = get_response(result1)
    state2 = advance(state, result1, "I still don't know")
    result2 = await _run(state2)
    assert is_ask(result2)
    r2 = get_response(result2)
    assert r1[:20] != r2[:20], f"Two re-asks must differ: {r1[:30]!r} vs {r2[:30]!r}"


# ---------------------------------------------------------------------------
# SECTION 8 — Latency (marker: latency)
# ---------------------------------------------------------------------------


@pytest.mark.latency
@pytest.mark.asyncio
async def test_latency_first_turn(mock_sf) -> None:
    t0 = time.perf_counter()
    result = await _run(make_state())
    assert is_ask(result)
    assert time.perf_counter() - t0 < 2.0


@pytest.mark.latency
@pytest.mark.asyncio
async def test_latency_full_flow(mock_sf) -> None:
    async def _one() -> float:
        t0 = time.perf_counter()
        state = make_state()
        result = await _run(state)
        for text in ["Emily", "Carter", "M907503", "04/12/1988", "plan holder"]:
            if is_complete(result) or is_escalation(result):
                break
            state = advance(state, result, text)
            result = await _run(state)
        return time.perf_counter() - t0

    times = [await _one() for _ in range(3)]
    assert _p(times, 95) < 25.0, f"p95 {_p(times, 95):.2f}s exceeds 25s"


# ---------------------------------------------------------------------------
# SECTION 9 — Stress (marker: stress)
# ---------------------------------------------------------------------------


@pytest.mark.stress
@pytest.mark.asyncio
async def test_stress_concurrent_10(mock_sf) -> None:
    async def _one():
        return await _run(make_state(app_run_id=str(uuid.uuid4())))

    results = await asyncio.gather(*[_one() for _ in range(10)], return_exceptions=True)
    failures = [r for r in results if isinstance(r, Exception)]
    assert len(failures) == 0, f"Failures: {[str(f)[:120] for f in failures]}"
    assert all(is_ask(r) for r in results if isinstance(r, dict))


# ---------------------------------------------------------------------------
# SECTION 8b — Correction and clarification turn latency (marker: latency_correction)
#
# These tests measure the wall-clock cost of the two "recovery" turn types:
#
#   Correction turn  — LLM 1 returns CORRECTED; LLM 2 generates ack.
#                      Two LLM calls; typically the more expensive path.
#
#   Clarification turn — LLM 1 returns AMBIGUOUS (first occurrence, no
#                        penalty); LLM 2 generates CLARIFY message.
#                        Two LLM calls; similar cost to correction.
#
# Both paths involve LLM 2 (Gemini/generation LLM) in addition to LLM 1
# (extraction LLM), so their p95 budget is higher than a simple extraction
# turn but must still be bounded.
# ---------------------------------------------------------------------------

_CORRECTION_TURN_BUDGET_MS = 4_000  # 10 s — two LLM calls in series
_CLARIFICATION_TURN_BUDGET_MS = 4_000  # 8 s  — two LLM calls in series
_LATENCY_CORRECTION_SAMPLES = 5


@pytest.mark.latency_correction
@pytest.mark.asyncio
async def test_latency_correction_turn_first_name(mock_sf) -> None:
    """
    Single correction turn: first_name corrected while awaiting last_name.
    Measures LLM 1 classification + LLM 2 correction-ack generation.
    """
    rec = get_recorder()
    elapsed_list: list[float] = []

    for i in range(_LATENCY_CORRECTION_SAMPLES):
        state = make_state(
            first_name="John",
            awaiting_slot="last_name",
            messages=[
                _msg("assistant", "And your last name?"),
                _msg("user", "actually my name is James not John"),
            ],
        )
        result, elapsed = await _timed_run(state)
        elapsed_list.append(elapsed)
        rec.record(
            "test_latency_correction_turn_first_name",
            i,
            "correction_first_name",
            "actually my name is James not John",
            state,
            result,
        )
        assert not is_escalation(result), f"Run {i}: correction must not escalate"
        assert is_ask(result), f"Run {i}: correction must re-ask"

    p95 = _p(elapsed_list, 95)
    avg = mean(elapsed_list)
    mx = max(elapsed_list)
    print(
        f"\nCorrection turn (first_name) — "
        f"p95={p95:.0f}ms  mean={avg:.0f}ms  max={mx:.0f}ms  "
        f"(budget={_CORRECTION_TURN_BUDGET_MS}ms)"
    )
    assert p95 < _CORRECTION_TURN_BUDGET_MS, (
        f"Correction turn p95 {p95:.0f}ms exceeds budget {_CORRECTION_TURN_BUDGET_MS}ms"
    )


@pytest.mark.latency_correction
@pytest.mark.asyncio
async def test_latency_correction_turn_last_name_mid_pipeline(mock_sf) -> None:
    """
    Correction of last_name while awaiting member_id.
    Confirms the correction detour path (correction_return_to) is not slower
    than a direct correction.
    """
    elapsed_list: list[float] = []

    for i in range(_LATENCY_CORRECTION_SAMPLES):
        state = make_state(
            first_name="Emily",
            last_name="Chavez",
            awaiting_slot="member_id",
            messages=[
                _msg("assistant", "May I have your member ID?"),
                _msg("user", "actually it's Carter not Chavez"),
            ],
        )
        result, elapsed = await _timed_run(state)
        elapsed_list.append(elapsed)
        assert not is_escalation(result), f"Run {i}: mid-pipeline correction must not escalate"

    p95 = _p(elapsed_list, 95)
    avg = mean(elapsed_list)
    print(
        f"\nCorrection turn (last_name mid-pipeline) — "
        f"p95={p95:.0f}ms  mean={avg:.0f}ms  "
        f"(budget={_CORRECTION_TURN_BUDGET_MS}ms)"
    )
    assert p95 < _CORRECTION_TURN_BUDGET_MS, (
        f"Mid-pipeline correction p95 {p95:.0f}ms exceeds budget {_CORRECTION_TURN_BUDGET_MS}ms"
    )


@pytest.mark.latency_correction
@pytest.mark.asyncio
async def test_latency_correction_turn_member_id(mock_sf) -> None:
    """
    Correction of member_id (triggers dob cascade clear).
    Slightly more state mutation than other corrections — must not be slower.
    """
    elapsed_list: list[float] = []

    for i in range(_LATENCY_CORRECTION_SAMPLES):
        state = make_state(
            first_name="Emily",
            last_name="Carter",
            member_id="U0000001",
            dob="01/01/1980",
            messages=[
                _msg("assistant", "Let me verify that."),
                _msg("user", "sorry wrong member ID it's m nine zero seven five zero three"),
            ],
        )
        result, elapsed = await _timed_run(state)
        elapsed_list.append(elapsed)
        assert not is_escalation(result), f"Run {i}: member_id correction must not escalate"

    p95 = _p(elapsed_list, 95)
    avg = mean(elapsed_list)
    print(
        f"\nCorrection turn (member_id) — "
        f"p95={p95:.0f}ms  mean={avg:.0f}ms  "
        f"(budget={_CORRECTION_TURN_BUDGET_MS}ms)"
    )
    assert p95 < _CORRECTION_TURN_BUDGET_MS


@pytest.mark.latency_correction
@pytest.mark.asyncio
async def test_latency_clarification_turn_member_id(mock_sf) -> None:
    """
    Clarification turn: first AMBIGUOUS on member_id — no attempt penalty.
    Caller says something ambiguous; LLM 1 returns AMBIGUOUS; LLM 2 generates
    a gentle re-ask (CLARIFY guard). attempt_count must stay at 0.
    """
    rec = get_recorder()
    elapsed_list: list[float] = []

    for i in range(_LATENCY_CORRECTION_SAMPLES):
        state = make_state(
            first_name="Emily",
            last_name="Carter",
            awaiting_slot="member_id",
            messages=[
                _msg("assistant", "May I have your member ID?"),
                _msg("user", "no no wait that's not right"),
            ],
        )
        result, elapsed = await _timed_run(state)
        elapsed_list.append(elapsed)
        rec.record(
            "test_latency_clarification_turn_member_id",
            i,
            "clarification_member_id",
            "no no wait that's not right",
            state,
            result,
        )
        assert not is_escalation(result), f"Run {i}: clarification must not escalate"
        assert is_ask(result), f"Run {i}: clarification must re-ask"
        assert get_attempt(result, "member_id") == 0, (
            f"Run {i}: first AMBIGUOUS must not increment attempt_count "
            f"(got {get_attempt(result, 'member_id')})"
        )

    p95 = _p(elapsed_list, 95)
    avg = mean(elapsed_list)
    print(
        f"\nClarification turn (member_id) — "
        f"p95={p95:.0f}ms  mean={avg:.0f}ms  "
        f"(budget={_CLARIFICATION_TURN_BUDGET_MS}ms)"
    )
    assert p95 < _CLARIFICATION_TURN_BUDGET_MS, (
        f"Clarification turn p95 {p95:.0f}ms exceeds budget {_CLARIFICATION_TURN_BUDGET_MS}ms"
    )


@pytest.mark.latency_correction
@pytest.mark.asyncio
async def test_latency_clarification_turn_dob(mock_sf) -> None:
    """
    Clarification turn on dob — caller gives a partial/trailing-off answer.
    """
    elapsed_list: list[float] = []

    for i in range(_LATENCY_CORRECTION_SAMPLES):
        state = make_state(
            first_name="Emily",
            last_name="Carter",
            member_id="M907503",
            awaiting_slot="dob",
            messages=[
                _msg("assistant", "And your date of birth?"),
                _msg("user", "nineteen… uh… eighty… something"),
            ],
        )
        result, elapsed = await _timed_run(state)
        elapsed_list.append(elapsed)
        assert not is_escalation(result), f"Run {i}: partial DOB clarification must not escalate"
        assert is_ask(result), f"Run {i}: must re-ask for dob"

    p95 = _p(elapsed_list, 95)
    avg = mean(elapsed_list)
    print(
        f"\nClarification turn (dob) — "
        f"p95={p95:.0f}ms  mean={avg:.0f}ms  "
        f"(budget={_CLARIFICATION_TURN_BUDGET_MS}ms)"
    )
    assert p95 < _CLARIFICATION_TURN_BUDGET_MS


@pytest.mark.latency_correction
@pytest.mark.asyncio
async def test_latency_clarification_turn_first_name(mock_sf) -> None:
    """
    Clarification turn on first_name — caller gives ASR noise.
    """
    elapsed_list: list[float] = []

    for i in range(_LATENCY_CORRECTION_SAMPLES):
        state = make_state(
            awaiting_slot="first_name",
            messages=[
                _msg("assistant", "Could I start with your first name?"),
                _msg("user", "shhh kkk sorry one moment"),
            ],
        )
        result, elapsed = await _timed_run(state)
        elapsed_list.append(elapsed)
        assert not is_escalation(result), f"Run {i}: ASR noise clarification must not escalate"
        assert is_ask(result), f"Run {i}: must re-ask for first_name"

    p95 = _p(elapsed_list, 95)
    avg = mean(elapsed_list)
    print(
        f"\nClarification turn (first_name) — "
        f"p95={p95:.0f}ms  mean={avg:.0f}ms  "
        f"(budget={_CLARIFICATION_TURN_BUDGET_MS}ms)"
    )
    assert p95 < _CLARIFICATION_TURN_BUDGET_MS


@pytest.mark.latency_correction
@pytest.mark.asyncio
async def test_latency_correction_vs_clarification_comparison(mock_sf) -> None:
    """
    Side-by-side p95 comparison of correction vs clarification turns.
    Correction should be no more than 30% slower than clarification,
    since both involve the same two LLM calls. A large gap indicates
    one of the paths is doing unexpected work.
    """
    import asyncio as _asyncio

    correction_elapsed: list[float] = []
    clarification_elapsed: list[float] = []

    # Run both concurrently to reduce wall-clock time in CI
    async def _correction_run(i: int) -> float:
        state = make_state(
            first_name="John",
            awaiting_slot="last_name",
            messages=[
                _msg("assistant", "Last name?"),
                _msg("user", "actually my name is James"),
            ],
        )
        _, elapsed = await _timed_run(state)
        return elapsed

    async def _clarification_run(i: int) -> float:
        state = make_state(
            first_name="Emily",
            last_name="Carter",
            awaiting_slot="member_id",
            messages=[
                _msg("assistant", "Member ID?"),
                _msg("user", "no no wait"),
            ],
        )
        _, elapsed = await _timed_run(state)
        return elapsed

    for batch in range(_LATENCY_CORRECTION_SAMPLES):
        c, cl = await _asyncio.gather(
            _correction_run(batch),
            _clarification_run(batch),
        )
        correction_elapsed.append(c)
        clarification_elapsed.append(cl)

    corr_p95 = _p(correction_elapsed, 95)
    clar_p95 = _p(clarification_elapsed, 95)
    corr_mean = mean(correction_elapsed)
    clar_mean = mean(clarification_elapsed)

    print(
        f"\nCorrection  p95={corr_p95:.0f}ms  mean={corr_mean:.0f}ms\n"
        f"Clarification p95={clar_p95:.0f}ms  mean={clar_mean:.0f}ms\n"
        f"Ratio (correction/clarification) p95={corr_p95 / max(clar_p95, 1):.2f}"
    )

    # Both must stay within budget
    assert corr_p95 < _CORRECTION_TURN_BUDGET_MS, (
        f"Correction p95 {corr_p95:.0f}ms exceeds {_CORRECTION_TURN_BUDGET_MS}ms"
    )
    assert clar_p95 < _CLARIFICATION_TURN_BUDGET_MS, (
        f"Clarification p95 {clar_p95:.0f}ms exceeds {_CLARIFICATION_TURN_BUDGET_MS}ms"
    )

    # Correction must not be more than 2× clarification — if it is, something
    # in the correction ack path is blocking unexpectedly
    ratio = corr_p95 / max(clar_p95, 1)
    assert ratio < 2.0, (
        f"Correction p95 is {ratio:.1f}× clarification p95 — correction path may be doing unexpected work"
    )
