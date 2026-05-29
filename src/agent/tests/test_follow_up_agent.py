"""
test_follow_up_agent.py — Live LLM tests for FollowUpAgent.

Requires AZURE_OPENAI_API_KEY. All tests are skipped when absent.
Salesforce is not called — answers come purely from session state.

Run all:    pytest src/agent/tests/test_follow_up_agent.py -v
By marker:  pytest src/agent/tests/test_follow_up_agent.py -v -m happy
"""

from __future__ import annotations

import asyncio
import os
import time

import pytest

from agent.tests.helpers import load_test_env as _load_env

_load_env()

CREDS_AVAILABLE = bool(os.getenv("AZURE_OPENAI_API_KEY"))
pytestmark = pytest.mark.skipif(not CREDS_AVAILABLE, reason="AZURE_OPENAI_API_KEY not set")

from agent.agents.follow_up.agent import FollowUpAgent  # noqa: E402
from agent.tests.fixtures import make_verified_state  # noqa: E402
from agent.tests.recorder import get_recorder  # noqa: E402

FAX_ON_FILE = "6175554101"


def make_fu_state(**overrides) -> dict:
    defaults: dict = {
        "call_intent": "provider_services",
        "member_status_verify": True,
        "provider_type": "Primary Care Physician",
        "zip_code_used": "12139",
        "zip_code": "12139",
        "provider_list_sent": True,
        "benefits_explained": True,
        "care_coach_offered": True,
        "care_coach_details_sent": True,
        "delivery_method": "fax",
        "fax": FAX_ON_FILE,
        "individual_deductible": "750",
        "family_deductible": "2500",
        "coinsurance_percent": "20",
        "individual_oop_max": "3000",
        "family_oop_max": "7000",
        "follow_up_turn_count": 0,
        "follow_up_last_question": "",
    }
    defaults.update(overrides)
    member_status_verify = defaults.pop("member_status_verify", True)
    state = make_verified_state(**defaults)
    state["member_status_verify"] = member_status_verify
    return state


def _msg(role: str, text: str) -> dict:
    return {"role": role, "content": text}


def advance(state, result, user_text):
    new_state = {**state}
    for key, val in result.items():
        if key == "messages":
            continue
        new_state[key] = val
    messages = list(state.get("messages") or [])
    if isinstance(result.get("messages"), dict):
        messages.append(result["messages"])
    messages.append({"role": "user", "content": user_text})
    new_state["messages"] = messages
    return new_state


async def _run(state: dict) -> dict:
    return await FollowUpAgent.from_state(state).execute(state)


def is_ask(result: dict) -> bool:
    return result.get("is_interrupt") is True and result.get("next_node") == "follow_up_agent"


def is_closure(result: dict) -> bool:
    signal = result.get("last_agent_signal") or {}
    return (
        result.get("is_interrupt") is False
        and result.get("next_node") == "orchestrator"
        and signal.get("closure_requested") is True
    )


def get_response(result: dict) -> str:
    msg = result.get("messages", {})
    if isinstance(msg, dict):
        return msg.get("content", "")
    if isinstance(msg, list) and msg:
        last = msg[-1]
        return last.get("content", "") if isinstance(last, dict) else str(last)
    return ""


def _p(data: list[float], pct: float) -> float:
    s = sorted(data)
    n = len(s)
    k = (pct / 100) * (n - 1)
    lo, hi = int(k), min(int(k) + 1, n - 1)
    return s[lo] + (k - lo) * (s[hi] - s[lo])


# ---------------------------------------------------------------------------
# SECTION 1 — Happy path (marker: happy)
# ---------------------------------------------------------------------------


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_closure_no_thanks() -> None:
    state = make_fu_state(
        messages=[
            _msg("assistant", "Would you like help with anything else?"),
            _msg("user", "No thanks that was helpful"),
        ]
    )
    result = await _run(state)
    assert is_closure(result), f"Expected closure, got next_node={result.get('next_node')!r}"


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_individual_deductible_question() -> None:
    state = make_fu_state(
        messages=[
            _msg("assistant", "Would you like help with anything else?"),
            _msg("user", "can you tell me my individual deductible"),
        ]
    )
    result = await _run(state)
    assert is_ask(result)
    assert "$750" in get_response(result), f"Expected $750 in response: {get_response(result)!r}"


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_benefits_summary() -> None:
    state = make_fu_state(
        messages=[
            _msg("assistant", "Would you like help with anything else?"),
            _msg("user", "Can you summarize the PCP benefits?"),
        ]
    )
    result = await _run(state)
    assert is_ask(result)
    response = get_response(result)
    assert "$750" in response, f"Missing deductible: {response!r}"
    assert "$2500" in response, f"Missing family deductible: {response!r}"
    assert "20%" in response, f"Missing coinsurance: {response!r}"


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_continuation_question_present() -> None:
    state = make_fu_state(
        messages=[
            _msg("assistant", "Would you like help with anything else?"),
            _msg("user", "what is my out-of-pocket maximum"),
        ]
    )
    result = await _run(state)
    assert is_ask(result)
    response = get_response(result)
    assert any(
        p in response.lower()
        for p in [
            "anything else",
            "anything i can",
            "other questions",
            "i can assist",
        ]
    ), f"Continuation question missing: {response!r}"


# ---------------------------------------------------------------------------
# SECTION 2 — Multi-turn flow (marker: happy)
# ---------------------------------------------------------------------------


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_two_questions_then_close() -> None:
    rec = get_recorder()

    user_1 = "what is my individual deductible"
    state = make_fu_state(messages=[_msg("assistant", "Is there anything else?"), _msg("user", user_1)])
    result1 = await _run(state)
    rec.record("test_happy_two_questions_then_close", 1, "deductible_q", user_1, state, result1)
    assert is_ask(result1)
    assert "$750" in get_response(result1)

    user_2 = "No thanks that was helpful"
    state2 = advance(state, result1, user_2)
    result2 = await _run(state2)
    rec.record("test_happy_two_questions_then_close", 2, "closure", user_2, state2, result2)
    assert is_closure(result2)


# ---------------------------------------------------------------------------
# SECTION 3 — Latency (marker: latency)
# ---------------------------------------------------------------------------


@pytest.mark.latency
@pytest.mark.asyncio
async def test_latency_answer_generation_p95_under_4s() -> None:
    """p95 of 5 answer-generation turns must be < 4 s."""
    elapsed_list: list[float] = []
    samples = 5
    for _ in range(samples):
        state = make_fu_state(
            messages=[
                _msg("assistant", "Anything else?"),
                _msg("user", "Can you summarize the PCP benefits?"),
            ]
        )
        t0 = time.perf_counter()
        result = await _run(state)
        elapsed_list.append(time.perf_counter() - t0)
        assert is_ask(result), "Answer turn must not close"

    p95 = _p(elapsed_list, 95)
    print(f"\nFollow-up answer generation p95={p95 * 1000:.0f}ms (budget=4000ms)")
    assert p95 < 4.0, f"p95 {p95 * 1000:.0f}ms exceeds 4000ms"


# ---------------------------------------------------------------------------
# SECTION 4 — Additional question types (marker: happy)
# ---------------------------------------------------------------------------


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_family_deductible_question() -> None:
    """LLM answers a family deductible question from session context."""
    rec = get_recorder()
    user_1 = "what is the family deductible?"
    state = make_fu_state(
        messages=[_msg("assistant", "Would you like help with anything else?"), _msg("user", user_1)]
    )
    result = await _run(state)
    rec.record("test_happy_family_deductible_question", 1, "family_ded", user_1, state, result)
    assert is_ask(result), f"Expected ask, got next_node={result.get('next_node')!r}"
    assert "$2500" in get_response(result), f"Expected $2500 in response: {get_response(result)!r}"


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_oop_max_question() -> None:
    """LLM answers an out-of-pocket maximum question from session context."""
    rec = get_recorder()
    user_1 = "what is my out-of-pocket maximum?"
    state = make_fu_state(
        messages=[_msg("assistant", "Would you like help with anything else?"), _msg("user", user_1)]
    )
    result = await _run(state)
    rec.record("test_happy_oop_max_question", 1, "oop_max", user_1, state, result)
    assert is_ask(result), f"Expected ask, got next_node={result.get('next_node')!r}"
    response = get_response(result)
    assert "$3000" in response or "$7000" in response, f"Expected OOP max amount in response: {response!r}"


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_care_coach_question() -> None:
    """LLM answers 'where were the care coach details sent?' from session context."""
    rec = get_recorder()
    user_1 = "where were the care coach details sent?"
    state = make_fu_state(
        care_coach_details_sent=True,
        delivery_method="fax",
        fax=FAX_ON_FILE,
        messages=[_msg("assistant", "Would you like help with anything else?"), _msg("user", user_1)],
    )
    result = await _run(state)
    rec.record("test_happy_care_coach_question", 1, "care_coach_q", user_1, state, result)
    assert is_ask(result), f"Expected ask, got next_node={result.get('next_node')!r}"
    response = get_response(result)
    assert FAX_ON_FILE in response or "fax" in response.lower(), (
        f"Expected fax contact in response: {response!r}"
    )


# ---------------------------------------------------------------------------
# SECTION 5 — Stress: concurrent answer generation (marker: stress)
# ---------------------------------------------------------------------------


@pytest.mark.stress
@pytest.mark.asyncio
async def test_stress_5_concurrent() -> None:
    """5 concurrent answer-generation turns — at most 1 failure allowed."""

    async def _one_run() -> dict:
        state = make_fu_state(
            messages=[
                _msg("assistant", "Is there anything else?"),
                _msg("user", "Can you summarize my benefits?"),
            ]
        )
        return await _run(state)

    results = await asyncio.gather(*[_one_run() for _ in range(5)], return_exceptions=True)
    failures = [r for r in results if isinstance(r, Exception)]
    bad_results = [r for r in results if isinstance(r, dict) and not is_ask(r) and not is_closure(r)]
    total_bad = len(failures) + len(bad_results)
    assert total_bad <= 1, (
        f"Too many failures under concurrency: {total_bad}/5 — "
        f"exceptions={len(failures)}, bad_results={len(bad_results)}"
    )
