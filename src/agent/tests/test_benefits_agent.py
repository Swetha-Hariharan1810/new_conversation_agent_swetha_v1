"""
test_benefits_agent.py — Live LLM tests for BenefitsAgent.

Requires AZURE_OPENAI_API_KEY to be set; all tests are skipped when absent.
Salesforce is mocked — only the LLM extraction is live.

Run all:    pytest src/agent/tests/test_benefits_agent.py -v
By marker:  pytest src/agent/tests/test_benefits_agent.py -v -m happy
"""

from __future__ import annotations

import asyncio
import os
import time
from unittest.mock import AsyncMock

import pytest

from agent.tests.helpers import load_test_env as _load_env

_load_env()

CREDS_AVAILABLE = bool(os.getenv("AZURE_OPENAI_API_KEY"))
pytestmark = pytest.mark.skipif(not CREDS_AVAILABLE, reason="AZURE_OPENAI_API_KEY not set")

from agent.agents.benefits.agent import BenefitsAgent  # noqa: E402
from agent.tests.fixtures import make_verified_state  # noqa: E402
from agent.tests.recorder import get_recorder  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_BENEFITS = {
    "individual_deductible": "750",
    "family_deductible": "2500",
    "coinsurance_percent": "20",
    "individual_oop_max": "3000",
    "family_oop_max": "7000",
}

FAX_ON_FILE = "6175554101"
EMAIL_ON_FILE = "emily@example.com"


def _msg(role: str, text: str) -> dict:
    return {"role": role, "content": text}


def make_benefits_state(**overrides) -> dict:
    defaults: dict = {
        "call_intent": "benefits_inquiry",
        "member_id": "M907503",
        "benefits_explained": False,
        "care_coach_offered": False,
        "care_coach_offer_made": False,
        "individual_deductible": "",
        "family_deductible": "",
        "coinsurance_percent": "",
        "individual_oop_max": "",
        "family_oop_max": "",
        "proactive_offer_available": True,
        "delivery_method": "fax",
        "fax": FAX_ON_FILE,
        "email": EMAIL_ON_FILE,
        "member_status_verify": True,
    }
    defaults.update(overrides)
    return make_verified_state(**defaults)


async def _run(state: dict) -> dict:
    return await BenefitsAgent.from_state(state).execute(state)


def is_ask(result: dict) -> bool:
    return result.get("is_interrupt") is True and result.get("next_node") == "benefits_agent"


def is_escalation(result: dict) -> bool:
    return result.get("next_node") == "escalation_agent" and result.get("is_interrupt") is False


def is_complete(result: dict) -> bool:
    return result.get("next_node") == "orchestrator" and result.get("is_interrupt") is False


def get_awaiting(result: dict) -> str:
    return result.get("awaiting_slot", "")


def get_response(result: dict) -> str:
    msg = result.get("messages", {})
    if isinstance(msg, dict):
        return msg.get("content", "")
    if isinstance(msg, list) and msg:
        last = msg[-1]
        return last.get("content", "") if isinstance(last, dict) else str(last)
    return ""


def advance(state: dict, result: dict, user_text: str | None = None) -> dict:
    new_state = {**state}
    for key, val in result.items():
        if key == "messages":
            continue
        new_state[key] = val
    messages = list(state.get("messages") or [])
    if isinstance(result.get("messages"), dict):
        messages.append(result["messages"])
    if user_text is not None:
        messages.append({"role": "user", "content": user_text})
    new_state["messages"] = messages
    return new_state


def _p(data: list[float], pct: float) -> float:
    s = sorted(data)
    n = len(s)
    k = (pct / 100) * (n - 1)
    lo, hi = int(k), min(int(k) + 1, n - 1)
    return s[lo] + (k - lo) * (s[hi] - s[lo])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_benefits_fetch(monkeypatch) -> AsyncMock:
    """Patch get_member_benefits to return SAMPLE_BENEFITS — no real Salesforce calls."""
    mock = AsyncMock(return_value=SAMPLE_BENEFITS)
    monkeypatch.setattr("agent.storage.queries.benefits.get_member_benefits", mock)
    return mock


# ---------------------------------------------------------------------------
# SECTION 1 — Happy path (marker: happy)
# ---------------------------------------------------------------------------


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_full_flow_care_coach_yes(mock_benefits_fetch) -> None:
    """Multi-turn: first entry delivers explanation + offer; member says yes → complete."""
    rec = get_recorder()
    user_input_1 = "can you tell me about my coverage?"

    # Turn 1: first entry
    state = make_benefits_state(messages=[_msg("user", user_input_1)])
    result = await _run(state)
    rec.record("test_happy_full_flow_care_coach_yes", 1, "yes_path", user_input_1, state, result)
    assert is_ask(result), "First entry must ask for care coach response"
    assert get_awaiting(result) == "care_coach_response"
    assert result.get("benefits_explained") is True

    # Turn 2: member accepts care coach offer
    user_input_2 = "yes that sounds great, please send details"
    state = advance(state, result, user_input_2)
    result = await _run(state)
    rec.record("test_happy_full_flow_care_coach_yes", 2, "yes_path", user_input_2, state, result)
    assert not is_escalation(result), "Accepting care coach must not escalate"
    assert is_complete(result) or is_ask(result), "Must complete or re-ask (not escalate)"
    if is_complete(result):
        assert result.get("proactive_offer_available") is True


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_full_flow_care_coach_no(mock_benefits_fetch) -> None:
    """Multi-turn: first entry delivers explanation + offer; member says no → complete."""
    rec = get_recorder()
    user_input_1 = "I need to understand my deductible"

    state = make_benefits_state(messages=[_msg("user", user_input_1)])
    result = await _run(state)
    rec.record("test_happy_full_flow_care_coach_no", 1, "no_path", user_input_1, state, result)
    assert is_ask(result), "First entry must ask for care coach response"
    assert get_awaiting(result) == "care_coach_response"

    user_input_2 = "no thank you, I'm all set"
    state = advance(state, result, user_input_2)
    result = await _run(state)
    rec.record("test_happy_full_flow_care_coach_no", 2, "no_path", user_input_2, state, result)
    assert not is_escalation(result), "Declining care coach must not escalate"
    assert is_complete(result) or is_ask(result)
    if is_complete(result):
        assert result.get("proactive_offer_available") is False


# ---------------------------------------------------------------------------
# SECTION 2 — Latency (marker: latency)
# ---------------------------------------------------------------------------


@pytest.mark.latency
@pytest.mark.asyncio
async def test_latency_benefits_explanation(mock_benefits_fetch) -> None:
    """p95 latency for the first entry (benefits explanation) turn must be < 4 s."""
    elapsed_list: list[float] = []
    samples = 5
    for _ in range(samples):
        state = make_benefits_state(messages=[_msg("user", "what are my benefits?")])
        t0 = time.perf_counter()
        result = await _run(state)
        elapsed_list.append(time.perf_counter() - t0)
        assert not is_escalation(result), "Explanation turn must not escalate"

    p95 = _p(elapsed_list, 95)
    print(f"\nBenefits explanation p95={p95 * 1000:.0f}ms (budget=4000ms)")
    assert p95 < 4.0, f"p95 {p95 * 1000:.0f}ms exceeds 4000ms budget"


# ---------------------------------------------------------------------------
# SECTION 3 — Additional happy + live LLM (marker: happy)
# ---------------------------------------------------------------------------


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_no_path_care_coach_offer(mock_benefits_fetch) -> None:
    """NO path (proactive_offer_available=False): SF not fetched, benefits_explained=False."""
    state = make_benefits_state(
        proactive_offer_available=False,
        messages=[_msg("user", "what can you tell me about care coaching?")],
    )
    result = await _run(state)
    assert is_ask(result), "NO path first entry must still ask (care coach offer)"
    assert get_awaiting(result) == "care_coach_response"
    assert result.get("benefits_explained") is False, (
        f"NO path must not set benefits_explained=True; got {result.get('benefits_explained')!r}"
    )
    mock_benefits_fetch.assert_not_called()


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_care_coach_yes_full_live(mock_benefits_fetch) -> None:
    """Phase B live LLM: member says 'yes' → complete with proactive_offer_available=True."""
    # Build Phase B state directly (already past first entry)
    state = make_benefits_state(
        proactive_offer_available=True,
        awaiting_slot="care_coach_response",
        benefits_explained=True,
        individual_deductible="750",
        family_deductible="2500",
        coinsurance_percent="20",
        individual_oop_max="3000",
        family_oop_max="7000",
        messages=[
            _msg("assistant", "Would you like me to send you Care Coach details?"),
            _msg("user", "yes please go ahead"),
        ],
    )
    result = await _run(state)
    assert not is_escalation(result), "Phase B 'yes' must not escalate"
    assert is_complete(result) or is_ask(result), "Phase B 'yes' must complete or re-ask"
    if is_complete(result):
        assert result.get("proactive_offer_available") is True


# ---------------------------------------------------------------------------
# SECTION 4 — Latency + stress (marker: latency / stress)
# ---------------------------------------------------------------------------


@pytest.mark.latency
@pytest.mark.asyncio
async def test_latency_care_coach_extraction(mock_benefits_fetch) -> None:
    """p95 latency for the care_coach_response extraction turn must be < 3 s."""
    elapsed_list: list[float] = []
    samples = 5
    for _ in range(samples):
        state = make_benefits_state(
            proactive_offer_available=True,
            awaiting_slot="care_coach_response",
            benefits_explained=True,
            individual_deductible="750",
            family_deductible="2500",
            coinsurance_percent="20",
            individual_oop_max="3000",
            family_oop_max="7000",
            messages=[
                _msg("assistant", "Would you like Care Coach details sent to your fax?"),
                _msg("user", "yes please"),
            ],
        )
        t0 = time.perf_counter()
        result = await _run(state)
        elapsed_list.append(time.perf_counter() - t0)
        assert not is_escalation(result), "Extraction turn must not escalate"

    p95 = _p(elapsed_list, 95)
    print(f"\nCare coach extraction p95={p95 * 1000:.0f}ms (budget=3000ms)")
    assert p95 < 3.0, f"p95 {p95 * 1000:.0f}ms exceeds 3000ms budget"


@pytest.mark.stress
@pytest.mark.asyncio
async def test_stress_5_concurrent_first_entry(mock_benefits_fetch) -> None:
    """5 concurrent YES-path first-entry runs via asyncio.gather — at most 1 failure allowed."""

    async def _one_run() -> dict:
        state = make_benefits_state(
            proactive_offer_available=True,
            messages=[_msg("user", "can you explain my benefits?")],
        )
        return await _run(state)

    results = await asyncio.gather(*[_one_run() for _ in range(5)], return_exceptions=True)
    failures = [r for r in results if isinstance(r, Exception) or is_escalation(r)]  # type: ignore[arg-type]
    assert len(failures) <= 1, f"Too many failures under concurrency: {len(failures)}/5 — {failures}"
