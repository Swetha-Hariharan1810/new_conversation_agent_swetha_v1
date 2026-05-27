"""
test_provider_search_agent.py — Live LLM tests for ProviderSearchAgent.

Requires AZURE_OPENAI_API_KEY to be set; all tests are skipped when absent.

Run all:    pytest src/agent/tests/test_provider_search_agent.py -v
By marker:  pytest src/agent/tests/test_provider_search_agent.py -v -m happy
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.tests.helpers import load_test_env as _load_env

_load_env()

CREDS_AVAILABLE = bool(os.getenv("AZURE_OPENAI_API_KEY"))
pytestmark = pytest.mark.skipif(not CREDS_AVAILABLE, reason="AZURE_OPENAI_API_KEY not set")

from agent.agents.provider_search.agent import ProviderSearchAgent  # noqa: E402
from agent.tests.fixtures import advance, get_awaiting, get_response, make_verified_state  # noqa: E402
from agent.tests.recorder import get_recorder  # noqa: E402

_VERIFIED_MEMBER_RECORD = {
    "verified": True,
    "phone_number": "6175554101",
    "zip_code": "12139",
    "relationship": "plan_holder",
}

ZIP_ON_FILE = "12139"
PROVIDER = "Primary Care Physician"


def _msg(role: str, text: str) -> dict:
    return {"role": role, "content": text}


def make_ps_state(**overrides) -> dict:
    defaults: dict = {
        "call_intent": "provider_services",
        "provider_type": "",
        "zip_code_used": "",
        "zip_code": ZIP_ON_FILE,
        "member_status_verify": True,
    }
    defaults.update(overrides)
    return make_verified_state(**defaults)


async def _run(state: dict) -> dict:
    return await ProviderSearchAgent.from_state(state).execute(state)


def is_ask(result: dict) -> bool:
    return result.get("is_interrupt") is True and result.get("next_node") == "provider_search_agent"


def is_escalation(result: dict) -> bool:
    return result.get("next_node") == "escalation_agent" and result.get("is_interrupt") is False


def is_done(result: dict) -> bool:
    return result.get("next_node") == "delivery_management_agent" and result.get("is_interrupt") is True


def _p(data: list[float], pct: float) -> float:
    s = sorted(data)
    n = len(s)
    k = (pct / 100) * (n - 1)
    lo, hi = int(k), min(int(k) + 1, n - 1)
    return s[lo] + (k - lo) * (s[hi] - s[lo])


@pytest.fixture
def mock_sf(monkeypatch):
    tool = MagicMock()
    tool.ainvoke = AsyncMock(return_value=_VERIFIED_MEMBER_RECORD)
    monkeypatch.setattr("agent.storage.tools.lookup_member", tool)
    return tool


@pytest.fixture
def mock_zip_update(monkeypatch):
    tool = MagicMock()
    tool.ainvoke = AsyncMock(return_value=True)
    monkeypatch.setattr("agent.storage.tools.update_zip_code", tool)
    return tool


# ---------------------------------------------------------------------------
# SECTION 1 — Happy path (marker: happy)
# ---------------------------------------------------------------------------


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_provider_step_by_step(mock_sf, mock_zip_update) -> None:
    """Full multi-turn: provider_type + zip confirmed → delivery routing."""
    rec = get_recorder()

    user_1 = "I need help finding a doctor"
    state = make_ps_state(messages=[_msg("user", user_1)])
    result = await _run(state)
    rec.record("test_happy_provider_step_by_step", 1, "provider_search", user_1, state, result)
    assert is_ask(result), "First turn must ask for provider type"
    assert get_awaiting(result) == "provider_type"

    user_2 = "primary care physician"
    state = advance(state, result, user_2)
    result = await _run(state)
    rec.record("test_happy_provider_step_by_step", 2, "provider_search", user_2, state, result)
    assert not is_escalation(result), "Provider type turn must not escalate"
    # Either asks for zip confirmation or already done
    if is_ask(result):
        assert get_awaiting(result) in ("zip_confirmed", "zip_code")
        assert ZIP_ON_FILE in get_response(result) or get_awaiting(result) == "zip_code"

        user_3 = "yes that's right"
        state = advance(state, result, user_3)
        result = await _run(state)
        rec.record("test_happy_provider_step_by_step", 3, "zip_confirmed", user_3, state, result)

    assert is_done(result) or (is_ask(result) and get_awaiting(result) != "provider_type"), (
        "After zip confirmed, must route to delivery management"
    )
    if is_done(result):
        assert result.get("next_node") == "delivery_management_agent"
        assert result.get("provider_type") in (PROVIDER, "Primary Care Physician")


# ---------------------------------------------------------------------------
# SECTION 2 — Latency (marker: latency)
# ---------------------------------------------------------------------------


@pytest.mark.latency
@pytest.mark.asyncio
async def test_latency_provider_type_collection(mock_sf, mock_zip_update) -> None:
    """p95 latency for a single provider_type collection turn must be < 4 s."""
    elapsed_list: list[float] = []
    samples = 5
    for _ in range(samples):
        state = make_ps_state(
            awaiting_slot="provider_type",
            messages=[
                _msg("assistant", "What type of provider are you looking for?"),
                _msg("user", "I need a cardiologist"),
            ],
        )
        t0 = time.perf_counter()
        result = await _run(state)
        elapsed_list.append(time.perf_counter() - t0)
        assert not is_escalation(result), "Provider type turn must not escalate"

    p95 = _p(elapsed_list, 95)
    print(f"\nProvider type collection p95={p95 * 1000:.0f}ms (budget=4000ms)")
    assert p95 < 4.0, f"p95 {p95 * 1000:.0f}ms exceeds 4000ms budget"


# ---------------------------------------------------------------------------
# SECTION 3 — Stress (marker: stress)
# ---------------------------------------------------------------------------


@pytest.mark.stress
@pytest.mark.asyncio
async def test_stress_10_concurrent(mock_sf, mock_zip_update) -> None:
    """10 concurrent provider_type collection turns — at least 9/10 must succeed."""

    async def _one() -> dict:
        state = make_ps_state(
            app_run_id=str(uuid.uuid4()),
            awaiting_slot="provider_type",
            messages=[
                _msg("assistant", "What type of provider?"),
                _msg("user", "primary care physician"),
            ],
        )
        return await _run(state)

    results = await asyncio.gather(*[_one() for _ in range(10)], return_exceptions=True)
    failures = [r for r in results if isinstance(r, Exception)]
    escalations = [r for r in results if isinstance(r, dict) and is_escalation(r)]
    bad = len(failures) + len(escalations)
    assert bad <= 1, (
        f"Expected at most 1 failure out of 10; got {bad} "
        f"(exceptions={len(failures)}, escalations={len(escalations)})"
    )


# ---------------------------------------------------------------------------
# SECTION 4 — PCP spoken natural language (marker: happy)
# ---------------------------------------------------------------------------


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_pcp_spoken(mock_sf, mock_zip_update) -> None:
    """'primary care physician' spoken → provider_type resolves to 'Primary Care Physician'."""
    rec = get_recorder()

    user_1 = "primary care physician"
    state = make_ps_state(
        awaiting_slot="provider_type",
        messages=[
            _msg("assistant", "What type of provider are you looking for?"),
            _msg("user", user_1),
        ],
    )
    result = await _run(state)
    rec.record("test_happy_pcp_spoken", 1, "pcp_spoken", user_1, state, result)
    assert not is_escalation(result), "PCP spoken must not escalate"
    if is_ask(result):
        assert get_awaiting(result) in ("zip_confirmed", "zip_code"), (
            f"Unexpected awaiting after PCP: {get_awaiting(result)!r}"
        )
    if result.get("provider_type"):
        assert result.get("provider_type") == PROVIDER, (
            f"Expected provider_type={PROVIDER!r}, got {result.get('provider_type')!r}"
        )


# ---------------------------------------------------------------------------
# SECTION 5 — Multi-turn: zip declined then new zip provided (marker: happy)
# ---------------------------------------------------------------------------


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_zip_declined_new_provided(mock_sf, mock_zip_update) -> None:
    """zip_confirmed → 'no' → new zip provided → confirm → done."""
    rec = get_recorder()

    # Turn 1: member declines current zip on file
    user_1 = "no that's not right"
    state = make_ps_state(
        provider_type=PROVIDER,
        awaiting_slot="zip_confirmed",
        messages=[
            _msg("assistant", f"The zip code we have on file is {ZIP_ON_FILE}. Is that correct?"),
            _msg("user", user_1),
        ],
    )
    result = await _run(state)
    rec.record("test_happy_zip_declined_new_provided", 1, "zip_declined", user_1, state, result)
    assert not is_escalation(result), "Zip declined must not escalate"

    if is_ask(result):
        assert get_awaiting(result) == "zip_code", (
            f"After decline, must ask for new zip; got {get_awaiting(result)!r}"
        )

        # Turn 2: provide a new zip code
        user_2 = "02134"
        state = advance(state, result, user_2)
        result = await _run(state)
        rec.record("test_happy_zip_declined_new_provided", 2, "zip_new", user_2, state, result)
        assert not is_escalation(result), "New zip turn must not escalate"

        if is_ask(result):
            assert get_awaiting(result) == "zip_confirmed", (
                f"After new zip, must confirm; got {get_awaiting(result)!r}"
            )

            # Turn 3: confirm the new zip
            user_3 = "yes"
            state = advance(state, result, user_3)
            result = await _run(state)
            rec.record("test_happy_zip_declined_new_provided", 3, "zip_confirmed", user_3, state, result)
            assert not is_escalation(result), "Zip confirmation must not escalate"

    assert is_done(result) or is_ask(result), (
        "Flow must either complete or continue asking — not escalate"
    )


# ---------------------------------------------------------------------------
# SECTION 6 — Latency: zip_confirmed extraction (marker: latency)
# ---------------------------------------------------------------------------


@pytest.mark.latency
@pytest.mark.asyncio
async def test_latency_zip_confirmation_extraction(mock_sf, mock_zip_update) -> None:
    """p95 latency for the zip_confirmed extraction turn must be < 4 s."""
    elapsed_list: list[float] = []
    samples = 5
    for _ in range(samples):
        state = make_ps_state(
            provider_type=PROVIDER,
            awaiting_slot="zip_confirmed",
            messages=[
                _msg("assistant", f"The zip code we have on file is {ZIP_ON_FILE}. Is that correct?"),
                _msg("user", "yes that is correct"),
            ],
        )
        t0 = time.perf_counter()
        result = await _run(state)
        elapsed_list.append(time.perf_counter() - t0)
        assert not is_escalation(result), "Zip confirmation must not escalate"

    p95 = _p(elapsed_list, 95)
    print(f"\nZip confirmation extraction p95={p95 * 1000:.0f}ms (budget=4000ms)")
    assert p95 < 4.0, f"p95 {p95 * 1000:.0f}ms exceeds 4000ms budget"
