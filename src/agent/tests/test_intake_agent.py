"""
tests/test_intake_agent_live.py

Live integration tests for IntakeAgent.
All tests call the real LLM — Azure OpenAI credentials must be
present in .env at the project root.

Run all live tests:
  pytest src/agent/tests/test_intake_agent.py -v

Run only happy/unhappy/latency/stress:
  pytest src/agent/tests/test_intake_agent_live.py -v -m happy
  pytest src/agent/tests/test_intake_agent_live.py -v -m unhappy
  pytest src/agent/tests/test_intake_agent_live.py -v -m latency
  pytest src/agent/tests/test_intake_agent_live.py -v -m stress

Skip live tests in CI:
  pytest --ignore=src/agent/tests/test_intake_agent_live.py

Custom marks: happy, unhappy, latency, stress
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from statistics import mean, median

import pytest

from agent.agents.intake.agent import IntakeAgent
from agent.agents.intake.constants import GREETING, INTENT_BRIDGE_MSG
from agent.orchestration.orchestration import AgentNode

# ---------------------------------------------------------------------------
# SECTION 0 — .env loading
# ---------------------------------------------------------------------------
from agent.tests.helpers import load_test_env as _load_dotenv_inline  # noqa: E402
from agent.tests.recorder import get_recorder

_load_dotenv_inline()

CREDS_AVAILABLE = bool(os.getenv("AZURE_OPENAI_API_KEY"))

pytestmark = pytest.mark.skipif(
    not CREDS_AVAILABLE,
    reason="AZURE_OPENAI_API_KEY not set — skipping live tests",
)

# ---------------------------------------------------------------------------
# SECTION 1 — Imports of production code (after .env is loaded)
# ---------------------------------------------------------------------------

GREETING_TEXT: str = GREETING
BRIDGE_TEXT: str = INTENT_BRIDGE_MSG

# ---------------------------------------------------------------------------
# SECTION 1 — Shared fixtures and helpers
# ---------------------------------------------------------------------------


def _make_state(messages=None, **overrides) -> dict:
    """Return the minimal state dict IntakeAgent needs."""
    state: dict = {
        "messages": messages if messages is not None else [],
        "metadata_events": [],
        "is_interrupt": False,
        "next_node": "",
        "app_run_id": str(uuid.uuid4()),
        "slot_attempts": {},
        "call_intent": "",
        "conversation_summary": None,
        "awaiting_slot": "",
        "active_agent": "",
    }
    state.update(overrides)
    return state


def _with_turn(state: dict, user_text: str, ai_text: str | None = None) -> dict:
    """Return a new state with messages extended by an optional AI turn then a user turn."""
    new_messages = list(state.get("messages") or [])
    if ai_text is not None:
        new_messages.append({"role": "assistant", "content": ai_text})
    new_messages.append({"role": "user", "content": user_text})
    return {**state, "messages": new_messages}


async def run_intake(state: dict) -> dict:
    """Convenience wrapper: build IntakeAgent from state and execute."""
    return await IntakeAgent.from_state(state).execute(state)


# ---------------------------------------------------------------------------
# SECTION 2 — Happy path tests
# ---------------------------------------------------------------------------


@pytest.mark.happy
@pytest.mark.asyncio
async def test_greeting_returns_static_message() -> None:
    state = _make_state()  # no messages
    result = await run_intake(state)

    assert result["is_interrupt"] is True
    assert result["next_node"] == "intake_agent"
    assert GREETING_TEXT in result["messages"]["content"]
    assert result.get("call_intent", "") == ""


@pytest.mark.happy
@pytest.mark.asyncio
async def test_provider_services_explicit() -> None:
    state = _make_state()
    state = _with_turn(
        state,
        user_text="I need to find a cardiologist in my network.",
        ai_text=GREETING_TEXT,
    )
    result = await run_intake(state)

    assert result["call_intent"] == "provider_services"
    assert result["next_node"] == AgentNode.VERIFICATION.value
    assert result["is_interrupt"] is True
    assert BRIDGE_TEXT in result["messages"]["content"]


@pytest.mark.happy
@pytest.mark.asyncio
async def test_provider_services_short() -> None:
    state = _make_state()
    state = _with_turn(state, user_text="provider", ai_text=GREETING_TEXT)
    result = await run_intake(state)

    assert result["call_intent"] == "provider_services"
    assert result["next_node"] == AgentNode.VERIFICATION.value
    assert result["is_interrupt"] is True
    assert BRIDGE_TEXT in result["messages"]["content"]


@pytest.mark.happy
@pytest.mark.asyncio
async def test_provider_services_natural() -> None:
    state = _make_state()
    state = _with_turn(
        state,
        user_text="uh yeah so I need to find a doctor in my network",
        ai_text=GREETING_TEXT,
    )
    result = await run_intake(state)

    assert result["call_intent"] == "provider_services"
    assert result["next_node"] == AgentNode.VERIFICATION.value
    assert result["is_interrupt"] is True
    assert BRIDGE_TEXT in result["messages"]["content"]


@pytest.mark.happy
@pytest.mark.asyncio
async def test_claim_services_explicit() -> None:
    state = _make_state()
    state = _with_turn(
        state,
        user_text="I want to check on my denied claim from last month.",
        ai_text=GREETING_TEXT,
    )
    result = await run_intake(state)

    assert result["call_intent"] == "claim_services"
    assert result["next_node"] == AgentNode.VERIFICATION.value


@pytest.mark.happy
@pytest.mark.asyncio
async def test_claim_services_short() -> None:
    state = _make_state()
    state = _with_turn(state, user_text="claim status please", ai_text=GREETING_TEXT)
    result = await run_intake(state)

    assert result["call_intent"] == "claim_services"


@pytest.mark.happy
@pytest.mark.asyncio
async def test_metadata_event_emitted() -> None:
    state = _make_state()
    state = _with_turn(state, user_text="find a provider", ai_text=GREETING_TEXT)
    result = await run_intake(state)

    events = result["metadata_events"]
    assert isinstance(events, list)
    assert len(events) >= 1

    first = events[0]
    assert first["eventType"] == "CallAgentField"
    assert first["data"]["field"] == "call_intent"
    assert first["data"]["value"] in {"provider_services", "claim_services"}


@pytest.mark.happy
@pytest.mark.asyncio
async def test_next_node_is_verification_after_bridge() -> None:
    state = _make_state()
    state = _with_turn(state, user_text="provider services", ai_text=GREETING_TEXT)
    result = await run_intake(state)

    assert result["next_node"] == AgentNode.VERIFICATION.value


@pytest.mark.happy
@pytest.mark.asyncio
async def test_reentry_guard_skips_llm() -> None:
    state = _make_state()
    state["call_intent"] = "provider_services"  # already classified
    state["messages"] = [{"role": "user", "content": "its emily"}]
    result = await run_intake(state)

    assert result["is_interrupt"] is False
    assert result["next_node"] == "orchestrator"
    assert result.get("call_intent") == "provider_services"


# ---------------------------------------------------------------------------
# SECTION 3 — Non-happy path tests
# ---------------------------------------------------------------------------


@pytest.mark.unhappy
@pytest.mark.asyncio
async def test_unclear_intent_first_attempt() -> None:
    state = _make_state()
    state = _with_turn(
        state,
        user_text="I have a question about my insurance.",
        ai_text=GREETING_TEXT,
    )
    result = await run_intake(state)

    assert result["is_interrupt"] is True
    assert result["next_node"] == "intake_agent"
    assert result.get("call_intent", "") == ""
    assert result["messages"]["content"] != ""


@pytest.mark.unhappy
@pytest.mark.asyncio
async def test_unclear_intent_second_attempt() -> None:
    state = _make_state(slot_attempts={"intent": {"attempt_count": 1}})
    state = _with_turn(
        state,
        user_text="I'm not sure what I need help with",
        ai_text=GREETING_TEXT,
    )
    result = await run_intake(state)

    assert result["is_interrupt"] is True
    assert result["next_node"] == "intake_agent"


@pytest.mark.unhappy
@pytest.mark.asyncio
async def test_unclear_intent_max_retries_escalates() -> None:
    state = _make_state(slot_attempts={"intent": {"attempt_count": 2}})
    state = _with_turn(state, user_text="umm I don't know", ai_text=GREETING_TEXT)
    result = await run_intake(state)

    assert result["next_node"] == AgentNode.ESCALATION.value
    assert result["is_interrupt"] is False


@pytest.mark.unhappy
@pytest.mark.asyncio
async def test_offtopic_first_attempt() -> None:
    state = _make_state()
    state = _with_turn(
        state,
        user_text="What's the weather like in Miami today?",
        ai_text=GREETING_TEXT,
    )
    result = await run_intake(state)

    assert result["is_interrupt"] is True
    assert result["next_node"] == "intake_agent"
    assert result.get("call_intent", "") == ""


@pytest.mark.unhappy
@pytest.mark.asyncio
async def test_offtopic_max_retries_escalates() -> None:
    # offtopic_global_count=1: guard increments to 2 >= MAX_SLOT_ATTEMPTS(2) → ESCALATE
    # Using the correct state key (offtopic_global_count, not slot_attempts)
    state = _make_state(offtopic_global_count=1)
    state = _with_turn(
        state,
        user_text="Can you recommend a good restaurant?",
        ai_text=GREETING_TEXT,
    )
    result = await run_intake(state)

    assert result["next_node"] == AgentNode.ESCALATION.value


@pytest.mark.unhappy
@pytest.mark.asyncio
async def test_transfer_request_escalates_immediately() -> None:
    state = _make_state()
    state = _with_turn(
        state,
        user_text="Get me a real person right now.",
        ai_text=GREETING_TEXT,
    )
    result = await run_intake(state)

    assert result["next_node"] == AgentNode.ESCALATION.value
    assert result["is_interrupt"] is False


@pytest.mark.unhappy
@pytest.mark.asyncio
async def test_transfer_request_polite() -> None:
    state = _make_state()
    state = _with_turn(
        state,
        user_text="Can I speak to a representative please?",
        ai_text=GREETING_TEXT,
    )
    result = await run_intake(state)

    assert result["next_node"] == AgentNode.ESCALATION.value


@pytest.mark.unhappy
@pytest.mark.asyncio
async def test_asr_empty_utterance() -> None:
    state = _make_state()
    state = _with_turn(state, user_text="", ai_text=GREETING_TEXT)
    result = await run_intake(state)

    assert result["is_interrupt"] is True
    assert result["next_node"] == "intake_agent"
    assert result.get("call_intent", "") == ""


@pytest.mark.unhappy
@pytest.mark.asyncio
async def test_asr_noise() -> None:
    state = _make_state()
    state = _with_turn(state, user_text="zzzshhh kkkk mmph", ai_text=GREETING_TEXT)
    result = await run_intake(state)

    assert result["is_interrupt"] is True
    assert result.get("call_intent", "") == ""


# ---------------------------------------------------------------------------
# SECTION 4 — Latency tests
# ---------------------------------------------------------------------------

LATENCY_THRESHOLD_S = 4.0
LATENCY_SAMPLE_RUNS = 5


def _print_latency_summary(label: str, elapsed: list[float]) -> None:
    n = len(elapsed)
    m = mean(elapsed)
    med = median(elapsed)
    mx = max(elapsed)
    print(f"\nLatency over {n} runs [{label}] — mean: {m:.2f}s  median: {med:.2f}s  max: {mx:.2f}s")


@pytest.mark.latency
@pytest.mark.asyncio
async def test_latency_provider_services() -> None:
    utterance = "I need to find an in-network cardiologist."
    elapsed: list[float] = []

    for _ in range(LATENCY_SAMPLE_RUNS):
        state = _make_state()
        state = _with_turn(state, user_text=utterance, ai_text=GREETING_TEXT)
        t0 = time.perf_counter()
        await run_intake(state)
        elapsed.append(time.perf_counter() - t0)

    _print_latency_summary("provider_services", elapsed)

    for run_time in elapsed:
        assert run_time < LATENCY_THRESHOLD_S, (
            f"Single run exceeded threshold: {run_time:.2f}s > {LATENCY_THRESHOLD_S}s"
        )
    assert mean(elapsed) < LATENCY_THRESHOLD_S * 0.8, (
        f"Mean latency {mean(elapsed):.2f}s exceeds 80% of threshold"
    )


@pytest.mark.latency
@pytest.mark.asyncio
async def test_latency_claim_services() -> None:
    utterance = "I want to follow up on my denied claim, reference REF-12345."
    elapsed: list[float] = []
    rec = get_recorder()

    for i in range(LATENCY_SAMPLE_RUNS):
        state = _make_state()
        state = _with_turn(state, user_text=utterance, ai_text=GREETING_TEXT)
        t0 = time.perf_counter()
        result = await run_intake(state)
        elapsed.append(time.perf_counter() - t0)
        rec.record("test_latency_claim_services", i + 1, "claim", utterance, state, result)

    _print_latency_summary("claim_services", elapsed)

    for run_time in elapsed:
        assert run_time < LATENCY_THRESHOLD_S, (
            f"Single run exceeded threshold: {run_time:.2f}s > {LATENCY_THRESHOLD_S}s"
        )
    assert mean(elapsed) < LATENCY_THRESHOLD_S * 0.8, (
        f"Mean latency {mean(elapsed):.2f}s exceeds 80% of threshold"
    )


@pytest.mark.latency
@pytest.mark.asyncio
async def test_latency_unclear_intent() -> None:
    utterance = "I have a question about my coverage."
    elapsed: list[float] = []

    for _ in range(LATENCY_SAMPLE_RUNS):
        state = _make_state()
        state = _with_turn(state, user_text=utterance, ai_text=GREETING_TEXT)
        t0 = time.perf_counter()
        await run_intake(state)
        elapsed.append(time.perf_counter() - t0)

    _print_latency_summary("unclear_intent", elapsed)

    for run_time in elapsed:
        assert run_time < LATENCY_THRESHOLD_S, (
            f"Single run exceeded threshold: {run_time:.2f}s > {LATENCY_THRESHOLD_S}s"
        )
    assert mean(elapsed) < LATENCY_THRESHOLD_S * 0.8, (
        f"Mean latency {mean(elapsed):.2f}s exceeds 80% of threshold"
    )


# ---------------------------------------------------------------------------
# SECTION 5 — Stress test
# ---------------------------------------------------------------------------

STRESS_RUNS = 20
STRESS_MIN_SUCCESS = 18  # 90% success threshold


@pytest.mark.stress
@pytest.mark.asyncio
async def test_stress_20_concurrent_provider_runs() -> None:
    utterance = "I need to find a provider in my network."

    def _fresh_state() -> dict:
        state = _make_state()
        return _with_turn(state, user_text=utterance, ai_text=GREETING_TEXT)

    coroutines = [run_intake(_fresh_state()) for _ in range(STRESS_RUNS)]

    t0 = time.perf_counter()
    results = await asyncio.gather(*coroutines, return_exceptions=True)
    total_elapsed = time.perf_counter() - t0

    successes = [r for r in results if isinstance(r, dict) and r.get("call_intent") == "provider_services"]
    failures = [r for r in results if isinstance(r, Exception) or not isinstance(r, dict)]
    wrong_intent = [r for r in results if isinstance(r, dict) and r.get("call_intent") != "provider_services"]

    failure_types = [type(f).__name__ for f in failures]
    print(
        f"\nStress test: {len(successes)}/{STRESS_RUNS} succeeded in {total_elapsed:.1f}s"
        f"\nFailures: {failure_types or 'none'}"
        f"\nWrong intent: {len(wrong_intent)}"
    )

    assert len(successes) >= STRESS_MIN_SUCCESS, (
        f"Only {len(successes)}/{STRESS_RUNS} runs succeeded "
        f"(threshold: {STRESS_MIN_SUCCESS}). Failures: {failure_types}"
    )

    for r in successes:
        assert r["next_node"] == AgentNode.VERIFICATION.value, (
            f"Successful run has wrong next_node: {r['next_node']}"
        )
        events = r.get("metadata_events", [])
        call_intent_events = [
            e
            for e in events
            if e.get("eventType") == "CallAgentField" and e.get("data", {}).get("field") == "call_intent"
        ]
        assert len(call_intent_events) >= 1, (
            "Successful run missing CallAgentField metadata event for call_intent"
        )


# ---------------------------------------------------------------------------
# SECTION 6 — Additional happy path (marker: happy)
# ---------------------------------------------------------------------------


@pytest.mark.happy
@pytest.mark.asyncio
async def test_happy_benefits_inquiry_intent() -> None:
    """'Benefits question' routes to a supported intent (provider_services or claim_services)."""
    state = _make_state()
    state = _with_turn(
        state,
        user_text="I have a question about my benefits coverage.",
        ai_text=GREETING_TEXT,
    )
    result = await run_intake(state)

    assert result.get("call_intent") in ("provider_services", "claim_services"), (
        f"Benefits inquiry must map to a supported intent, got {result.get('call_intent')!r}"
    )
    assert result["next_node"] == AgentNode.VERIFICATION.value


# ---------------------------------------------------------------------------
# SECTION 7 — Stress: concurrent claim_services (marker: stress)
# ---------------------------------------------------------------------------


@pytest.mark.stress
@pytest.mark.asyncio
async def test_stress_20_concurrent_claim_services() -> None:
    """20 concurrent claim_services classification runs — at least 18/20 must succeed."""
    utterance = "I need to check on my denied insurance claim."

    def _fresh_state() -> dict:
        state = _make_state()
        return _with_turn(state, user_text=utterance, ai_text=GREETING_TEXT)

    coroutines = [run_intake(_fresh_state()) for _ in range(STRESS_RUNS)]

    t0 = time.perf_counter()
    results = await asyncio.gather(*coroutines, return_exceptions=True)
    total_elapsed = time.perf_counter() - t0

    successes = [r for r in results if isinstance(r, dict) and r.get("call_intent") == "claim_services"]
    failures = [r for r in results if isinstance(r, Exception) or not isinstance(r, dict)]

    failure_types = [type(f).__name__ for f in failures]
    print(
        f"\nClaim stress test: {len(successes)}/{STRESS_RUNS} succeeded in {total_elapsed:.1f}s"
        f"\nFailures: {failure_types or 'none'}"
    )

    assert len(successes) >= STRESS_MIN_SUCCESS, (
        f"Only {len(successes)}/{STRESS_RUNS} claim runs succeeded "
        f"(threshold: {STRESS_MIN_SUCCESS}). Failures: {failure_types}"
    )


# Run everything:
# uv run pytest src/agent/tests/test_intake_agent.py -v -s

# Run by category:
# Happy path only
# uv run pytest src/agent/tests/test_intake_agent.py -v -s -m happy

# Non-happy path only
# uv run pytest src/agent/tests/test_intake_agent.py -v -s -m unhappy

# Latency only
# uv run pytest src/agent/tests/test_intake_agent.py -v -s -m latency

# Stress only
# uv run pytest src/agent/tests/test_intake_agent.py -v -s -m stress
