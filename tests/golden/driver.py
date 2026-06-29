"""
driver.py — Deterministic, hermetic driver for golden multi-intent fixtures.

Phase 0 goal: lock the *current (broken)* behavior under test so every later
phase is measurable. To do that without secrets or network we replace the two
external seams every agent touches:

  1. The LLM seam  — each agent calls ``get_extraction_llm()`` /
     ``get_follow_up_llm()`` and then ``.with_structured_output(Model).ainvoke()``.
     We patch those getters to return a ``FakeLLM`` whose structured output is
     *replayed from the fixture*, one ``WorkerResult`` per member turn. This is
     the deterministic stand-in for LLM-1 and, crucially, it documents exactly
     what the single-intent ``WorkerResult`` schema can and cannot represent —
     which IS the defect (a second intent in the same utterance is simply not
     in ``extracted``).
  2. The storage seam — handlers lazily ``from agent.storage.tools import X`` and
     call ``X.ainvoke({...})``. We patch those tool objects with ``FakeTool``s
     that record their call args (so we can assert "dispatched on the disputed
     ZIP") and return success.

The driver advances one agent node at a time (the defect is conversation-wide,
so each fixture targets the agent where a given multi-intent turn lands),
merging the agent's returned update dict back into state with the same reducer
semantics LangGraph uses: ``messages`` appends, everything else is
last-write-wins. A wall-clock probe wraps every turn — the seed of the
latency bench Phase 3/4 will assert against.

Nothing here imports the compiled graph; agents are invoked as plain callables,
so no checkpointer, env var, or warm-up is required.
"""

from __future__ import annotations

import json
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator
from unittest.mock import patch

from agent.llm.schema import FollowUpResult, WorkerResult

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


# ──────────────────────────────────────────────────────────────────────────────
# Fixture loading
# ──────────────────────────────────────────────────────────────────────────────


def load_fixture(name: str) -> dict:
    """Load a golden fixture JSON by file name (with or without .json)."""
    fname = name if name.endswith(".json") else f"{name}.json"
    return json.loads((FIXTURES_DIR / fname).read_text())


def all_fixtures() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(FIXTURES_DIR.glob("*.json"))]


def build_result(extraction: dict | None, *, schema: str = "worker"):
    """Turn a fixture ``extraction`` block into the structured object LLM-1
    would have returned. Pydantic coerces the string enum values
    (event_type / guard / follow_up_intent) for us."""
    extraction = dict(extraction or {})
    if schema == "follow_up":
        return FollowUpResult(**extraction)
    return WorkerResult(**extraction)


# ──────────────────────────────────────────────────────────────────────────────
# Fakes — LLM seam
# ──────────────────────────────────────────────────────────────────────────────


class _FakeStructured:
    """What ``llm.with_structured_output(Model)`` returns. ``ainvoke`` replays
    the next scripted result from the shared queue."""

    def __init__(self, queue: list) -> None:
        self._queue = queue

    async def ainvoke(self, _messages, **_kw):
        if not self._queue:
            raise AssertionError(
                "FakeLLM: structured-output queue is empty — the agent made more "
                "LLM-1 calls this turn than the fixture scripted. Re-check the fixture."
            )
        return self._queue.pop(0)


class FakeLLM:
    """Stand-in for every LLM getter.

    * ``with_structured_output(Model).ainvoke()`` → next queued result (LLM-1).
    * ``ainvoke()`` → a fixed ``.content`` string (LLM-2 recovery generation).
      The exact text is irrelevant to baseline assertions, which check state,
      tool-call args, and whether a side intent was acknowledged at all.
    """

    def __init__(self, gen_text: str = "[[deterministic-recovery-message]]") -> None:
        self.queue: list = []
        self.gen_text = gen_text

    def enqueue(self, result) -> None:
        self.queue.append(result)

    def with_structured_output(self, _model, **_kw) -> _FakeStructured:
        return _FakeStructured(self.queue)

    async def ainvoke(self, _messages, **_kw):
        return SimpleNamespace(content=self.gen_text)


# ──────────────────────────────────────────────────────────────────────────────
# Fakes — storage seam
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ToolRecorder:
    """Records every faked storage-tool call so tests can assert on args
    (e.g. the ZIP a provider list was dispatched against)."""

    calls: list[tuple[str, dict]] = field(default_factory=list)

    def for_tool(self, name: str) -> list[dict]:
        return [payload for (tool_name, payload) in self.calls if tool_name == name]

    def count(self, name: str) -> int:
        return len(self.for_tool(name))


class FakeTool:
    """Mimics a LangChain ``@tool`` object: callers invoke ``.ainvoke({...})``."""

    def __init__(self, name: str, recorder: ToolRecorder, return_value: Any = True) -> None:
        self._name = name
        self._recorder = recorder
        self._return_value = return_value

    async def ainvoke(self, payload: dict, **_kw):
        self._recorder.calls.append((self._name, dict(payload)))
        return self._return_value


# Storage tools that the targeted agents reach for, by attribute name on
# ``agent.storage.tools``. Patched as module attributes so the handlers'
# lazy ``from agent.storage.tools import X`` picks up the fake.
_PATCHED_TOOLS = ("dispatch_provider_list", "update_member_contact", "update_zip_code")

# LLM getter sites — patched where they are *used* (each agent module imports the
# getter by value), so the source ``agent.llm.config`` indirection is bypassed.
_LLM_GETTER_SITES = (
    "agent.agents.delivery_management.agent.get_extraction_llm",
    "agent.agents.provider_search.agent.get_extraction_llm",
    "agent.agents.follow_up.agent.get_follow_up_llm",
    "agent.llm.response_generator.get_generation_llm",
)


@contextmanager
def deterministic_env(fake_llm: FakeLLM, recorder: ToolRecorder) -> Iterator[None]:
    """Patch the LLM getters and storage tools for the duration of a scenario."""
    with ExitStack() as stack:
        for site in _LLM_GETTER_SITES:
            stack.enter_context(patch(site, lambda *_a, **_k: fake_llm))
        for tool_name in _PATCHED_TOOLS:
            stack.enter_context(
                patch(f"agent.storage.tools.{tool_name}", FakeTool(tool_name, recorder))
            )
        yield


# ──────────────────────────────────────────────────────────────────────────────
# State reducer (LangGraph semantics, hermetic)
# ──────────────────────────────────────────────────────────────────────────────


def _append_messages(existing: list, incoming) -> list:
    msgs = list(existing or [])
    if incoming is None:
        return msgs
    if isinstance(incoming, list):
        msgs.extend(incoming)
    else:
        msgs.append(incoming)
    return msgs


def merge_state(state: dict, updates: dict) -> dict:
    """Apply an agent's returned update dict to state.

    Matches the reducers in ``agent.state.State``: ``messages`` uses
    ``add_messages`` (append); every other key is last-write-wins.
    """
    new = dict(state)
    for key, value in (updates or {}).items():
        if key == "messages":
            new["messages"] = _append_messages(new.get("messages"), value)
        else:
            new[key] = value
    return new


# ──────────────────────────────────────────────────────────────────────────────
# Transcript / run record
# ──────────────────────────────────────────────────────────────────────────────


def _last_ai(messages: list) -> str:
    for m in reversed(messages or []):
        role = m.get("role") if isinstance(m, dict) else getattr(m, "type", "")
        if role in ("assistant", "ai"):
            content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
            return (content or "").strip()
    return ""


@dataclass
class TurnRecord:
    index: int
    user: str
    ai: str
    awaiting_slot: str
    wall_clock_s: float
    updates: dict


@dataclass
class RunRecord:
    fixture_id: str
    agent: str
    final_state: dict = field(default_factory=dict)
    turns: list[TurnRecord] = field(default_factory=list)
    recorder: ToolRecorder = field(default_factory=ToolRecorder)

    @property
    def latencies_ms(self) -> list[float]:
        return [round(t.wall_clock_s * 1000, 2) for t in self.turns]

    def last_ai(self) -> str:
        return self.turns[-1].ai if self.turns else ""


# ──────────────────────────────────────────────────────────────────────────────
# Agent registry — fixture "driver" string → callable
# ──────────────────────────────────────────────────────────────────────────────


def _agent_callable(name: str) -> Callable:
    if name == "delivery_management_agent":
        from agent.agents.delivery_management.agent import delivery_management_agent

        return delivery_management_agent
    if name == "provider_search_agent":
        from agent.agents.provider_search.agent import provider_search_agent

        return provider_search_agent
    if name == "follow_up_agent":
        from agent.agents.follow_up.agent import follow_up_agent

        return follow_up_agent
    raise ValueError(f"golden driver: unknown agent {name!r}")


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────


async def run_fixture(fixture: dict, *, print_latency: bool = True) -> RunRecord:
    """Drive a single agent through the fixture's scripted turns, deterministically.

    Returns a ``RunRecord`` carrying the per-turn transcript, final merged state,
    per-turn wall-clock latency, and every faked storage-tool call.
    """
    agent_name = fixture["driver"]
    schema = fixture.get("schema", "worker")
    agent = _agent_callable(agent_name)

    fake_llm = FakeLLM()
    record = RunRecord(fixture_id=fixture["id"], agent=agent_name)
    state = dict(fixture["initial_state"])

    with deterministic_env(fake_llm, record.recorder):
        for i, turn in enumerate(fixture["turns"]):
            user = turn["user"]
            # Simulate human_node delivering the member's utterance.
            state = merge_state(
                state,
                {"messages": {"role": "user", "content": user}, "is_interrupt": False},
            )
            # Script LLM-1's single decode for this turn.
            fake_llm.enqueue(build_result(turn.get("extraction"), schema=schema))

            t0 = time.perf_counter()
            updates = await agent(state)
            dt = time.perf_counter() - t0

            state = merge_state(state, updates)
            ai = _last_ai(state.get("messages", []))
            record.turns.append(
                TurnRecord(
                    index=i,
                    user=user,
                    ai=ai,
                    awaiting_slot=state.get("awaiting_slot", ""),
                    wall_clock_s=dt,
                    updates=updates,
                )
            )
            if print_latency:
                print(
                    f"[golden-latency] {fixture['id']} turn {i} "
                    f"({agent_name}) wall_clock={dt * 1000:.1f}ms "
                    f"awaiting_slot={state.get('awaiting_slot', '')!r}"
                )

    record.final_state = state
    return record
