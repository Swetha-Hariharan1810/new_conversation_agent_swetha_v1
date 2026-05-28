"""
LangGraph-backed runner for latency benchmarking.

Drives a scripted conversation through the compiled graph and measures
wall-clock latency for each human turn (time from resume to next interrupt
or graph completion).
"""

from __future__ import annotations

import time
import uuid
from typing import List

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END
from langgraph.types import Command


def run_one_conversation(steps: List[str], *, tag: str) -> List[float]:
    """
    Run a scripted conversation and return per-step latencies in seconds.

    Args:
        steps: Ordered list of user utterances (human: lines from the scenario file).
        tag:   Identifier string used in the LangGraph config for tracing.

    Returns:
        List of wall-clock durations, one per human turn consumed.
    """
    from agent.app_graph import build_graph

    memory = MemorySaver()
    graph = build_graph(checkpointer=memory)

    config = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "tags": ["bench", tag],
        "metadata": {"mode": "bench", "bench_tag": tag},
    }

    # Initial invocation: no state — graph runs to first interrupt (greeting)
    state = graph.invoke({}, config=config)

    latencies: List[float] = []
    idx = 0

    while True:
        if state.get("next_node") == END:
            break

        if state.get("is_interrupt"):
            if idx >= len(steps):
                break

            user_input = steps[idx]
            idx += 1

            t0 = time.perf_counter()
            state = graph.invoke(Command(resume=user_input), config=config)
            latencies.append(time.perf_counter() - t0)

        else:
            # Non-interrupt state — advance graph without consuming a user turn
            state = graph.invoke(Command(resume=""), config=config)

    return latencies
