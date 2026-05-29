"""
LangGraph-backed runner for latency benchmarking.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import List

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END
from langgraph.types import Command


async def _run_one_conversation_async(steps: List[str], *, tag: str) -> List[float]:
    from agent.app_graph import build_graph

    memory = MemorySaver()
    graph = build_graph(checkpointer=memory)

    config = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "tags": ["bench", tag],
        "metadata": {"mode": "bench", "bench_tag": tag},
    }

    state = await graph.ainvoke({}, config=config)

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
            state = await graph.ainvoke(Command(resume=user_input), config=config)
            latencies.append(time.perf_counter() - t0)

        else:
            state = await graph.ainvoke(Command(resume=""), config=config)

    return latencies


def run_one_conversation(steps: List[str], *, tag: str) -> List[float]:
    """Sync wrapper so main.py and ThreadPoolExecutor callers are unchanged."""
    return asyncio.run(_run_one_conversation_async(steps, tag=tag))
