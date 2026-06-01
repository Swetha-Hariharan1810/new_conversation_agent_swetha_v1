#!/usr/bin/env python3
# Run with: python src/agent/scripts/test_warmup.py
# To make executable: chmod +x src/agent/scripts/test_warmup.py

import asyncio
import pathlib
import sys

# Add src/ to sys.path so agent modules can be imported without installation.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))


async def main() -> None:
    # Step 1 — warm prompt cache if available
    try:
        from agent.utils import warm_prompt_cache

        warm_prompt_cache()
    except ImportError:
        print("Note: warm_prompt_cache not found in agent.utils — skipping.")

    # Step 2 — warm all LLM connections
    from agent.app_graph import warm_llm_connections

    await warm_llm_connections()

    # Step 3 — measure warm-path latency
    import time

    from agent.llm.config import get_extraction_llm
    from agent.llm.extractor import build_worker_input
    from agent.llm.schema import WorkerResult
    from agent.utils import build_extraction_prompt_core

    messages = build_worker_input(
        build_extraction_prompt_core("extraction/benefits.md"),
        awaiting_slot="care_coach_response",
        last_agent_message="Would you like Care Coach details?",
        last_user_message="yes",
        attempt=0,
    )
    llm = get_extraction_llm()

    # cold measurement (before warm-up — but warm-up already ran, so this
    # measures the WARM path, which is what we want to confirm is fast)
    t0 = time.perf_counter()
    result = await llm.with_structured_output(WorkerResult).ainvoke(messages)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    print(f"Warm-path latency: {elapsed_ms} ms")
    print(f"Extracted: {result.extracted}")
    print(f"Guard:     {result.guard}")

    # At the bottom of main(), after the first timed call, add:

    print("\n--- 5-call consecutive probe (true warm baseline) ---")
    times = []
    for i in range(5):
        t0 = time.perf_counter()
        await llm.with_structured_output(WorkerResult).ainvoke(messages)
        ms = round((time.perf_counter() - t0) * 1000, 1)
        times.append(ms)
        print(f"  call {i + 1}: {ms} ms")

    print(f"\n  min={min(times)}ms  max={max(times)}ms  avg={round(sum(times) / len(times), 1)}ms")
    print(f"  variance={round(max(times) - min(times), 1)}ms")


asyncio.run(main())
