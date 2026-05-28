#!/usr/bin/env python3
"""
CLI entry point for the LangGraph latency benchmark.

Usage:
    python -m scripts.latency_workload.main \\
        --iterations 3 \\
        --threshold-sec 2.5 \\
        --scenarios-path scripts/latency_workload/scenarios
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

from agent.logger import get_logger
from scripts.latency_workload.data_loader import load_scenarios
from scripts.latency_workload.metrics import compute_metrics
from scripts.latency_workload.git_output import report_to_github
from scripts.latency_workload.reporter import print_markdown_table, save_json
from scripts.latency_workload.runner import _run_one_conversation_async

logger = get_logger(__name__)


def _ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


async def _run_all_scenarios_async(
    scenarios: Dict[str, List[str]],
    iterations: int,
    parallel: bool,
) -> Dict[str, List[float]]:
    per_demo: Dict[str, List[float]] = {name: [] for name in scenarios}

    # parallel mode not supported in async context, ignore flag
    for i in range(iterations):
        for name, steps in scenarios.items():
            tag = f"{name}-iter-{i + 1}"
            latencies = await _run_one_conversation_async(steps, tag=tag)  # await directly
            per_demo[name].extend(latencies)

    return per_demo


def _get_config(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LangGraph latency benchmark")
    parser.add_argument("--iterations", type=int, default=int(os.getenv("ITERATIONS", "1")))
    parser.add_argument(
        "--threshold-sec", type=float, default=float(os.getenv("THRESHOLD_SEC", "2.5"))
    )
    parser.add_argument("--output", default="bench-metrics.json")
    parser.add_argument(
        "--scenarios-path",
        default=str(Path(__file__).parent / "scenarios"),
    )
    parser.add_argument("--parallel", action="store_true", default=False)
    return parser.parse_args(argv)


async def main_async(argv=None) -> int:
    args = _get_config(argv)

    output_dir_env = os.getenv("BENCH_OUTPUT_DIR")
    if output_dir_env:
        _ensure_dir(output_dir_env)
        args.output = str(Path(output_dir_env) / Path(args.output).name)

    scenarios = load_scenarios(args.scenarios_path)
    if not scenarios:
        logger.info("No scenario files found in %s", args.scenarios_path)
        return 1

    logger.info("Loaded %d scenario(s): %s", len(scenarios), ", ".join(scenarios.keys()))
    logger.info("Running %d iteration(s) per scenario", args.iterations)

    t0 = time.perf_counter()
    per_demo_latencies = await _run_all_scenarios_async(scenarios, args.iterations, args.parallel)
    all_latencies = [lat for vals in per_demo_latencies.values() for lat in vals]
    duration = time.perf_counter() - t0

    overall = compute_metrics(all_latencies)
    per_demo_metrics = {name: compute_metrics(vals) for name, vals in per_demo_latencies.items()}

    payload = {
        "overall": {**overall, "step_count": len(all_latencies)},
        "per_demo": per_demo_metrics,
        "meta": {
            "iterations": args.iterations,
            "scenarios": list(scenarios.keys()),
            "threshold_sec": args.threshold_sec,
            "duration_sec": round(duration, 3),
        },
    }

    save_json(args.output, payload)
    print_markdown_table(
        title="LangGraph Latency Bench (PCP per-step)",
        metrics=overall,
        step_count=len(all_latencies),
        iterations=args.iterations,
        threshold=args.threshold_sec,
    )
    report_to_github(overall, args.output, len(all_latencies))

    if overall["avg"] > args.threshold_sec:
        logger.info(
            "SLA violated: avg %.3fs > threshold %.3fs",
            overall["avg"],
            args.threshold_sec,
        )
        return 1

    logger.info("Completed in %.2fs with %d total steps.", duration, len(all_latencies))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main_async()))