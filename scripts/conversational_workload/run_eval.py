#!/usr/bin/env python3
"""
Entry point for the conversational evaluation benchmark.

Runs all 5 PCP scenarios, writes per-conversation reports and a
summary JSON to scripts/conversational_workload/results/.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import aiofiles

from scripts.conversational_workload.runner import run_evaluation_async

ENTITY_DATA = {
    "first_name": "Emily",
    "last_name": "Carter",
    "member_id": "M907503",
    "date_of_birth": "1988-04-12",
    "subscriber_type": "myself",
    "provider_type": "Primary Care Physician",
    "zip_code": "12139",
    "fax_number": "6175554199",
}

SCENARIOS = [
    "pcp_happy_path",
    "pcp_clarification_zip",
    "pcp_correction_first_name",
    "pcp_correction_member_id",
    "pcp_clarification_fax",
]

SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.5"))


async def main_async() -> int:
    num_runs = int(os.getenv("ITERATIONS", "1"))

    repo_root = Path(__file__).resolve().parents[2]
    results_dir = Path(os.getenv("RESULTS_DIR", str(repo_root / "scripts/conversational_workload/results")))
    results_dir.mkdir(parents=True, exist_ok=True)

    report_file = os.getenv("REPORT_FILE", "conversation_eval_report.json")

    print("\n" + "=" * 42)
    print("Starting Conversational Evaluation Benchmark")
    print(f"Scenarios: {', '.join(SCENARIOS)}")
    print(f"Runs per scenario: {num_runs}")
    print("=" * 42 + "\n")

    overall_summary: dict = {}

    for scenario_tag in SCENARIOS:
        all_reports = []
        for run_idx in range(num_runs):
            print(f"  [{scenario_tag}] run {run_idx + 1}/{num_runs}")
            report = await run_evaluation_async(
                entity_data=ENTITY_DATA, flow="pcp", scenario_tag=scenario_tag
            )
            print(f"    score={report.final_score}  turns={len(report.turns)}  completed={report.completed}")
            all_reports.append(report.model_dump())

        avg_score = round(sum(r["final_score"] for r in all_reports) / num_runs, 2)
        overall_summary[scenario_tag] = {
            "total_runs": num_runs,
            "average_final_score": avg_score,
            "runs": all_reports,
        }
        print(f"  {scenario_tag}: avg_score={avg_score}\n")

    summary_path = results_dir / report_file

    async with aiofiles.open(summary_path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(overall_summary, indent=2))

    # Per-scenario pass/fail
    failures = [
        (tag, data["average_final_score"])
        for tag, data in overall_summary.items()
        if data["average_final_score"] < SCORE_THRESHOLD
    ]
    if failures:
        print("\nFAILED scenarios (below threshold {:.2f}):".format(SCORE_THRESHOLD))
        for tag, score in failures:
            print(f"  {tag}: {score:.2f}")
        return 1

    print(f"\nAll scenarios passed (threshold={SCORE_THRESHOLD}).")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main_async()))
