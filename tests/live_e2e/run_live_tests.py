"""
run_live_tests.py — Canonical CLI entry point for the live E2E suite.

Usage (from the repo root):
    python -m tests.live_e2e.run_live_tests
    python -m tests.live_e2e.run_live_tests --only pcp_happy_path_fax,claim_happy_path
    python -m tests.live_e2e.run_live_tests --skip-mutating
    python -m tests.live_e2e.run_live_tests --results-dir /tmp/live_results
    python -m tests.live_e2e.run_live_tests --list

Scenarios run SEQUENTIALLY — they share Salesforce fixture data and must not
be parallelized. Exits non-zero on any failure (and 2 on preflight failure).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import tests.live_e2e  # noqa: F401 — ensures src/ is on sys.path
from tests.live_e2e.harness import Scenario, ScenarioResult, run_scenario
from tests.live_e2e.preflight import PreflightError, restore_contacts, run_preflight
from tests.live_e2e.scenarios import SCENARIOS, SCENARIOS_BY_NAME

logger = logging.getLogger("live_e2e")

DEFAULT_RESULTS_DIR = Path(__file__).parent / "results"


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    # Quieten chatty third-party loggers; keep agent + harness at INFO.
    for noisy in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _select(only: str | None, skip_mutating: bool) -> list[Scenario]:
    if only:
        names = [n.strip() for n in only.split(",") if n.strip()]
        unknown = [n for n in names if n not in SCENARIOS_BY_NAME]
        if unknown:
            raise SystemExit(
                f"Unknown scenario name(s): {', '.join(unknown)}\n"
                f"Available: {', '.join(SCENARIOS_BY_NAME)}"
            )
        selected = [SCENARIOS_BY_NAME[n] for n in names]
    else:
        selected = list(SCENARIOS)
    if skip_mutating:
        skipped = [s.name for s in selected if s.mutating]
        if skipped:
            logger.info("--skip-mutating: excluding %s", ", ".join(skipped))
        selected = [s for s in selected if not s.mutating]
    return selected


def _print_summary(results: list[ScenarioResult]) -> None:
    name_w = max((len(r.name) for r in results), default=8) + 2
    header = (
        f"{'scenario':<{name_w}}{'result':<10}{'turns':<7}{'duration':<10}escalation_reason"
    )
    print("\n" + "=" * len(header))
    print("LIVE E2E SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        if r.passed and r.flaky:
            status = "PASS*"  # passed on retry — flaky
        print(
            f"{r.name:<{name_w}}{status:<10}{r.turns:<7}{r.duration_s:<10.1f}"
            f"{r.escalation_reason or '-'}"
        )
    print("-" * len(header))
    passed = sum(1 for r in results if r.passed)
    flaky = sum(1 for r in results if r.flaky)
    print(f"{passed}/{len(results)} passed" + (f" ({flaky} flaky — passed on retry)" if flaky else ""))
    for r in results:
        if not r.passed:
            print(f"\n--- FAILURES: {r.name} ---")
            for f in r.failures:
                print(f"  * {f}")


async def _amain(args: argparse.Namespace) -> int:
    results_dir = Path(args.results_dir)
    scenarios = _select(args.only, args.skip_mutating)
    if not scenarios:
        print("Nothing to run.")
        return 0

    print(f"Running {len(scenarios)} scenario(s) sequentially against LIVE services.")
    print("This makes real Azure OpenAI and Salesforce calls — expect cost and latency.\n")

    try:
        snapshot = await run_preflight(warm=True)
    except PreflightError as exc:
        print(f"\nPREFLIGHT FAILED — no scenarios were run.\n\n{exc}", file=sys.stderr)
        return 2

    results: list[ScenarioResult] = []
    for scenario in scenarios:
        logger.info("=" * 70)
        logger.info("SCENARIO: %s (%s)%s", scenario.name, scenario.flow,
                     " [mutating]" if scenario.mutating else "")
        logger.info("=" * 70)
        try:
            result = await run_scenario(scenario, results_dir)
        finally:
            if scenario.mutating:
                # Restore snapshotted SF contact fields even if the scenario failed.
                await restore_contacts(snapshot)
        results.append(result)

    _print_summary(results)
    return 0 if all(r.passed for r in results) else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m tests.live_e2e.run_live_tests",
        description="Live end-to-end conversation tests (real LLM + Salesforce).",
    )
    parser.add_argument(
        "--only",
        help="Comma-separated scenario names to run (default: all).",
    )
    parser.add_argument(
        "--skip-mutating",
        action="store_true",
        help="Exclude scenarios that write to Salesforce "
        "(pcp_zip_update, pcp_fax_update, claim_email_change_on_upload, "
        "email_change_loop_in_notification).",
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help=f"Directory for per-scenario JSON results (default: {DEFAULT_RESULTS_DIR}).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List scenario names and exit.",
    )
    args = parser.parse_args()

    if args.list:
        for s in SCENARIOS:
            flags = []
            if s.mutating:
                flags.append("mutating")
            if s.retries:
                flags.append(f"retries={s.retries}")
            print(f"{s.name:<40} flow={s.flow:<6} {' '.join(flags)}")
        return

    _configure_logging()
    raise SystemExit(asyncio.run(_amain(args)))


if __name__ == "__main__":
    main()
