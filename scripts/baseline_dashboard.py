"""
baseline_dashboard.py — Phase 0 baseline dashboard.

Replays the deterministic golden fixtures (``tests/golden``) and, when live
credentials are present, the live E2E suite (``tests/live_e2e``), and reports —
per run — the two rebuild surfaces we are driving to zero:

  * issue-1 surface: how each turn's spoken text was produced —
    ``generator`` turns (LLM-2 / Gemini ``recovery.md`` generation) vs.
    ``template`` turns (closed-set ``responses/turn_acts`` templates). The rebuild
    unifies these; the baseline is the split we start from.
  * issue-2 surface: ``dropped_request_count`` — how many multi-intent secondary
    requests were silently dropped (the Phase 2 metric surfaced on every run).

It changes no behavior: the golden replays are hermetic (fakes for the LLM +
storage seams, see ``tests/golden/driver.py``) and the counting is done with
non-invasive spies around ``generate_recovery_message`` (generator) and the
``turn_acts.render_*`` functions (templates).

Usage:
    uv run python -m scripts.baseline_dashboard
    uv run python -m scripts.baseline_dashboard --json      # machine-readable
    uv run python -m scripts.baseline_dashboard --live      # also run tests/live_e2e

The live suite makes real Azure OpenAI + Salesforce calls and is only attempted
with ``--live`` AND when credentials are configured; otherwise it is reported as
skipped so the dashboard still runs anywhere (CI included).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
for _p in (str(_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tests.golden.driver import RunRecord, all_fixtures, run_fixture  # noqa: E402

# turn_acts render functions whose invocation marks a template-produced turn.
_TEMPLATE_RENDERERS = (
    "render_correction_ack",
    "render_re_ask",
    "render_clarify",
    "render_unsupported_decline",
    "render_open_redirect",
    "render_stalling_ack",
    "render_multi_intent_ack",
)


@dataclass
class RunCounts:
    """Per-run baseline surfaces."""

    run_id: str
    turns: int
    generator_turns: int
    template_turns: int
    dropped_request_count: int
    latencies_ms: list = field(default_factory=list)
    # Phase 4: turn-gate understanding-decode latency, split by path
    # (metric="turn_gate_latency_ms", tag fast_path=true/false).
    gate_fast_ms: list = field(default_factory=list)
    gate_decode_ms: list = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "run": self.run_id,
            "turns": self.turns,
            "generator_turns": self.generator_turns,
            "template_turns": self.template_turns,
            "dropped_request_count": self.dropped_request_count,
            "latencies_ms": self.latencies_ms,
            "gate_fast_ms": self.gate_fast_ms,
            "gate_decode_ms": self.gate_decode_ms,
            **({"error": self.error} if self.error else {}),
        }


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (deterministic; no numpy dependency)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return round(ordered[k], 2)


@dataclass
class _Counter:
    generator: int = 0
    template: int = 0


class _SpyContext:
    """Install spies that count generator vs. template invocations for one run.

    A generator turn = one ``generate_recovery_message`` call (LLM-2, Gemini
    ``recovery.md``). A template turn = one ``turn_acts.render_*`` call. Both are
    wrapped so the real function still runs — the hermetic driver's fakes make the
    generator call return deterministic text, so behavior is unchanged.
    """

    def __init__(self) -> None:
        self.counts = _Counter()
        self._stack = ExitStack()

    def __enter__(self) -> _Counter:
        import agent.llm.response_generator as rg
        from agent.responses import turn_acts

        real_gen = rg.generate_recovery_message

        async def _spied_gen(*args, **kwargs):
            self.counts.generator += 1
            return await real_gen(*args, **kwargs)

        # Patch at the call sites (slot_manager imports the symbol lazily from the
        # module, so patching the module attribute is what the collector sees).
        self._stack.enter_context(patch.object(rg, "generate_recovery_message", _spied_gen))

        for name in _TEMPLATE_RENDERERS:
            real = getattr(turn_acts, name)

            def _make(real_fn):
                def _spied(*args, **kwargs):
                    self.counts.template += 1
                    return real_fn(*args, **kwargs)

                return _spied

            self._stack.enter_context(patch.object(turn_acts, name, _make(real)))
        return self.counts

    def __exit__(self, *exc) -> None:
        self._stack.close()


class _GateLatencyCapture(logging.Handler):
    """Collect the Phase 4 ``turn_gate_latency_ms`` records emitted during one
    replay, split by path (fast_path=true/false). Log-capture only — never
    changes gate behavior."""

    def __init__(self) -> None:
        super().__init__()
        self.fast_ms: list[float] = []
        self.decode_ms: list[float] = []

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(record, "metric", None) != "turn_gate_latency_ms":
            return
        bucket = self.fast_ms if getattr(record, "fast_path", False) else self.decode_ms
        bucket.append(float(getattr(record, "latency_ms", 0.0)))

    def __enter__(self) -> "_GateLatencyCapture":
        lg = logging.getLogger("agent.orchestration.turn_gate")
        self._logger, self._prev_level = lg, lg.level
        lg.setLevel(logging.DEBUG)
        lg.addHandler(self)
        return self

    def __exit__(self, *exc) -> None:
        self._logger.removeHandler(self)
        self._logger.setLevel(self._prev_level)


async def _count_golden_run(fixture: dict) -> RunCounts:
    run_id = fixture.get("id", "?")
    # A few fixtures probe the shared collector directly (driver "_collect_slot")
    # and are driven by their own bespoke test, not the single-agent replay.
    if fixture.get("driver") == "_collect_slot":
        return RunCounts(
            run_id=run_id,
            turns=0,
            generator_turns=0,
            template_turns=0,
            dropped_request_count=0,
            error="collector-probe fixture (driven by its own test)",
        )
    try:
        with _SpyContext() as counts, _GateLatencyCapture() as gate:
            record: RunRecord = await run_fixture(fixture, print_latency=False)
        return RunCounts(
            run_id=run_id,
            turns=len(record.turns),
            generator_turns=counts.generator,
            template_turns=counts.template,
            dropped_request_count=record.dropped_request_count,
            latencies_ms=list(record.latencies_ms),
            gate_fast_ms=gate.fast_ms,
            gate_decode_ms=gate.decode_ms,
        )
    except Exception as exc:  # a fixture that this single-agent driver can't drive
        return RunCounts(
            run_id=run_id,
            turns=0,
            generator_turns=0,
            template_turns=0,
            dropped_request_count=0,
            error=f"{type(exc).__name__}: {exc}",
        )


async def collect_golden() -> list[RunCounts]:
    results: list[RunCounts] = []
    for fixture in all_fixtures():
        results.append(await _count_golden_run(fixture))
    return results


def _live_credentials_present() -> bool:
    try:
        from agent.llm.config import Config
    except Exception:
        return False
    return bool(Config.AZURE_OPENAI_API_KEY and Config.AZURE_OPENAI_ENDPOINT and Config.SF_CLIENT_ID)


@dataclass
class DashboardReport:
    golden: list[RunCounts] = field(default_factory=list)
    live_status: str = "skipped"  # "skipped" | "ran" | "unavailable"
    live_note: str = ""

    def totals(self) -> dict:
        ok = [r for r in self.golden if not r.error]
        all_latencies = [ms for r in ok for ms in r.latencies_ms]
        gate_fast = [ms for r in ok for ms in r.gate_fast_ms]
        gate_decode = [ms for r in ok for ms in r.gate_decode_ms]
        return {
            "golden_runs": len(self.golden),
            "golden_runs_counted": len(ok),
            "generator_turns": sum(r.generator_turns for r in ok),
            "template_turns": sum(r.template_turns for r in ok),
            "dropped_request_count": sum(r.dropped_request_count for r in ok),
            "turn_latency_ms_p50": _percentile(all_latencies, 50),
            "turn_latency_ms_p95": _percentile(all_latencies, 95),
            # Phase 4: understanding-decode cost at the turn gate, per path.
            "gate_fast_path_turns": len(gate_fast),
            "gate_fast_path_ms_p50": _percentile(gate_fast, 50),
            "gate_fast_path_ms_p95": _percentile(gate_fast, 95),
            "gate_decode_turns": len(gate_decode),
            "gate_decode_ms_p50": _percentile(gate_decode, 50),
            "gate_decode_ms_p95": _percentile(gate_decode, 95),
        }

    def to_dict(self) -> dict:
        return {
            "golden": [r.to_dict() for r in self.golden],
            "totals": self.totals(),
            "live": {"status": self.live_status, "note": self.live_note},
        }


def _print_table(report: DashboardReport) -> None:
    rows = report.golden
    name_w = max((len(r.run_id) for r in rows), default=8) + 2
    header = f"{'run':<{name_w}}{'turns':>7}{'generator':>11}{'template':>10}{'dropped':>9}"
    print("=" * len(header))
    print("PHASE 0 BASELINE DASHBOARD — tests/golden (hermetic replay)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        if r.error:
            print(f"{r.run_id:<{name_w}}({r.error})")
            continue
        print(
            f"{r.run_id:<{name_w}}{r.turns:>7}{r.generator_turns:>11}"
            f"{r.template_turns:>10}{r.dropped_request_count:>9}"
        )
    print("-" * len(header))
    t = report.totals()
    print(
        f"{'TOTAL':<{name_w}}{'':>7}{t['generator_turns']:>11}"
        f"{t['template_turns']:>10}{t['dropped_request_count']:>9}"
    )
    print(
        f"\nissue-1 surface (generator vs template turns): "
        f"{t['generator_turns']} generator / {t['template_turns']} template"
    )
    print(f"issue-2 surface (dropped_request_count): {t['dropped_request_count']}")
    print(
        f"turn latency (golden, deterministic): "
        f"p50={t['turn_latency_ms_p50']}ms p95={t['turn_latency_ms_p95']}ms "
        f"(live_e2e p50/p95 validated separately — needs credentials)"
    )
    print(
        f"turn-gate understanding decode: "
        f"fast-path n={t['gate_fast_path_turns']} "
        f"p50={t['gate_fast_path_ms_p50']}ms p95={t['gate_fast_path_ms_p95']}ms | "
        f"decode n={t['gate_decode_turns']} "
        f"p50={t['gate_decode_ms_p50']}ms p95={t['gate_decode_ms_p95']}ms "
        f"(acceptance: fast path < 1ms; one decode per turn)"
    )
    print(f"\nlive_e2e: {report.live_status}" + (f" — {report.live_note}" if report.live_note else ""))


async def _amain(args: argparse.Namespace) -> int:
    report = DashboardReport(golden=await collect_golden())

    if args.live:
        if _live_credentials_present():
            report.live_status = "unavailable"
            report.live_note = (
                "credentials present but the live suite is run via "
                "`python -m tests.live_e2e.run_live_tests` (real network/cost); "
                "not driven inline by this dashboard"
            )
        else:
            report.live_status = "skipped"
            report.live_note = "no Azure/Salesforce credentials configured in this environment"
    else:
        report.live_status = "skipped"
        report.live_note = "pass --live to attempt the live suite"

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        _print_table(report)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.baseline_dashboard",
        description="Phase 0 baseline dashboard (generator vs template turns, dropped_request_count).",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--live", action="store_true", help="Also attempt the live E2E suite.")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_amain(args)))


if __name__ == "__main__":
    main()
