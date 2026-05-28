"""
Output formatting (Markdown table + JSON) and CI reporting.
"""

import json
from pathlib import Path
from typing import Dict

from scripts.latency_workload.git_output import set_github_outputs


def print_markdown_table(
    title: str,
    metrics: Dict[str, float],
    step_count: int,
    iterations: int,
    threshold: float,
) -> None:
    print(f"\n## {title}")
    print(f"Iterations: {iterations} | Steps: {step_count} | Threshold: {threshold:.3f}s")
    print("\n| Metric | Value (s) |\n|---|---:|")
    for k, v in metrics.items():
        print(f"| {k} | {v:.3f} |")
    print()


def save_json(output_path: str, data: Dict) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")


def report_to_github(overall: Dict[str, float], json_path: str, step_count: int) -> None:
    set_github_outputs(
        {
            "avg": f"{overall['avg']:.6f}",
            "p50": f"{overall['p50']:.6f}",
            "p95": f"{overall['p95']:.6f}",
            "std_dev": f"{overall['std_dev']:.6f}",
            "json_path": str(json_path),
            "step_count": str(step_count),
        }
    )
