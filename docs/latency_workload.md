# scripts/latency_workload/main.py

Purpose
- CLI benchmark that measures per-node and LLM-only latency for conversation graph scenarios.
- Produces markdown and JSON reports with p50/p95/p99/avg/std and SLA checks.

Key commands / usage
- Entrypoint: `python -m scripts.latency_workload.main`
- Important args: --iterations, --threshold-sec, --scenarios-path, --sla-metric

Output
- Writes bench-metrics.json by default and produces console-markdown tables; integrates with GitHub Actions outputs if `GITHUB_OUTPUT` is set.

Source
- https://github.com/Swetha-Hariharan1810/new_conversation_agent_swetha_v1/blob/08f9ac4e873a1501f04fa095e34f2f33617ca417/scripts/latency_workload/main.py
