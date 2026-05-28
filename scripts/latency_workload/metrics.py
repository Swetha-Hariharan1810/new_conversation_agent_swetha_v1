"""
Metrics computation (p50, p95, avg, std-dev) for performance benchmarking.
Uses only the Python standard library — no numpy dependency.
"""

import math
import statistics
from typing import Dict, List


def compute_metrics(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "avg": 0.0, "std_dev": 0.0, "min": 0.0, "max": 0.0}

    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def percentile(p: float) -> float:
        if n == 1:
            return sorted_vals[0]
        idx = (p / 100.0) * (n - 1)
        lo = int(math.floor(idx))
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])

    return {
        "p50": percentile(50),
        "p95": percentile(95),
        "avg": statistics.mean(values),
        "std_dev": statistics.pstdev(values),
        "min": sorted_vals[0],
        "max": sorted_vals[-1],
    }
