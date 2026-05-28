"""
Write GitHub Actions workflow outputs for CI integration.
"""

import os

from agent.logger import get_logger

logger = get_logger(__name__)


def set_github_outputs(values: dict) -> None:
    path = os.getenv("GITHUB_OUTPUT")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            for k, v in values.items():
                fh.write(f"{k}={v}\n")
    except Exception as exc:
        logger.info(f"Warning: Could not write GitHub outputs: {exc}")
