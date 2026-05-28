"""
Builds the expected ground-truth user response for a given AI message.

Strategy:
  1. Match the AI message against the per-scenario static transcript
     using keyword similarity (no LLM call — deterministic and fast).
  2. Fall back to the slot-map ground truth if no transcript match is found.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

BASE_PATH = Path(__file__).parent / "static_transcripts"

SCENARIO_FILE_MAP = {
    "pcp_happy_path": "pcp_happy_path.txt",
    "pcp_clarification_zip": "pcp_clarification_zip.txt",
    "pcp_correction_first_name": "pcp_correction_first_name.txt",
    "pcp_correction_member_id": "pcp_correction_member_id.txt",
    "pcp_clarification_fax": "pcp_clarification_fax.txt",
    # Default fallback for unknown scenario tags
    "pcp": "pcp_happy_path.txt",
}


def _load_static_turns(scenario_tag: str) -> list[dict]:
    filename = SCENARIO_FILE_MAP.get(scenario_tag)
    if not filename:
        return []
    path = BASE_PATH / filename
    if not path.exists():
        return []

    turns = []
    ai_msg = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if re.match(r"(?i)^ai\s*:", line):
            ai_msg = re.sub(r"(?i)^ai\s*:\s*", "", line)
        elif re.match(r"(?i)^(human|user)\s*:", line):
            user_msg = re.sub(r"(?i)^(human|user)\s*:\s*", "", line)
            if ai_msg is not None:
                turns.append({"ai": ai_msg, "user": user_msg})
                ai_msg = None
    return turns


def _keyword_overlap(a: str, b: str) -> float:
    """Simple token overlap ratio between two strings."""
    tokens_a = set(re.findall(r"\w+", a.lower()))
    tokens_b = set(re.findall(r"\w+", b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / max(len(tokens_a), len(tokens_b))


def build_dynamic_ground_truth(ai_message: str, entity, flow: str, scenario_tag: str = "pcp_happy_path") -> str:
    """Return the best-matching expected user response for the given AI message."""
    static_turns = _load_static_turns(scenario_tag)

    best_score = 0.0
    best_user = ""
    for turn in static_turns:
        score = _keyword_overlap(ai_message, turn["ai"])
        if score > best_score:
            best_score = score
            best_user = turn["user"]

    if best_score >= 0.3 and best_user:
        return best_user

    # Fallback: use slot-map ground truth
    from scripts.conversational_workload.intent_classifier import classify_ai_slot
    from scripts.conversational_workload.slot_ground_truth import ground_truth_for_slot

    slot = classify_ai_slot(ai_message, flow)
    return ground_truth_for_slot(slot, entity, flow)
