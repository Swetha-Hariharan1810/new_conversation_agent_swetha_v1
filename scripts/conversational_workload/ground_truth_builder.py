"""
Builds the expected ground-truth user response for a given AI message.

Strategy:
  1. Match the AI message against the per-scenario static transcript using
     keyword overlap.  When multiple candidates score within 10% of the best,
     use turn_counters to pick the correct nth occurrence (handles re-ask turns
     such as pcp_clarification_zip and pcp_clarification_fax).
  2. Fall back to slot_ground_truth (with scenario overrides) if no match
     clears the threshold.
"""

from __future__ import annotations

import re
from pathlib import Path

BASE_PATH = Path(__file__).parent / "static_transcripts"

SCENARIO_FILE_MAP = {
    "pcp_happy_path": "pcp_happy_path.txt",
    "pcp_clarification_zip": "pcp_clarification_zip.txt",
    "pcp_correction_first_name": "pcp_correction_first_name.txt",
    "pcp_correction_member_id": "pcp_correction_member_id.txt",
    "pcp_clarification_fax": "pcp_clarification_fax.txt",
    "pcp": "pcp_happy_path.txt",
}

_MATCH_THRESHOLD = 0.3
_AMBIGUITY_MARGIN = 0.1  # candidates within this fraction of best score are "tied"


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
    tokens_a = set(re.findall(r"\w+", a.lower()))
    tokens_b = set(re.findall(r"\w+", b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / max(len(tokens_a), len(tokens_b))


def build_dynamic_ground_truth(
    ai_message: str,
    entity,
    flow: str,
    scenario_tag: str = "",
    turn_counters: dict | None = None,
) -> str:
    """Return the best-matching expected user response for the given AI message."""
    if turn_counters is None:
        turn_counters = {}

    static_turns = _load_static_turns(scenario_tag)

    # Score every transcript turn against the live AI message
    scored: list[tuple[int, float, dict]] = []
    for idx, turn in enumerate(static_turns):
        score = _keyword_overlap(ai_message, turn["ai"])
        if score >= _MATCH_THRESHOLD:
            scored.append((idx, score, turn))

    if scored:
        best_score = max(s for _, s, _ in scored)
        # Collect all candidates within the ambiguity margin of the best score
        # (retain transcript order so the nth visit picks the nth occurrence)
        candidates = [
            (idx, turn)
            for idx, score, turn in scored
            if score >= best_score * (1 - _AMBIGUITY_MARGIN)
        ]
        candidates.sort(key=lambda x: x[0])  # ensure transcript order

        if len(candidates) == 1:
            return candidates[0][1]["user"]

        # Multiple close matches: use turn_counters to disambiguate
        from scripts.conversational_workload.intent_classifier import classify_ai_slot

        slot = classify_ai_slot(ai_message, flow)
        visit = turn_counters.get((scenario_tag, slot), 0)
        pick = min(visit, len(candidates) - 1)
        return candidates[pick][1]["user"]

    # Fallback: slot-based ground truth (includes scenario overrides)
    from scripts.conversational_workload.intent_classifier import classify_ai_slot
    from scripts.conversational_workload.slot_ground_truth import ground_truth_for_slot

    slot = classify_ai_slot(ai_message, flow)
    return ground_truth_for_slot(
        slot, entity, flow, scenario_tag=scenario_tag, turn_counters=turn_counters
    )
