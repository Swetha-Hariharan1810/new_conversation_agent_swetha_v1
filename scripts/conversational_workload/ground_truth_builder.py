"""
ground_truth_builder.py — Authoritative ground truth from static transcripts.

This module's ONLY job is to return the expected human line from the static
transcript for a given AI message.  It is the authoritative reference for
what a correct user response looks like at each point in the conversation.

IT MUST NEVER:
  - Call simulate_user_response or any simulator code
  - Generate responses via LLM
  - Share any data with the simulator

MATCHING STRATEGY:
  1. Load the static transcript for the scenario.
  2. Score each AI transcript line against the live AI message via keyword
     overlap (Jaccard over word tokens).
  3. If the best match clears the threshold, return its paired human line.
     For ties (re-ask turns like zip_clarification), use turn_counters to
     pick the correct nth occurrence.
  4. Fall back to the slot-based map if no transcript match is found.
     This handles off-script agent turns (unexpected branches, rephrasing
     that is too different from the transcript to match).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
_AMBIGUITY_MARGIN = 0.10


def _load_static_turns(scenario_tag: str) -> List[Dict[str, str]]:
    filename = SCENARIO_FILE_MAP.get(scenario_tag)
    if not filename:
        return []
    path = BASE_PATH / filename
    if not path.exists():
        return []

    turns = []
    ai_msg: Optional[str] = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if re.match(r"(?i)^ai\s*:", line):
            ai_msg = re.sub(r"(?i)^ai\s*:\s*", "", line).strip()
        elif re.match(r"(?i)^(human|user)\s*:", line):
            user_msg = re.sub(r"(?i)^(human|user)\s*:\s*", "", line).strip()
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
    turn_counters: Dict[Tuple[str, str], int] | None = None,
) -> str:
    """
    Return the expected user response for this AI message.

    Source priority:
      1. Static transcript (most reliable — this is the spec)
      2. Slot-based map (fallback for off-script agent turns)

    NEVER calls the simulator or any LLM.
    """
    if turn_counters is None:
        turn_counters = {}

    from scripts.conversational_workload.intent_classifier import classify_ai_slot

    slot = classify_ai_slot(ai_message, flow)

    static_turns = _load_static_turns(scenario_tag)
    if static_turns:
        scored: List[Tuple[int, float, Dict[str, str]]] = []
        for idx, turn in enumerate(static_turns):
            score = _keyword_overlap(ai_message, turn["ai"])
            if score >= _MATCH_THRESHOLD:
                scored.append((idx, score, turn))

        if scored:
            best_score = max(s for _, s, _ in scored)
            candidates = [
                (idx, turn) for idx, score, turn in scored if score >= best_score * (1 - _AMBIGUITY_MARGIN)
            ]
            candidates.sort(key=lambda x: x[0])

            if len(candidates) == 1:
                return candidates[0][1]["user"]

            # Multiple close matches: use visit count
            visit = turn_counters.get((scenario_tag, slot), 0)
            pick = min(visit, len(candidates) - 1)
            return candidates[pick][1]["user"]

    # Fallback: slot-based ground truth
    from scripts.conversational_workload.slot_ground_truth import (
        ground_truth_for_slot,
    )

    return ground_truth_for_slot(
        slot,
        entity,
        flow,
        scenario_tag=scenario_tag,
        turn_counters=turn_counters,
    )
