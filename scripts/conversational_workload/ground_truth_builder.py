"""
ground_truth_builder.py — Position-anchored ground truth.

Uses TranscriptCursor to walk the static transcript in lockstep
with the live conversation. Falls back to slot_ground_truth for
off-script turns.

Cursor instances are stored in turn_counters under a reserved key
"__cursor__" so runner.py does not need a separate data structure.
"""

from __future__ import annotations

from typing import Dict, Tuple

from scripts.conversational_workload.intent_classifier import classify_ai_slot
from scripts.conversational_workload.transcript_cursor import TranscriptCursor


def _get_or_create_cursor(
    turn_counters: Dict,
    scenario_tag: str,
) -> TranscriptCursor:
    key = ("__cursor__", scenario_tag)
    if key not in turn_counters:
        turn_counters[key] = TranscriptCursor.load(scenario_tag)
    return turn_counters[key]


def build_dynamic_ground_truth(
    ai_message: str,
    entity,
    flow: str,
    scenario_tag: str = "",
    turn_counters: Dict[Tuple[str, str], int] | None = None,
) -> str:
    if turn_counters is None:
        turn_counters = {}

    slot = classify_ai_slot(ai_message, flow)

    # --- Verification restart detection ---
    # When agent re-asks for first_name after already collecting it,
    # the pipeline restarted. Reset cursor AND slot counters.
    if slot == "first_name":
        fn_visits = turn_counters.get((scenario_tag, "first_name"), 0)
        if fn_visits > 0:
            cursor_key = ("__cursor__", scenario_tag)
            if cursor_key in turn_counters:
                turn_counters[cursor_key].reset()
            for identity_slot in ("first_name", "last_name", "member_id", "dob"):
                turn_counters[(scenario_tag, identity_slot)] = 0

    # --- Primary: cursor-based transcript match ---
    cursor = _get_or_create_cursor(turn_counters, scenario_tag)
    gt = cursor.get_ground_truth(ai_message)
    if gt is not None:
        return gt

    # --- Fallback: slot-based map ---
    from scripts.conversational_workload.slot_ground_truth import ground_truth_for_slot

    return ground_truth_for_slot(
        slot,
        entity,
        flow,
        scenario_tag=scenario_tag,
        turn_counters=turn_counters,
    )
