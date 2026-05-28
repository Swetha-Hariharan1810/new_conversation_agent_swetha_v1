"""
Simulates user responses by classifying the AI message and returning
the scripted reply for that slot from the ground-truth map.
"""

from scripts.conversational_workload.intent_classifier import classify_ai_slot
from scripts.conversational_workload.slot_ground_truth import ground_truth_for_slot


def simulate_user_response(ai_message: str, entity, flow: str = "pcp") -> str:
    slot = classify_ai_slot(ai_message, flow)
    return ground_truth_for_slot(slot, entity, flow)
