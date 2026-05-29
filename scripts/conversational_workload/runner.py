"""
LangGraph-backed conversational evaluation runner for the PCP flow.
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END
from langgraph.types import Command

from scripts.conversational_workload.ground_truth_builder import build_dynamic_ground_truth
from scripts.conversational_workload.intent_classifier import classify_ai_slot
from scripts.conversational_workload.judge import judge_turn
from scripts.conversational_workload.models import (
    ConversationReport,
    PCPInquiryEntity,
    TurnEvaluation,
)
from scripts.conversational_workload.report_builder import save_report
from scripts.conversational_workload.user_simulator import simulate_user_response
from scripts.conversational_workload.utils import extract_last_ai_message, new_conversation_id

logger = logging.getLogger(__name__)

MAX_TURNS = 30


async def run_evaluation_async(
    entity_data: dict,
    flow: str = "pcp",
    scenario_tag: str = "pcp_happy_path",
) -> ConversationReport:
    entity = PCPInquiryEntity(**entity_data)

    from agent.app_graph import build_graph

    memory = MemorySaver()
    graph = build_graph(checkpointer=memory)

    config = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "tags": ["evaluation", flow, scenario_tag],
        "metadata": {"flow": flow, "scenario_tag": scenario_tag},
    }

    state = await graph.ainvoke({}, config=config)

    conversation_id = new_conversation_id()
    turns: list[TurnEvaluation] = []
    turn_counters: dict = {}

    for _ in range(MAX_TURNS):
        if state.get("next_node") == END:
            break

        if state.get("is_interrupt"):
            ai_msg = extract_last_ai_message(state.get("messages", []))
            if not ai_msg:
                state = await graph.ainvoke(Command(resume=""), config=config)
                continue

            slot = classify_ai_slot(ai_msg, flow)
            user_text = simulate_user_response(
                ai_msg,
                entity,
                flow,
                scenario_tag=scenario_tag,
                turn_counters=turn_counters,
            )
            ground_truth = build_dynamic_ground_truth(
                ai_msg,
                entity,
                flow,
                scenario_tag=scenario_tag,
                turn_counters=turn_counters,
            )
            judge_result = judge_turn(user_text, ground_truth)

            turns.append(
                TurnEvaluation(
                    ai_prompt=ai_msg,
                    user_response=user_text,
                    ground_truth=ground_truth,
                    slot=slot,
                    scenario=scenario_tag,
                    scores=judge_result.model_dump(),
                )
            )

            state = await graph.ainvoke(Command(resume=user_text), config=config)

            # Increment after recording so both simulator and ground-truth builder
            # see the same visit count for this turn.
            key = (scenario_tag, slot)
            turn_counters[key] = turn_counters.get(key, 0) + 1
        else:
            state = await graph.ainvoke(Command(resume=""), config=config)

    final_score = round(sum(t.scores["overall"] for t in turns) / len(turns), 2) if turns else 0.0

    report = ConversationReport(
        conversation_id=conversation_id,
        flow=flow,
        scenario_tag=scenario_tag,
        completed=state.get("next_node") == END,
        turns=turns,
        final_score=final_score,
    )

    save_report(report)
    return report


def run_evaluation(
    entity_data: dict,
    flow: str = "pcp",
    scenario_tag: str = "pcp_happy_path",
) -> ConversationReport:
    """Sync wrapper — keeps run_eval.py and all callers unchanged."""
    return asyncio.run(run_evaluation_async(entity_data, flow, scenario_tag))
