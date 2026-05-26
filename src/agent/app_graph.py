"""
app_graph.py — Trimmed LangGraph workflow (core agents only).
Agents: intake · verification · closure · escalation · orchestrator
"""

from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from agent.agents.closure.agent import closure_agent
from agent.agents.escalation.agent import escalation_agent
from agent.agents.intake.agent import intake_agent
from agent.agents.provider_search.agent import provider_search_agent
from agent.agents.verification.agent import verification_agent
from agent.logger import get_logger
from agent.orchestration.orchestration import (
    ALL_AGENT_NODES,
    AgentNode,
    orchestrator,
)
from agent.state import State
from agent.utils import clean_asr_input

logger = get_logger(__name__)


# ── Intake routing — deterministic, bypasses Orchestrator ────────────────────
def intake_routing(state: State) -> str:
    if state.get("is_interrupt"):
        return "human_node"
    if state.get("next_node") == END:
        return END
    if state.get("next_node") == AgentNode.ESCALATION.value:
        return AgentNode.ESCALATION.value
    return AgentNode.VERIFICATION.value


# ── Shared conditional router ────────────────────────────────────────────────
def conditional_routing(state: State) -> str:
    next_node = state.get("next_node", "")
    is_interrupt = state.get("is_interrupt", False)
    logger.info("conditional_routing", extra={"next_node": next_node, "is_interrupt": is_interrupt})
    if is_interrupt:
        return "human_node"
    if next_node == END:
        return END
    known = ALL_AGENT_NODES + ["orchestrator", "human_node"]
    if next_node in known:
        return next_node
    logger.warning("conditional_routing: unknown node — fallback to intake", extra={"next_node": next_node})
    return AgentNode.INTAKE.value


# ── Orchestrator routing ──────────────────────────────────────────────────────
def orchestrator_routing(state: State) -> str:
    next_node = state.get("next_node", "")
    if state.get("is_interrupt"):
        return "human_node"
    if next_node == END:
        return END
    if next_node in ALL_AGENT_NODES:
        return next_node
    logger.warning("orchestrator_routing: unknown node — fallback", extra={"next_node": next_node})
    return AgentNode.INTAKE.value


# ── Human interrupt node ──────────────────────────────────────────────────────
def human_node(state: State) -> Command:
    interrupt_message = ""
    try:
        messages = state.get("messages")
        if isinstance(messages, list) and messages:
            last = messages[-1]
            interrupt_message = (
                last.get("content", "") if isinstance(last, dict) else getattr(last, "content", "")
            )
        elif isinstance(messages, dict):
            interrupt_message = messages.get("content", "")
        else:
            interrupt_message = getattr(messages, "content", "")
    except Exception:
        logger.exception("human_node: failed reading interrupt message")
    value = interrupt(interrupt_message)
    value = clean_asr_input(str(value) if value is not None else "")
    next_node = state.get("next_node", AgentNode.INTAKE.value)
    logger.info(f"human_node: collected → {next_node}", extra={"len": len(value)})
    return Command(
        goto=next_node,
        update={"is_interrupt": False, "messages": [{"role": "user", "content": value}]},
    )


# ── Graph builder ─────────────────────────────────────────────────────────────
def build_graph():
    g = StateGraph(State)
    g.add_node("orchestrator", orchestrator)
    g.add_node("human_node", human_node)
    g.add_node(AgentNode.INTAKE.value, intake_agent)
    g.add_node(AgentNode.VERIFICATION.value, verification_agent)
    g.add_node(AgentNode.ESCALATION.value, escalation_agent)
    g.add_node(AgentNode.CLOSURE.value, closure_agent)
    g.add_node(AgentNode.PROVIDER_SEARCH.value, provider_search_agent)
    g.add_conditional_edges(AgentNode.INTAKE.value, intake_routing)
    for node in [
        AgentNode.VERIFICATION.value,
        AgentNode.ESCALATION.value,
        AgentNode.CLOSURE.value,
        AgentNode.PROVIDER_SEARCH.value,
    ]:
        g.add_conditional_edges(node, conditional_routing)
    g.add_conditional_edges("orchestrator", orchestrator_routing)
    g.set_entry_point(AgentNode.INTAKE.value)
    return g.compile()


graph = build_graph()
