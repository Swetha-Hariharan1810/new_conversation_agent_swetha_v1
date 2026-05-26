"""
agent.py — DeliveryManagementAgent stub.
Registers the node in the graph for Phase 2 routing.
Full implementation added in Phase 3.
"""

from __future__ import annotations

from agent.core.agent import BaseAgent
from agent.logger import get_logger
from agent.state import State

logger = get_logger(__name__)


class DeliveryManagementAgent(BaseAgent):
    AGENT_NAME = "delivery_management_agent"

    async def run(self, state: State) -> dict:
        return self.signal_complete(
            state,
            message="",
            resolved_intents=["provider_list_delivery"],
            context_updates={},
            reasoning="Delivery management stub — Phase 3 not yet implemented",
        )


async def delivery_management_agent(state: State) -> dict:
    logger.info("delivery_management_agent: entered", extra={"call_intent": state.get("call_intent", "")})
    return await DeliveryManagementAgent.from_state(state).execute(state)
