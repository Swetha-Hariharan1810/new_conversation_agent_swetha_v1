"""
provider_search_agent.py — Stub: registers the node in the graph for Phase 1 fast-path routing.
Full implementation added in Phase 2.
"""

from __future__ import annotations

from agent.core.agent import BaseAgent
from agent.logger import get_logger
from agent.state import State

logger = get_logger(__name__)


class ProviderSearchAgent(BaseAgent):
    AGENT_NAME = "provider_search_agent"

    async def run(self, state: State) -> dict:
        return self.signal_complete(
            state,
            message="",
            resolved_intents=["provider_services"],
            context_updates={},
            reasoning="Provider search stub — Phase 2 not yet implemented",
        )


async def provider_search_agent(state: State) -> dict:
    logger.info("provider_search_agent: entered", extra={"call_intent": state.get("call_intent", "")})
    return await ProviderSearchAgent.from_state(state).execute(state)
