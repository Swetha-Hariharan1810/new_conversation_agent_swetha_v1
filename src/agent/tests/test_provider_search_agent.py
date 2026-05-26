"""
test_provider_search_agent.py — Live integration test suite for ProviderSearchAgent.

Requires valid Salesforce/LLM credentials via environment variables.
Skipped automatically when credentials are absent.

Run:  pytest src/agent/tests/test_provider_search_agent.py -v -m live
"""

from __future__ import annotations

import os

import pytest

SKIP_LIVE = not (
    os.environ.get("SALESFORCE_USERNAME")
    and os.environ.get("SALESFORCE_PASSWORD")
    and os.environ.get("ANTHROPIC_API_KEY")
)

pytestmark = pytest.mark.skipif(SKIP_LIVE, reason="Live credentials not configured")


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_provider_type_collection() -> None:
    """Live: collect provider_type from a real LLM extraction call."""
    from agent.agents.provider_search.agent import ProviderSearchAgent
    from agent.tests.fixtures import make_verified_state

    state = make_verified_state(
        zip_code="12139",
        messages=[
            {"role": "assistant", "content": "What type of provider are you looking for?"},
            {"role": "user", "content": "I need a primary care physician"},
        ],
        awaiting_slot="provider_type",
    )
    result = await ProviderSearchAgent.from_state(state).execute(state)
    assert result.get("is_interrupt") is True or result.get("next_node") in (
        "provider_search_agent",
        "delivery_management_agent",
        "escalation_agent",
    )


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_zip_confirmation_yes() -> None:
    """Live: confirm existing ZIP."""
    from agent.agents.provider_search.agent import ProviderSearchAgent
    from agent.tests.fixtures import make_verified_state

    state = make_verified_state(
        zip_code="12139",
        provider_type="Primary Care Physician",
        awaiting_slot="zip_confirmed",
        messages=[
            {"role": "assistant", "content": "Your ZIP code is 12139, correct?"},
            {"role": "user", "content": "yes that is correct"},
        ],
    )
    result = await ProviderSearchAgent.from_state(state).execute(state)
    assert result.get("next_node") in (
        "delivery_management_agent",
        "provider_search_agent",
        "escalation_agent",
    )
