"""
intake_models.py — IntakeAgent typed schemas.

IntentTag is used by the agent for call routing logic.
"""

from __future__ import annotations

from enum import Enum


class IntentTag(str, Enum):
    """Supported caller intent categories."""

    PROVIDER_SERVICES = "provider_services"
    CLAIM_SERVICES = "claim_services"
    OUT_OF_SCOPE = "out_of_scope"
    UNCLEAR = "unclear"
