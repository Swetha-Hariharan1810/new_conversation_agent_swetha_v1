"""
message_builders.py — Verification-specific prompt builder functions.

Moved from agents/verification/prompts.py. These functions require runtime
values from state and cannot be satisfied by static string pools alone.
"""

from __future__ import annotations

import re


def build_phone_confirmation_prompt(phone_raw: str) -> str:
    """
    Phone confirmation with live formatted number from SF record.
    Cannot be in generic components — value comes from state at runtime.
    """
    digits = "".join(c for c in phone_raw if c.isdigit())
    formatted = f"{digits[:3]}-{digits[3:6]}-{digits[6:]}" if len(digits) == 10 else phone_raw
    return f"Thank you. Is your phone number {formatted}?"


def build_offtopic_redirect(next_slot_prompt: str, prefix: str) -> str:
    """
    Acknowledge the off-topic request and steer back to verification.
    """
    return f"{prefix} {next_slot_prompt}"


def _parse_relationships(relationship_str: str) -> list[str]:
    """Split "plan holder, subscriber or spouse" into["plan holder", "subscriber", "spouse"]."""
    if not relationship_str:
        return []
    parts = re.split(r",\s*|\s+or\s+", relationship_str.strip())
    return [p.strip() for p in parts if p.strip()]


def build_relationship_confirmation_prompt(relationship_str: str) -> str:
    return "Thank you, I found your account. Are you the plan holder or dependent?"
