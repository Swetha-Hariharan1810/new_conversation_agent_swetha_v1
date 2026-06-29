"""
triage.py — classify a single captured intent into an IntentKind.

Pure and deterministic. Priority order matters: safety first, then a
correction of a committed value, then off topic, then in domain but
unsupported, then a known in scope intent, otherwise unsupported.
"""

from __future__ import annotations

from typing import Optional

from agent.core.pending_intents import (
    CORRECTION_OWNER,
    INTENT_AGENT,
    IntentKind,
)

# In domain requests that no flow serves today.
UNSUPPORTED_TOPICS: set[str] = {"billing", "pharmacy", "preauth", "preauthorisation", "preauthorization"}

# Guards that mean stop and escalate.
SAFETY_GUARDS: set[str] = {"ABUSE", "SELF_HARM", "TRANSFER_REQUEST"}


def classify_intent(
    *,
    guard: str,
    correction_target: Optional[str],
    intent_label: Optional[str],
    topic: Optional[str],
) -> IntentKind:
    if guard in SAFETY_GUARDS:
        return IntentKind.SAFETY
    if correction_target:
        # Only a correction of a value owned by an agent can be rewound, written
        # to Salesforce, and resolved (zip_code, fax, email, phone_number — see
        # CORRECTION_OWNER). These are intentionally updatable mid-call via the
        # rewind-and-rebuild flow. A correction_target with no owner (e.g. "dob",
        # "member_id") has no rewind route, so treating it as IN_SCOPE_INVALIDATING
        # would leave an open intent that blocks closure (safeguard 5b) yet can
        # never be resolved, permanently trapping the call. Unowned targets fall
        # through to UNSUPPORTED, which does not block closure.
        if correction_target in CORRECTION_OWNER:
            return IntentKind.IN_SCOPE_INVALIDATING
        return IntentKind.UNSUPPORTED
    if guard == "OFFTOPIC_GLOBAL":
        return IntentKind.OFF_TOPIC
    if topic and topic.lower() in UNSUPPORTED_TOPICS:
        return IntentKind.UNSUPPORTED
    if intent_label and intent_label in INTENT_AGENT:
        return IntentKind.IN_SCOPE_INDEPENDENT
    return IntentKind.UNSUPPORTED
