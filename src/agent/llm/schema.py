from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class EventType(str, Enum):
    ANSWERED = "answered"
    ANSWERED_WITH_FOLLOWUP = "answered_with_followup"
    CORRECTED = "corrected"
    AMBIGUOUS = "ambiguous"
    # Caller asked for time ("give me a few seconds", "let me grab that") — they
    # have NOT declined and have NOT answered. Acknowledge and wait; never count a
    # failed slot attempt or re-prompt the slot question.
    STALLING = "stalling"
    NONE = "none"


class GuardType(str, Enum):
    TRANSFER_REQUEST = "TRANSFER_REQUEST"
    ABUSE = "ABUSE"
    SELF_HARM = "SELF_HARM"
    INTERRUPTION = "INTERRUPTION"
    OFFTOPIC_GLOBAL = "OFFTOPIC_GLOBAL"  # non-healthcare — static response
    OFFTOPIC_AGENT = "OFFTOPIC_AGENT"  # wrong agent — dynamic LLM response
    NONE = "NONE"


class WorkerResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Slot values extracted from the caller's utterance this turn
    extracted: Optional[Dict[str, str]] = None
    # Slot corrections detected (e.g. "actually my name is James")
    corrections: Optional[Dict[str, str]] = None
    # What the caller's utterance did relative to the awaiting slot
    event_type: EventType = EventType.ANSWERED
    # Safety / routing guard triggered this turn
    guard: GuardType = GuardType.NONE
    # LLM's confidence in the guard classification (0.0–1.0)
    guard_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class FollowUpIntent(str, Enum):
    DONE = "done"
    QUESTION = "question"
    UNSURE = "unsure"
    UPDATE_REQUEST = "update_request"
    NEW_INTENT = "new_intent"


class FollowUpResult(BaseModel):
    """Dedicated schema for follow_up_agent: WorkerResult + generated answer."""

    model_config = ConfigDict(extra="forbid")

    extracted: Optional[Dict[str, str]] = None
    guard: GuardType = GuardType.NONE
    guard_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    follow_up_intent: FollowUpIntent = FollowUpIntent.UNSURE
    answer: Optional[str] = None
    detected_intent: Optional[str] = None


# ── TurnPlan: full multi-intent understanding decode (Phase 3) ─────────────────
# Generalizes the FollowUpResult single-decode pattern so any agent's turn can
# carry the primary slot answer PLUS any secondary intents / corrections present
# in the same utterance. Deliberately contains NO free-text/prose field: every
# member-facing sentence is a templated speech-act chosen downstream by the
# deterministic resolver, never improvised by the decode.


class SecondaryIntentType(str, Enum):
    IN_SCOPE_INDEPENDENT = "in_scope_independent"
    INVALIDATING_CORRECTION = "invalidating_correction"
    OUT_OF_SCOPE = "out_of_scope"
    IN_DOMAIN_UNSUPPORTED = "in_domain_unsupported"
    SAFETY = "safety"
    UNKNOWN = "unknown"


class SecondaryIntent(BaseModel):
    type: SecondaryIntentType
    owner: Optional[str] = None  # must resolve in the registry or be dropped
    verbatim_span: str  # MUST appear in the utterance (deterministic check)


class Correction(BaseModel):
    field: str
    owner: str  # must resolve in registry or rejected
    new_value: Optional[str] = None


class TurnPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slot_answer: Optional[str] = None  # passes existing normalizer+validator before acceptance
    secondary_intents: list[SecondaryIntent] = Field(default_factory=list)
    correction: Optional[Correction] = None
    guard: GuardType = GuardType.NONE
    guard_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
