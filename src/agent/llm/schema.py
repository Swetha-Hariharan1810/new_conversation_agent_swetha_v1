from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class EventType(str, Enum):
    ANSWERED = "answered"
    ANSWERED_WITH_FOLLOWUP = "answered_with_followup"
    CORRECTED = "corrected"
    AMBIGUOUS = "ambiguous"
    WAIT = "wait"  # caller asked for time: "give me a minute", "hold on"
    NONE = "none"


class FollowupDisposition(str, Enum):
    ANSWER_NOW = "answer_now"  # answerable from confirmed slots / repeat request / update ack
    PARK = "park"  # answerable later in this call (maps to a pending slot or later stage)
    DECLINE = "decline"  # irrelevant or never answerable in this call
    NONE = "none"  # default when event_type != answered_with_followup


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
    # How to handle the side question when event_type == ANSWERED_WITH_FOLLOWUP
    followup_disposition: FollowupDisposition = FollowupDisposition.NONE
    # The side question, condensed, verbatim-ish
    followup_query: Optional[str] = None
    # Slot the caller wants to change when NO new value was given
    update_target: Optional[str] = None


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
