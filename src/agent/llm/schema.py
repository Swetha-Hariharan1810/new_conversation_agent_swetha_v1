from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class EventType(str, Enum):
    ANSWERED = "answered"
    CORRECTED = "corrected"
    AMBIGUOUS = "ambiguous"
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
