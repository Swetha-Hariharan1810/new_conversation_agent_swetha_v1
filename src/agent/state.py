from typing import Annotated, Literal, Optional, Union

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

ConversationContextDict = dict


class CallAgentFieldData(TypedDict):
    field: str
    value: str


class CallAgentFieldEvent(TypedDict):
    eventType: Literal["CallAgentField"]
    data: CallAgentFieldData


class AgentCallEventData(TypedDict):
    eventName: Literal["AgentCallEnded", "AgentCallTransfer"]
    detail: str
    transferInitiator: Optional[Literal["Agent", "Caller"]]


class AgentCallLifecycleEvent(TypedDict):
    eventType: Literal["AgentCallEvent"]
    data: AgentCallEventData


Event = Union[CallAgentFieldEvent, AgentCallLifecycleEvent]


class SlotState(TypedDict, total=False):
    attempt_count: int
    confirmed: bool
    last_value: Optional[str]


class State(TypedDict):
    # ── Core LangGraph fields ────────────────────────────────────────────────
    messages: Annotated[list, add_messages]
    metadata_events: list[Event]
    is_interrupt: bool
    next_node: str
    app_run_id: str
    last_agent_signal: dict
    active_agent: str
    previous_agents: list[str]
    intent_queue: list[str]
    orchestrator_reasoning: str
    router_loop_count: int
    call_intent: str
    ref_no: str
    slot_attempts: dict[str, SlotState]
    conversation_context: Optional[ConversationContextDict]

    # ── Caller identity (set by verification) ────────────────────────────────
    first_name: str
    last_name: str
    member_id: str
    dob: str
    relationship: str
    caller_role: str
    member_status_verify: bool

    # ── Contact fields (set by verification via context_updates) ─────────────
    phone_number: str
    zip_code: str
    fax: str
    email: str
    phone_confirmed: bool
    phone_update_requested: bool

    # ── Escalation ───────────────────────────────────────────────────────────
    escalation_reference_number: str
    escalation_reason: str

    # ── Slot tracking ────────────────────────────────────────────────────────
    awaiting_slot: str
    correction_return_to: str
    ambiguous_counts: dict

    # ── Verification restart boundary ────────────────────────────────────────
    verification_restart_index: int

    # ── Flow control ─────────────────────────────────────────────────────────
    new_intent_detected: str
    offtopic_global_count: int
    closure_requested: bool
