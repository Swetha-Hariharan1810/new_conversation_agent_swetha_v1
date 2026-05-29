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
    proactive_offer_available: bool

    # ── Provider Search context ───────────────────────────────────────────────
    provider_type: str
    zip_code_used: str
    provider_list_sent: bool
    delivery_timestamp: str
    fax_confirmed: bool
    fax_update_requested: bool
    email_confirmed: bool
    email_update_requested: bool
    delivery_method: str
    benefits_offer_made: bool

    # ── Benefits & Wellness context ──────────────────────────────────────────
    individual_deductible: str  # e.g. "750"
    family_deductible: str  # e.g. "2500"
    coinsurance_percent: str  # e.g. "20"
    individual_oop_max: str  # e.g. "3000"
    family_oop_max: str  # e.g. "7000"
    benefits_explained: bool  # True once benefits_agent has read the summary
    care_coach_offer_made: bool  # True once care coach offer was presented
    care_coach_offered: bool  # True once member accepted or declined
    care_coach_details_sent: bool  # True once details were dispatched
    rewards_portal_shared: bool  # True once portal link was given
    care_coach_nooffer_sent: bool  # True when member declined and no-offer msg was sent

    # ── Follow-up Agent context ──────────────────────────────────────────────────
    follow_up_turn_count: int  # incremented each time follow_up_agent runs
    follow_up_last_question: str  # last question the member asked this flow

    # ── Caller type detection ────────────────────────────────────────────────
    caller_type: str  # "member" | "provider" | "employer_group" | "other_carrier" | "unknown"
    caller_type_handled: bool  # prevents re-triggering once handled
    slot_before_caller_type_check: str  # saves awaiting_slot so we can resume if caller wants to continue
