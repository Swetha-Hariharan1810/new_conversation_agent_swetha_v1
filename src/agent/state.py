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
    pending_intent: Optional[str]  # new intent staged by reset_for_new_intent (mid-call switch)
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

    # ── Name confirmation (new) ──────────────────────────────────────────────────
    name_confirmed: bool
    name_confirm_attempts: int

    # ── Contact fields (set by verification via context_updates) ─────────────
    phone_number: str
    zip_code: str
    fax: str
    email: str
    phone_confirmed: bool
    phone_update_requested: bool

    # ── Pending reconfirmation values (held until member confirms) ──────────
    pending_zip_code: str
    pending_fax: str
    pending_email: str
    pending_phone: str

    # ── Escalation ───────────────────────────────────────────────────────────
    escalation_reference_number: str
    escalation_reason: str
    escalation_pre_message: str  # pre-escalation context message from the calling agent

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
    zip_code_updated: bool  # True when the member changed their ZIP this call
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
    follow_up_cannot_answer_count: int

    # ── Caller type detection ────────────────────────────────────────────────
    caller_type: str  # "member" | "provider" | "employer_group" | "other_carrier" | "unknown"
    caller_type_handled: bool  # prevents re-triggering once handled

    # ── Claim Adjustment context ─────────────────────────────────────────────────
    reference_number: str  # adjustment request reference number from member
    claim_status: str  # retrieved from Salesforce e.g. "open for Review"
    last_update_date: str  # ISO date of most recent SF status update
    records_required: bool  # flag from Salesforce adjustment record
    records_branch_taken: str  # "member_upload"|"provider_direct"|"personal_guide"|"declined"
    upload_link_sent: bool  # True once Salesforce upload link generated and sent
    personal_guide_outreach_requested: bool  # True once Personal Guide Salesforce workflow triggered
    notification_channel: str  # "sms"|"email"|"not_set"
    claim_notification_contact: str  # confirmed phone or email to receive notifications
    claim_timeline_notification_channel: str  # "sms"|"email"|"not_set" — for progress updates
    claim_timeline_notification_contact: str  # contact for progress update notifications
    claim_flow_complete: bool  # True once both notification preferences are saved


def reset_for_new_intent(state: State, new_intent: Optional[str]) -> dict:
    """Return the state updates that fully reset the conversation for a brand-new
    intent detected mid-call, forcing identity re-verification from scratch.

    Unlike ``follow_up.constants.NEW_INTENT_CLEAR_FIELDS`` (which deliberately
    *keeps* identity + verification so the member fast-paths through
    verification), this reset zeroes identity and verification too, so the
    verification agent restarts at the first slot ("first name").

    The new intent is staged in BOTH:
      * ``call_intent``   — the field routing actually reads (verification's
        pipeline/prompt selection AND the post-verification fast-path branch).
      * ``pending_intent`` — a dedicated, durable marker that a re-verification
        triggered by a mid-call switch is in flight, independent of the
        transient ``new_intent_detected`` signal (which is gated on the member
        already being verified and so cannot survive this reset).

    Fields NOT returned here are preserved by LangGraph's last-write-wins
    reducers — in particular the transcript (``messages``, an ``add_messages``
    field), the session/run id (``app_run_id``), the call reference (``ref_no``),
    ``metadata_events`` and ``last_agent_signal``. ``state`` is accepted for API
    symmetry and future conditional resets; the current reset is unconditional.
    """
    return {
        # ── Intent carriers ──────────────────────────────────────────────────
        "call_intent": new_intent or "",  # field routing reads (pipeline + post-verify)
        "pending_intent": new_intent,  # durable mid-call-switch marker
        "new_intent_detected": "",  # consumed — clear the trigger
        "intent_queue": [],
        # ── Member identity → None ───────────────────────────────────────────
        "first_name": None,
        "last_name": None,
        "member_id": None,
        "dob": None,
        "zip_code": None,
        "relationship": None,
        "caller_role": None,
        "phone_number": None,
        "fax": None,
        "email": None,
        "conversation_context": None,  # rebuilt fresh from cleared identity
        # ── Verification flag(s) → False/0 ───────────────────────────────────
        "member_status_verify": False,
        "name_confirmed": False,
        "name_confirm_attempts": 0,
        "phone_confirmed": False,
        "phone_update_requested": False,
        # ── Verification sub-step pointer → initial (restart at first_name) ───
        "awaiting_slot": "",  # verification recomputes first empty slot = first_name
        "slot_attempts": {},
        "correction_return_to": "",
        "ambiguous_counts": {},
        "verification_restart_index": 0,
        # ── Provider search ──────────────────────────────────────────────────
        "provider_type": "",
        "zip_code_used": "",
        "zip_code_updated": False,
        "provider_list_sent": False,
        "delivery_timestamp": "",
        "fax_confirmed": False,
        "fax_update_requested": False,
        "email_confirmed": False,
        "email_update_requested": False,
        "delivery_method": "",
        "benefits_offer_made": False,
        # ── Pending reconfirmation values ────────────────────────────────────
        "pending_zip_code": "",
        "pending_fax": "",
        "pending_email": "",
        "pending_phone": "",
        # ── Benefits & wellness ──────────────────────────────────────────────
        "individual_deductible": "",
        "family_deductible": "",
        "coinsurance_percent": "",
        "individual_oop_max": "",
        "family_oop_max": "",
        "benefits_explained": False,
        "care_coach_offer_made": False,
        "care_coach_offered": False,
        "care_coach_details_sent": False,
        "rewards_portal_shared": False,
        "care_coach_nooffer_sent": False,
        # ── Follow-up counters ───────────────────────────────────────────────
        "follow_up_turn_count": 0,
        "follow_up_last_question": "",
        "follow_up_cannot_answer_count": 0,
        # ── Claim adjustment ─────────────────────────────────────────────────
        "reference_number": "",
        "claim_status": "",
        "last_update_date": "",
        "records_required": False,
        "records_branch_taken": "",
        "upload_link_sent": False,
        "personal_guide_outreach_requested": False,
        "notification_channel": "not_set",
        "claim_notification_contact": "",
        "claim_timeline_notification_channel": "not_set",
        "claim_timeline_notification_contact": "",
        "claim_flow_complete": False,
        # ── Escalation ───────────────────────────────────────────────────────
        "escalation_reference_number": "",
        "escalation_reason": "",
        "escalation_pre_message": "",
        # ── Flow control / orchestration bookkeeping ─────────────────────────
        "offtopic_global_count": 0,
        "closure_requested": False,
        "proactive_offer_available": False,
        "router_loop_count": 0,
        "orchestrator_reasoning": "",
        "previous_agents": [],
    }
