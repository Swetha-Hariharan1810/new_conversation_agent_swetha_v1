"""
Extraction variance matrix (Phase 6) — the codified definition of
"stable results per run".

For each of the five production transcripts (BUG-1…BUG-5), every plausible
WorkerResult/FollowUpResult the extraction LLM has been observed to return
(including event_type=AMBIGUOUS mislabels) must produce an IDENTICAL final
routing decision: next_node + awaiting_slot + message class. The regex
fallback + veto layer (request_detection) is what makes the column constant.
"""

import pytest

from agent.agents.benefits.agent import BenefitsAgent
from agent.agents.delivery_management.agent import DeliveryManagementAgent
from agent.agents.follow_up.agent import FollowUpAgent
from agent.agents.verification.agent import VerificationAgent
from agent.llm.schema import (
    EventType,
    FollowupDisposition,
    FollowUpIntent,
    FollowUpResult,
    RequestKind,
    WorkerResult,
)

# ── message-class helper ─────────────────────────────────────────────────────


def _msg_class(out: dict) -> str:
    """Coarse classification of the member-facing message this turn."""
    msg = ""
    messages = out.get("messages")
    if isinstance(messages, dict):
        msg = (messages.get("content") or "").lower()
    if not msg:
        return "none"
    spoken_email = " at " in msg and " dot " in msg
    if ("email" in msg or spoken_email) and ("correct" in msg or "right" in msg):
        return "email_confirm"
    if "fax number" in msg:
        return "fax_confirm"
    if "zip" in msg:
        return "ask_zip"
    return "other"


# ══ BUG-5: "wait — my ZIP changed" during fax confirmation ═══════════════════

_BUG5_VARIANTS = [
    WorkerResult(update_target="zip_code", request_kind=RequestKind.UPDATE),
    WorkerResult(corrections={"zip_code": "60660"}),
    WorkerResult(),
    WorkerResult(event_type=EventType.WAIT),
    WorkerResult(event_type=EventType.AMBIGUOUS),
    WorkerResult(extracted={"fax_confirmed": "no"}),
]


def _delivery_state(user: str, awaiting="fax_confirmed") -> dict:
    return {
        "messages": [
            {"role": "assistant", "content": "The fax on file is 555-123-4567. Is this correct?"},
            {"role": "user", "content": user},
        ],
        "awaiting_slot": awaiting,
        "delivery_method": "fax",
        "fax": "5551234567",
        "email": "jane.doe@example.com",
        "zip_code": "90210",
        "zip_code_used": "90210",
        "parked_followups": [],
    }


def _mk_delivery(monkeypatch, result) -> DeliveryManagementAgent:
    import agent.agents.delivery_management.agent as dma

    async def fake_extract(*a, **k):
        return result

    monkeypatch.setattr(dma, "extract_delivery_management_decision", fake_extract)
    monkeypatch.setattr(dma, "get_extraction_llm", lambda: object())
    return DeliveryManagementAgent()


@pytest.mark.parametrize("variant", _BUG5_VARIANTS)
async def test_bug5_zip_update_routing_is_invariant(monkeypatch, variant):
    agent = _mk_delivery(monkeypatch, variant.model_copy(deep=True))
    out = await agent.run(_delivery_state("wait — my zip code changed, i moved"))
    signature = {
        "next_node": out.get("next_node"),
        "awaiting_slot": out.get("awaiting_slot"),
        "request": {k: out.get("pending_cross_agent_request", {}).get(k) for k in ("kind", "target")},
        "msg_class": _msg_class(out),
    }
    assert signature == {
        "next_node": "provider_search_agent",
        "awaiting_slot": "zip_code",
        "request": {"kind": "update", "target": "zip_code"},
        "msg_class": "ask_zip",
    }


# ══ BUG-3: "can you email it instead" during fax confirmation ════════════════

_BUG3_VARIANTS = [
    WorkerResult(extracted={"delivery_method": "email"}),
    WorkerResult(update_target="delivery_method", request_kind=RequestKind.REDO),
    WorkerResult(),
    WorkerResult(event_type=EventType.WAIT),
    WorkerResult(event_type=EventType.AMBIGUOUS),
]


@pytest.mark.parametrize("variant", _BUG3_VARIANTS)
async def test_bug3_method_switch_is_invariant(monkeypatch, variant):
    agent = _mk_delivery(monkeypatch, variant.model_copy(deep=True))
    out = await agent.run(_delivery_state("can you email it instead"))
    signature = {
        "awaiting_slot": out.get("awaiting_slot"),
        "delivery_method": out.get("delivery_method"),
        "pending_fax": out.get("pending_fax"),
        "msg_class": _msg_class(out),
    }
    assert signature == {
        "awaiting_slot": "email_confirmed",
        "delivery_method": "email",
        "pending_fax": "",
        "msg_class": "email_confirm",
    }


# ══ BUG-4: member_id answer + "also I need to update my last name" ═══════════

_BUG4_SPOKEN_ID = "m nine zero seven five zero three"
_BUG4_VARIANTS = [
    WorkerResult(
        extracted={"member_id": _BUG4_SPOKEN_ID},
        event_type=EventType.ANSWERED_WITH_FOLLOWUP,
        update_target="last_name",
        request_kind=RequestKind.UPDATE,
    ),
    WorkerResult(
        extracted={"member_id": _BUG4_SPOKEN_ID},
        event_type=EventType.ANSWERED_WITH_FOLLOWUP,
        followup_query="I need to update my last name",
        followup_disposition=FollowupDisposition.PARK,
    ),
    WorkerResult(
        extracted={"member_id": _BUG4_SPOKEN_ID},
        event_type=EventType.ANSWERED_WITH_FOLLOWUP,
        followup_disposition=FollowupDisposition.DECLINE,
        update_target="last_name",
        request_kind=RequestKind.UPDATE,
    ),
    WorkerResult(extracted={"member_id": _BUG4_SPOKEN_ID}),  # flattened ANSWERED
    WorkerResult(extracted={"member_id": _BUG4_SPOKEN_ID}, event_type=EventType.CORRECTED),
    WorkerResult(extracted={"member_id": _BUG4_SPOKEN_ID}, event_type=EventType.AMBIGUOUS),
]


@pytest.mark.parametrize("variant", _BUG4_VARIANTS)
async def test_bug4_identity_update_is_invariant(monkeypatch, variant):
    import agent.agents.verification.agent as va
    import agent.llm.response_generator as rg

    async def fake_extract(*a, **k):
        return variant.model_copy(deep=True)

    captured: dict = {}

    async def fake_generate(**kwargs):
        captured.setdefault("guards", []).append(kwargs.get("guard"))
        return "Got it — and the new last name?"

    monkeypatch.setattr(va, "extract_verification_decision", fake_extract)
    monkeypatch.setattr(va, "get_extraction_llm", lambda: object())
    monkeypatch.setattr(rg, "generate_recovery_message", fake_generate)

    agent = VerificationAgent()
    out = await agent.run(
        {
            "messages": [
                {"role": "assistant", "content": "Could I have your member ID?"},
                {"role": "user", "content": f"{_BUG4_SPOKEN_ID} — oh, also I need to update my last name"},
            ],
            "awaiting_slot": "member_id",
            "first_name": "Emily",
            "last_name": "Carter",
            "name_confirmed": True,
            "call_intent": "provider_services",
            "parked_followups": [],
            "ambiguous_counts": {},
        }
    )
    signature = {
        "awaiting_slot": out.get("awaiting_slot"),
        "correction_return_to": out.get("correction_return_to"),
        "member_id": out.get("member_id"),
        "last_name": out.get("last_name"),
        "guards": captured.get("guards"),
    }
    assert signature == {
        "awaiting_slot": "last_name",
        "correction_return_to": "dob",
        "member_id": "M907503",
        "last_name": "",
        "guards": ["FOLLOWUP_ANSWER"],
    }


# ══ BUG-2: "send that list to my email instead of fax" during Care Coach ═════

_BUG2_VARIANTS = [
    WorkerResult(update_target="delivery_method", request_kind=RequestKind.REDO),
    WorkerResult(update_target="provider_list", request_kind=RequestKind.REDO),
    WorkerResult(),
    WorkerResult(event_type=EventType.WAIT),
    WorkerResult(event_type=EventType.AMBIGUOUS),
    WorkerResult(extracted={"care_coach_response": "no"}),
]


@pytest.mark.parametrize("variant", _BUG2_VARIANTS)
async def test_bug2_care_coach_redo_is_invariant(monkeypatch, variant):
    import agent.agents.benefits.agent as ba

    async def fake_extract(*a, **k):
        return variant.model_copy(deep=True)

    monkeypatch.setattr(ba, "extract_benefits_decision", fake_extract)
    monkeypatch.setattr(ba, "get_extraction_llm", lambda: object())

    agent = BenefitsAgent()
    out = await agent.run(
        {
            "messages": [
                {"role": "assistant", "content": "Would you like details about our Care Coach Guides?"},
                {"role": "user", "content": "please send that list to my email instead of fax"},
            ],
            "awaiting_slot": "care_coach_response",
            "provider_list_sent": True,
            "benefits_explained": True,
            "slot_attempts": {},
            "parked_followups": [],
        }
    )
    signature = {
        "next_node": out.get("next_node"),
        "request": out.get("pending_cross_agent_request"),
        "msg_class": _msg_class(out),
    }
    assert signature == {
        "next_node": "delivery_management_agent",
        "request": {
            "kind": "redo",
            "target": "delivery",
            "return_to_agent": "benefits_agent",
            "return_awaiting": "care_coach_response",
        },
        "msg_class": "none",  # the owner speaks, not benefits
    }


# ══ BUG-1: parked notification question surfacing in follow_up ═══════════════

# The routing is deterministic and pre-LLM, so even a hallucination-shaped
# answer or a DONE/UNSURE misclassification cannot change the outcome — the
# classifier is never consulted for an owned parked question.
_BUG1_VARIANTS = [
    FollowUpResult(
        follow_up_intent=FollowUpIntent.QUESTION,
        answer="Yes — we sent it to your email!",  # the BUG-1 hallucination
    ),
    FollowUpResult(follow_up_intent=FollowUpIntent.UNSURE),
    FollowUpResult(follow_up_intent=FollowUpIntent.UPDATE_REQUEST),
    FollowUpResult(),
]


@pytest.mark.parametrize("variant", _BUG1_VARIANTS)
async def test_bug1_parked_notification_is_invariant(monkeypatch, variant):
    import agent.agents.follow_up.agent as fua

    async def fake_extract(*a, **k):
        return variant.model_copy(deep=True)

    monkeypatch.setattr(fua, "extract_follow_up_decision", fake_extract)
    monkeypatch.setattr(fua, "get_follow_up_llm", lambda: object())

    agent = FollowUpAgent()
    out = await agent.run(
        {
            "messages": [
                {"role": "assistant", "content": "Aside from this, anything else?"},
                {"role": "user", "content": "yes actually"},
            ],
            "follow_up_turn_count": 1,
            "follow_up_cannot_answer_count": 0,
            "call_intent": "provider_services",
            "provider_list_sent": True,
            "delivery_method": "fax",
            "fax": "5551234567",
            "parked_followups": [
                {
                    "query": "will I get a notification when the list is sent?",
                    "kind": "question",
                    "target": "",
                }
            ],
        }
    )
    signature = {
        "next_node": out.get("next_node"),
        "request": {k: out.get("pending_cross_agent_request", {}).get(k) for k in ("kind", "target")},
        "parked": out.get("parked_followups"),
        "msg_class": _msg_class(out),
    }
    assert signature == {
        "next_node": "delivery_management_agent",
        "request": {"kind": "replay", "target": "provider_list"},
        "parked": [],
        "msg_class": "none",  # delivery answers from state on the hop turn
    }


# ══ Claims path (Phase 7): update voiced while awaiting reference_number ═════

_CLAIMS_ZIP_VARIANTS = [
    WorkerResult(update_target="zip_code", request_kind=RequestKind.UPDATE),
    WorkerResult(),
    WorkerResult(event_type=EventType.WAIT),
    WorkerResult(event_type=EventType.AMBIGUOUS),
]


@pytest.mark.parametrize("variant", _CLAIMS_ZIP_VARIANTS)
async def test_claims_zip_update_routing_is_invariant(monkeypatch, variant):
    import agent.agents.claim_adjustment.agent as caa
    from agent.agents.claim_adjustment.agent import ClaimAdjustmentAgent

    async def fake_extract(*a, **k):
        return variant.model_copy(deep=True)

    monkeypatch.setattr(caa, "extract_claim_adjustment_decision", fake_extract)
    monkeypatch.setattr(caa, "get_extraction_llm", lambda: object())

    out = await ClaimAdjustmentAgent().run(
        {
            "messages": [
                {"role": "assistant", "content": "Could I get the reference number?"},
                {"role": "user", "content": "wait — my zip code changed, i moved"},
            ],
            "awaiting_slot": "reference_number",
            "call_intent": "claim_services",
            "member_status_verify": True,
            "zip_code": "90210",
            "parked_followups": [],
        }
    )
    signature = {
        "next_node": out.get("next_node"),
        "awaiting_slot": out.get("awaiting_slot"),
        "request": {k: out.get("pending_cross_agent_request", {}).get(k) for k in ("kind", "target")},
        "msg_class": _msg_class(out),
    }
    assert signature == {
        "next_node": "provider_search_agent",
        "awaiting_slot": "zip_code",
        "request": {"kind": "update", "target": "zip_code"},
        "msg_class": "ask_zip",
    }


# ══ Claims path (Phase 7): channel switch during the phone read-back ═════════

_CHANNEL_SWITCH_VARIANTS = [
    WorkerResult(extracted={"notification_method": "email"}),
    WorkerResult(),
    WorkerResult(event_type=EventType.WAIT),
    WorkerResult(event_type=EventType.AMBIGUOUS),
]


@pytest.mark.parametrize("variant", _CHANNEL_SWITCH_VARIANTS)
async def test_notification_channel_switch_is_invariant(monkeypatch, variant):
    import agent.agents.notification_setup.agent as nsa
    from agent.agents.notification_setup.agent import NotificationSetupAgent

    async def fake_extract(*a, **k):
        return variant.model_copy(deep=True)

    monkeypatch.setattr(nsa, "extract_notification_decision", fake_extract)
    monkeypatch.setattr(nsa, "get_extraction_llm", lambda: object())

    out = await NotificationSetupAgent().run(
        {
            "messages": [
                {"role": "assistant", "content": "Is 555-987-6543 the right number for SMS updates?"},
                {"role": "user", "content": "actually email me instead"},
            ],
            "awaiting_slot": "phone_confirmed",
            "call_intent": "claim_services",
            "member_status_verify": True,
            "notification_channel": "sms",
            "phone_number": "5559876543",
            "email": "emily@example.com",
            "parked_followups": [],
        }
    )
    signature = {
        "awaiting_slot": out.get("awaiting_slot"),
        "notification_channel": out.get("notification_channel"),
        "msg_class": _msg_class(out),
    }
    assert signature == {
        "awaiting_slot": "email_confirmed",
        "notification_channel": "email",
        "msg_class": "email_confirm",
    }
