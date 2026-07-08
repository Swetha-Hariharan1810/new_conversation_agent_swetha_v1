"""
Deterministic request-detection layer (Phase 1 — fixes the instability root
cause: the extraction LLM intermittently drops update_target/request_kind or
labels a correction turn WAIT).

Covers:
  - detect_request keyword/regex tables: per-slot updates (registry-derived
    patterns + aliases + hand-written "I moved"), redo delivery, replay
    benefits/provider_list
  - precedence: update beats redo beats replay
  - exhaustive negatives: plain answers, bare yes/no, wait-only phrases,
    cannot-provide statements
  - registry derivation: every SLOT_OWNERSHIP key gets baseline coverage
  - reconcile_worker_result: fallback fills gaps, never overrides a concrete
    LLM detection, WAIT veto downgrades to CORRECTED / ANSWERED_WITH_FOLLOWUP
  - agent.utils.detect_wait_request returns False on correction turns
  - slot_manager wiring (BUG-4 / BUG-5 style variants): missing update_target
    is backfilled on the C2 bare-request path and in _handle_answered_followup,
    and the detour/route path wins over the LLM's park/decline disposition
"""

import logging
from types import SimpleNamespace

import pytest

from agent.conversation.context import ConversationContext
from agent.core.guards import ConversationGuardsMixin
from agent.core.request_detection import (
    SLOT_LABEL_ALIASES,
    DetectedRequest,
    detect_request,
    reconcile_worker_result,
)
from agent.core.signals import SignalsMixin
from agent.core.slot_manager import SlotManagerMixin, _InternalSlotConfig
from agent.core.slot_ownership import SLOT_OWNERSHIP
from agent.llm.schema import EventType, RequestKind, WorkerResult
from agent.utils import detect_wait_request

# ── fakes ────────────────────────────────────────────────────────────────────


class _FakeAgent(ConversationGuardsMixin, SlotManagerMixin, SignalsMixin):
    AGENT_NAME = "delivery_management_agent"
    SUPPORTED_TOPICS: set = set()

    def __init__(self):
        self.logger = logging.getLogger("test_fake_agent")
        self._slots = {}
        self._newly_confirmed = set()
        self._pending_ambiguous_resets = set()


def _ctx(confirmed=()):
    return ConversationContext(confirmed_slots=list(confirmed))


def _slot_cfg(slot_name="fax_confirmed"):
    return _InternalSlotConfig(
        slot_name=slot_name,
        prompt=f"{slot_name}?",
        normalizer=str,
        validator=lambda v: SimpleNamespace(valid=bool(v)),
    )


# ── detect_request: update table ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,target",
    [
        # zip_code
        ("my zip code changed", "zip_code"),
        ("the zip is different", "zip_code"),
        ("my zip code is wrong", "zip_code"),
        ("update my zip", "zip_code"),
        ("can you change my zip code", "zip_code"),
        ("i have a new zip", "zip_code"),
        ("new zip code", "zip_code"),
        ("i moved", "zip_code"),
        ("i just moved", "zip_code"),
        ("i've recently moved", "zip_code"),
        ("we just moved", "zip_code"),
        ("update my postal code", "zip_code"),
        # last_name / first_name
        ("change my last name", "last_name"),
        ("my last name is wrong", "last_name"),
        ("correct my last name please", "last_name"),
        ("update my first name", "first_name"),
        ("my first name is different", "first_name"),
        # member_id
        ("update my member id", "member_id"),
        ("my member id is wrong", "member_id"),
        ("change my member number", "member_id"),
        # dob
        ("correct my date of birth", "dob"),
        ("my birthday is wrong", "dob"),
        ("change my dob", "dob"),
        ("my date of birth is different", "dob"),
        # email
        ("i need to change my email", "email"),
        ("update my email address", "email"),
        ("my email is wrong", "email"),
        ("i have a new email address", "email"),
        # fax
        ("change my fax number", "fax"),
        ("my fax is wrong", "fax"),
        ("update the fax", "fax"),
        # notification_method
        ("change my notification preference", "notification_method"),
        ("update my notification method", "notification_method"),
    ],
)
def test_detect_update(text, target):
    d = detect_request(text)
    assert d is not None, text
    assert d.kind == "update"
    assert d.target == target
    assert d.matched  # phrase captured for logging


def test_every_registry_slot_gets_baseline_coverage():
    # A future SLOT_OWNERSHIP entry must be covered the day it is added:
    # the plain "update my <label>" phrasing detects for every key.
    for slot in SLOT_OWNERSHIP:
        label = slot.replace("_", " ")
        d = detect_request(f"i need to update my {label}")
        assert d is not None, slot
        assert d.kind == "update"
        assert d.target == slot


def test_aliases_reference_registry_slots():
    assert set(SLOT_LABEL_ALIASES) <= set(SLOT_OWNERSHIP)


# ── detect_request: redo table ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "actually send that list to my email instead of fax",
        "send it to my email instead",
        "can you email it instead",
        "resend that to my fax instead",
        "send it by email instead",
        "by fax instead please",
        "to my email instead",
        "instead of fax",
        "instead of the email",
        "use the other method",
        "use a different method",
        "actually email is better",
        "actually, fax works better",
        "can you resend that",
        "re-send the list",
        "send it again",
    ],
)
def test_detect_redo_delivery(text):
    d = detect_request(text)
    assert d is not None, text
    assert d.kind == "redo"
    assert d.target == "delivery"


# ── detect_request: replay table ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,target",
    [
        ("repeat my benefits", "benefits"),
        ("can you repeat the benefits", "benefits"),
        ("what are my benefits", "benefits"),
        ("what were my benefits again", "benefits"),
        ("go over my benefits one more time", "benefits"),
        ("tell me my benefits again", "benefits"),
        ("what did you send", "provider_list"),
        ("what exactly did you send me?", "provider_list"),
        ("read that back", "provider_list"),
    ],
)
def test_detect_replay(text, target):
    d = detect_request(text)
    assert d is not None, text
    assert d.kind == "replay"
    assert d.target == target


# ── precedence ───────────────────────────────────────────────────────────────


def test_update_beats_redo():
    d = detect_request("change my email and then resend the list")
    assert d == DetectedRequest(kind="update", target="email", matched=d.matched)


def test_redo_beats_replay():
    d = detect_request("resend the list — what did you send before?")
    assert d.kind == "redo"
    assert d.target == "delivery"


def test_concrete_slot_beats_capability_topic():
    # "fax" names both a slot and the delivery capability; naming the slot
    # with an update verb resolves to the slot, not the redo topic.
    d = detect_request("i need to update my fax number")
    assert d.kind == "update"
    assert d.target == "fax"


# ── negatives: plain answers, yes/no, waits, cannot-provide ──────────────────


@pytest.mark.parametrize(
    "text",
    [
        None,
        "",
        "   ",
        "yes",
        "no",
        "yes that's correct",
        "yeah that's right",
        "nope",
        "m nine zero seven five zero three",
        "M451982",
        "april twelfth nineteen ninety",
        "hold on let me grab my card",
        "give me a minute",
        "just a sec",
        "wait",
        "hi there",
        "not sure",
        # cannot-provide — must never classify as a request
        "i don't have my member id",
        "i don't have it",
        "no, i don't know it",
        "i can't remember my zip code",
        "i lost my card",
        "i misplaced the letter with my member id",
        "i never received one",
        "it's not with me right now",
        "i don't have access to my email",
        # meta-questions about a promised update's timing/status
        "when will you update my zip code?",
        "when are you going to change my email?",
        "have you updated my zip yet?",
        "did you change my email already?",
    ],
)
def test_detect_request_negatives(text):
    assert detect_request(text) is None


# ── reconcile_worker_result ──────────────────────────────────────────────────


def test_reconcile_fills_gap_when_llm_missed():
    result = WorkerResult()  # no update_target, request_kind none
    out = reconcile_worker_result(result, "wait — my zip code changed, i moved")
    assert out.update_target == "zip_code"
    assert out.request_kind == RequestKind.UPDATE


def test_reconcile_fills_redo():
    out = reconcile_worker_result(WorkerResult(), "send it to my email instead")
    assert out.update_target == "delivery"
    assert out.request_kind == RequestKind.REDO


def test_reconcile_never_overrides_concrete_llm_target():
    result = WorkerResult(update_target="email", request_kind=RequestKind.UPDATE)
    out = reconcile_worker_result(result, "my zip code changed")
    assert out.update_target == "email"
    assert out.request_kind == RequestKind.UPDATE


def test_reconcile_keeps_partial_llm_detection():
    # LLM set a kind but no target (or vice versa): still an LLM detection —
    # the regex must not rewrite it.
    result = WorkerResult(request_kind=RequestKind.REDO)
    out = reconcile_worker_result(result, "my zip code changed")
    assert out.request_kind == RequestKind.REDO
    assert not out.update_target


def test_reconcile_noop_when_nothing_detected():
    result = WorkerResult(event_type=EventType.WAIT)
    out = reconcile_worker_result(result, "hold on a second")
    assert out.event_type == EventType.WAIT  # genuine wait untouched
    assert not out.update_target
    assert out.request_kind == RequestKind.NONE


def test_reconcile_wait_veto_bare_correction():
    result = WorkerResult(event_type=EventType.WAIT)
    out = reconcile_worker_result(result, "wait, actually my zip changed")
    assert out.event_type == EventType.CORRECTED  # C2 path handles it
    assert out.update_target == "zip_code"


def test_reconcile_wait_veto_with_value_downgrades_to_followup():
    result = WorkerResult(event_type=EventType.WAIT, extracted={"fax": "5551234567"})
    out = reconcile_worker_result(result, "hold on — my zip is wrong by the way")
    assert out.event_type == EventType.ANSWERED_WITH_FOLLOWUP  # value wins


# ── detect_wait_request veto (agent.utils) ───────────────────────────────────


def test_wait_request_false_on_correction_turn():
    assert detect_wait_request("hold on, new zip") is False
    assert detect_wait_request("wait, my email is wrong") is False


def test_wait_request_still_true_on_genuine_wait():
    assert detect_wait_request("hold on") is True
    assert detect_wait_request("give me a minute") is True


# ── slot_manager wiring: C2 bare-request backfill (BUG-4 style) ──────────────


@pytest.mark.parametrize(
    "utterance",
    [
        "wait — my zip code changed, i moved",
        "actually i just moved",
        "my zip is wrong",
        "hold on, i need to update my zip code",
    ],
)
async def test_c2_backfill_routes_zip_update(utterance):
    # LLM said CORRECTED but returned neither corrections nor update_target —
    # previously this downgraded to ANSWERED and burned a retry on the
    # awaiting slot. The regex backfill recovers the target and routes.
    agent = _FakeAgent()
    decision = SimpleNamespace(
        event_type=SimpleNamespace(value="corrected"),
        corrections={},
        update_target="",
        extracted={},
    )
    state = {"awaiting_slot": "fax_confirmed", "messages": [], "parked_followups": []}
    value, interrupt = await agent._collect_slot(
        state,
        _slot_cfg("fax_confirmed"),
        [{"role": "user", "content": utterance}],
        "",
        decision=decision,
        slot_configs={"fax": None, "email": None},
    )
    assert value is None
    assert interrupt["next_node"] == "provider_search_agent"
    assert interrupt["pending_cross_agent_request"]["kind"] == "update"
    assert interrupt["pending_cross_agent_request"]["target"] == "zip_code"
    assert interrupt["pending_cross_agent_request"]["return_awaiting"] == "fax_confirmed"


async def test_c2_backfill_detours_own_slot(monkeypatch):
    # Bare "change my email" while delivery awaits fax confirmation: email is
    # collected by this pipeline and has a value → detour, not decline.
    async def fake_generate(**kwargs):
        return "Sure — what's the new email?"

    import agent.llm.response_generator as rg

    monkeypatch.setattr(rg, "generate_recovery_message", fake_generate)

    agent = _FakeAgent()
    decision = SimpleNamespace(
        event_type=SimpleNamespace(value="corrected"),
        corrections={},
        update_target="",
        extracted={},
    )
    state = {
        "awaiting_slot": "fax_confirmed",
        "email": "old@example.com",
        "messages": [],
        "parked_followups": [],
    }
    value, interrupt = await agent._collect_slot(
        state,
        _slot_cfg("fax_confirmed"),
        [{"role": "user", "content": "i need to change my email"}],
        "",
        decision=decision,
        slot_configs={"email": None},
    )
    assert value is None
    assert interrupt["awaiting_slot"] == "email"
    assert interrupt["correction_return_to"] == "fax_confirmed"
    assert interrupt["email"] == ""  # cleared for re-collection


async def test_c2_plain_non_answer_still_downgrades(monkeypatch):
    # No request in the caller's words → the empty-CORRECTED downgrade to
    # ANSWERED is unchanged (retry path, no routing).
    async def fake_generate(**kwargs):
        return "Sorry, could you confirm that fax number?"

    import agent.llm.response_generator as rg

    monkeypatch.setattr(rg, "generate_recovery_message", fake_generate)

    agent = _FakeAgent()
    decision = SimpleNamespace(
        event_type=SimpleNamespace(value="corrected"),
        corrections={},
        update_target="",
        extracted={},
    )
    state = {"awaiting_slot": "fax_confirmed", "messages": [], "parked_followups": []}
    value, interrupt = await agent._collect_slot(
        state,
        _slot_cfg("fax_confirmed"),
        [{"role": "user", "content": "m nine zero seven five zero three"}],
        "",
        decision=decision,
        slot_configs={"fax": None},
    )
    assert value is None
    assert interrupt["awaiting_slot"] == "fax_confirmed"
    assert "next_node" not in interrupt or interrupt.get("next_node") != "provider_search_agent"


# ── slot_manager wiring: _handle_answered_followup backfill (BUG-5 style) ────


async def test_followup_backfill_route_beats_llm_disposition():
    # Caller answers the slot AND mentions a ZIP change; the LLM captured the
    # answer but dropped update_target and chose "decline". The backfilled
    # route path must win — never a decline of the caller's own update.
    agent = _FakeAgent()
    ctx = _ctx(confirmed=["first_name"])
    decision = SimpleNamespace(
        extracted={},
        corrections={},
        update_target="",
        followup_query="",
        followup_disposition="decline",
    )
    _, interrupt = await agent._handle_answered_followup(
        {"awaiting_slot": "dob", "parked_followups": []},
        _slot_cfg("dob"),
        [{"role": "user", "content": "march first 1990 — also my zip code changed, i moved"}],
        "03/01/1990",
        ctx,
        decision=decision,
        pending_slots=["dob"],
        slot_configs={"dob": None},
        collected={},
    )
    assert interrupt["next_node"] == "provider_search_agent"
    assert interrupt["pending_cross_agent_request"]["kind"] == "update"
    assert interrupt["pending_cross_agent_request"]["target"] == "zip_code"
    # The awaiting slot was still confirmed before the hand-off.
    assert agent.get_slot("dob").confirmed


async def test_followup_backfill_from_followup_query():
    agent = _FakeAgent()
    decision = SimpleNamespace(
        extracted={},
        corrections={},
        update_target="",
        followup_query="caller wants to update their zip code",
        followup_disposition="park",
    )
    _, interrupt = await agent._handle_answered_followup(
        {"awaiting_slot": "dob", "parked_followups": []},
        _slot_cfg("dob"),
        [{"role": "user", "content": "march first 1990, and my zip changed"}],
        "03/01/1990",
        _ctx(),
        decision=decision,
        pending_slots=["dob"],
        slot_configs={"dob": None},
        collected={},
    )
    assert interrupt["next_node"] == "provider_search_agent"
    assert interrupt["pending_cross_agent_request"]["target"] == "zip_code"


async def test_followup_no_request_keeps_disposition(monkeypatch):
    # Plain side question with no update phrasing: the LLM's disposition is
    # respected exactly as before.
    captured = {}

    async def fake_generate(**kwargs):
        captured.update(kwargs)
        return "Good question — I'll get to that."

    import agent.llm.response_generator as rg

    monkeypatch.setattr(rg, "generate_recovery_message", fake_generate)

    agent = _FakeAgent()
    decision = SimpleNamespace(
        extracted={},
        corrections={},
        update_target="",
        followup_query="does my plan cover massages",
        followup_disposition="park",
    )
    _, interrupt = await agent._handle_answered_followup(
        {"awaiting_slot": "dob", "parked_followups": []},
        _slot_cfg("dob"),
        [{"role": "user", "content": "march first 1990 — does my plan cover massages?"}],
        "03/01/1990",
        _ctx(),
        decision=decision,
        pending_slots=["dob"],
        slot_configs={"dob": None},
        collected={},
    )
    assert captured["guard"] == "FOLLOWUP_PARK"
    assert interrupt["parked_followups"][0]["kind"] == "question"


async def test_followup_decline_updatable_slot_logs_warning(monkeypatch, caplog):
    # Invariant: decline is only legitimate for human_only/unknown ownership.
    # An in_flow slot owned by THIS agent with no value to correct falls to
    # decline — the mismatch is loudly logged.
    async def fake_generate(**kwargs):
        return "I can help with the fax first."

    import agent.llm.response_generator as rg

    monkeypatch.setattr(rg, "generate_recovery_message", fake_generate)

    agent = _FakeAgent()  # delivery_management owns email in_flow
    decision = SimpleNamespace(
        extracted={},
        corrections={},
        update_target="email",  # no email value anywhere → not "allow"
        followup_query="",
        followup_disposition="answer_now",
    )
    with caplog.at_level(logging.WARNING, logger="test_fake_agent"):
        await agent._handle_answered_followup(
            {"awaiting_slot": "dob", "parked_followups": []},
            _slot_cfg("dob"),
            [{"role": "user", "content": "march first 1990"}],
            "03/01/1990",
            _ctx(),
            decision=decision,
            pending_slots=["dob"],
            slot_configs={"dob": None},
            collected={},
        )
    assert any("resolution/registry mismatch" in r.message for r in caplog.records)
