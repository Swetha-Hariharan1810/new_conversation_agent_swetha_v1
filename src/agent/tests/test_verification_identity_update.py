"""
Verification: identity updates mid-collection are always honored in-flow
(Phase 3 — BUG-4).

The transcript turn: awaiting member_id, the caller answers AND asks to
update their last name — "m nine zero seven five zero three — oh, also I
need to update my last name". Regardless of how the extraction LLM labeled
the turn, the pipeline must (a) confirm the captured member_id, (b) open an
"answer"-flavor detour asking for the new last name NOW, (c) point
correction_return_to at the next pending identity slot (dob), and (d) clear
member_status_verify when verification had completed. Park ("in just a
moment") and decline ("a representative") can NEVER fire for identity slots
owned in_flow by the active verification agent.

These tests run the REAL VerificationAgent.run → identity pipeline →
_collect_slot → _handle_answered_followup with the real _NORMALIZERS /
_VALIDATORS — only the extraction LLM call and the LLM-2 message generator
are faked.
"""

import logging

import pytest

from agent.agents.verification.agent import VerificationAgent
from agent.llm.schema import EventType, FollowupDisposition, RequestKind, WorkerResult

TRANSCRIPT = "m nine zero seven five zero three — oh, also I need to update my last name"
SPOKEN_MEMBER_ID = "m nine zero seven five zero three"

# ── harness ──────────────────────────────────────────────────────────────────


def _mk_agent(monkeypatch, result: WorkerResult, captured: dict | None = None) -> VerificationAgent:
    import agent.agents.verification.agent as va
    import agent.llm.response_generator as rg

    async def fake_extract(*a, **k):
        return result

    async def fake_generate(**kwargs):
        if captured is not None:
            captured.setdefault("calls", []).append(kwargs)
            captured.update(kwargs)
        return "Got it — and what should the new value be?"

    monkeypatch.setattr(va, "extract_verification_decision", fake_extract)
    monkeypatch.setattr(va, "get_extraction_llm", lambda: object())
    monkeypatch.setattr(rg, "generate_recovery_message", fake_generate)
    return VerificationAgent()


def _state(user=TRANSCRIPT, awaiting="member_id", **over):
    state = {
        "messages": [
            {"role": "assistant", "content": "Could I have your member ID?"},
            {"role": "user", "content": user},
        ],
        "awaiting_slot": awaiting,
        "first_name": "Emily",
        "last_name": "Carter",
        "name_confirmed": True,
        "call_intent": "provider_services",
        "parked_followups": [],
        "ambiguous_counts": {},
    }
    state.update(over)
    return state


def _assert_last_name_detour(out: dict):
    assert out["is_interrupt"] is True
    # (b) the detour asks for the new last name NOW
    assert out["awaiting_slot"] == "last_name"
    assert out["last_name"] == ""  # cleared for re-collection
    # (c) verification resumes exactly where it left off
    assert out["correction_return_to"] == "dob"
    # (a) the captured member_id answer is confirmed and carried forward
    assert out["member_id"] == "M907503"
    # cascade table must NOT fire on the detour path
    assert out["first_name"] == "Emily"


# ── BUG-4: answer + identity-update request, all extraction variants ─────────


class TestBug4IdentityUpdate:
    @pytest.mark.parametrize(
        "result",
        [
            # Ideal extraction (prompt-level contract)
            WorkerResult(
                extracted={"member_id": SPOKEN_MEMBER_ID},
                event_type=EventType.ANSWERED_WITH_FOLLOWUP,
                update_target="last_name",
                request_kind=RequestKind.UPDATE,
            ),
            # Followup flagged but target dropped — backfill from followup_query
            WorkerResult(
                extracted={"member_id": SPOKEN_MEMBER_ID},
                event_type=EventType.ANSWERED_WITH_FOLLOWUP,
                followup_query="I need to update my last name",
                followup_disposition=FollowupDisposition.PARK,
            ),
            # Followup flagged, no query either — backfill from the raw turn
            WorkerResult(
                extracted={"member_id": SPOKEN_MEMBER_ID},
                event_type=EventType.ANSWERED_WITH_FOLLOWUP,
            ),
            # LLM chose park/decline WITH the target — the detour invariant wins
            WorkerResult(
                extracted={"member_id": SPOKEN_MEMBER_ID},
                event_type=EventType.ANSWERED_WITH_FOLLOWUP,
                update_target="last_name",
                request_kind=RequestKind.UPDATE,
                followup_disposition=FollowupDisposition.PARK,
                followup_query="update last name",
            ),
            WorkerResult(
                extracted={"member_id": SPOKEN_MEMBER_ID},
                event_type=EventType.ANSWERED_WITH_FOLLOWUP,
                update_target="last_name",
                request_kind=RequestKind.UPDATE,
                followup_disposition=FollowupDisposition.DECLINE,
            ),
            # Event flattened to ANSWERED — reconcile fills the target and the
            # valid-value path still routes to the followup handler
            WorkerResult(extracted={"member_id": SPOKEN_MEMBER_ID}),
            # Mislabeled CORRECTED with the value — same routing
            WorkerResult(
                extracted={"member_id": SPOKEN_MEMBER_ID},
                event_type=EventType.CORRECTED,
            ),
        ],
    )
    async def test_transcript_turn_opens_detour(self, monkeypatch, result):
        captured: dict = {}
        agent = _mk_agent(monkeypatch, result, captured)
        out = await agent.run(_state())
        _assert_last_name_detour(out)
        # Park/decline messaging can NEVER be produced for in_flow identity
        # slots of the active agent — the detour speaks FOLLOWUP_ANSWER.
        guards_used = [c.get("guard") for c in captured.get("calls", [])]
        assert guards_used == ["FOLLOWUP_ANSWER"]

    async def test_detour_clears_member_status_verify(self, monkeypatch):
        result = WorkerResult(
            extracted={"member_id": SPOKEN_MEMBER_ID},
            event_type=EventType.ANSWERED_WITH_FOLLOWUP,
            update_target="last_name",
            request_kind=RequestKind.UPDATE,
        )
        agent = _mk_agent(monkeypatch, result, {})
        out = await agent.run(_state(member_status_verify=True))
        _assert_last_name_detour(out)
        # (d) a completed verification is invalidated by the identity update
        assert out["member_status_verify"] is False

    async def test_park_decline_never_fire_for_in_flow_identity(self, monkeypatch, caplog):
        # Regression guard for the invariant: even when the LLM explicitly
        # says park or decline, no FOLLOWUP_PARK/FOLLOWUP_DECLINE generation
        # ever runs for an identity target the active agent owns in_flow, no
        # parked_followups entry is created, and the mismatch warning stays
        # silent (resolution was "allow", not a registry mismatch).
        for disposition in (FollowupDisposition.PARK, FollowupDisposition.DECLINE):
            captured: dict = {}
            result = WorkerResult(
                extracted={"member_id": SPOKEN_MEMBER_ID},
                event_type=EventType.ANSWERED_WITH_FOLLOWUP,
                update_target="last_name",
                request_kind=RequestKind.UPDATE,
                followup_disposition=disposition,
                followup_query="update my last name",
            )
            agent = _mk_agent(monkeypatch, result, captured)
            with caplog.at_level(logging.WARNING):
                out = await agent.run(_state())
            guards_used = [c.get("guard") for c in captured.get("calls", [])]
            assert "FOLLOWUP_PARK" not in guards_used
            assert "FOLLOWUP_DECLINE" not in guards_used
            assert not out.get("parked_followups")
            assert "in just a moment" not in out["messages"]["content"].lower()
            assert "representative" not in out["messages"]["content"].lower()

    async def test_bare_update_request_detours_without_answer(self, monkeypatch):
        # "hold on — I need to update my last name" with NO member_id answer:
        # C2 path (bare request). Covers the ANSWERED and WAIT mislabels —
        # reconcile upgrades both to CORRECTED with the backfilled target.
        for result in (WorkerResult(), WorkerResult(event_type=EventType.WAIT)):
            agent = _mk_agent(monkeypatch, result, {})
            out = await agent.run(_state(user="hold on — I need to update my last name"))
            assert out["awaiting_slot"] == "last_name"
            assert out["last_name"] == ""
            # Bare request: the detour returns to the slot we were collecting.
            assert out["correction_return_to"] == "member_id"

    async def test_dob_answer_plus_member_id_update_keeps_dob(self, monkeypatch):
        # Cascade rule check: the detour path must not trigger the
        # member_id → dob cascade — the dob captured this turn survives.
        result = WorkerResult(
            extracted={"dob": "april twelfth nineteen eighty eight"},
            event_type=EventType.ANSWERED_WITH_FOLLOWUP,
            update_target="member_id",
            request_kind=RequestKind.UPDATE,
        )
        agent = _mk_agent(monkeypatch, result, {})
        out = await agent.run(
            _state(
                user="april twelfth nineteen eighty eight — oh and my member id is wrong",
                awaiting="dob",
                member_id="M907503",
            )
        )
        assert out["awaiting_slot"] == "member_id"
        assert out["member_id"] == ""  # only the detour target is cleared
        assert out["dob"] == "04/12/1988"  # the captured answer survives
        assert out["first_name"] == "Emily"
        assert out["last_name"] == "Carter"

    async def test_last_name_update_never_clears_first_name(self, monkeypatch):
        # Explicit cascade-table check: first_name → last_name exists, but
        # the reverse must not — updating last_name keeps first_name intact.
        result = WorkerResult(
            extracted={"member_id": SPOKEN_MEMBER_ID},
            event_type=EventType.ANSWERED_WITH_FOLLOWUP,
            update_target="last_name",
            request_kind=RequestKind.UPDATE,
        )
        agent = _mk_agent(monkeypatch, result, {})
        out = await agent.run(_state())
        cleared = [k for k in ("first_name", "member_id", "dob") if out.get(k, None) == ""]
        assert cleared == []
        assert out["first_name"] == "Emily"

    async def test_plain_answer_proceeds_without_detour(self, monkeypatch):
        # Control: a clean member_id answer advances to dob — no detour, no
        # correction_return_to, no request machinery involved.
        result = WorkerResult(extracted={"member_id": SPOKEN_MEMBER_ID})
        agent = _mk_agent(monkeypatch, result, {})
        out = await agent.run(_state(user=SPOKEN_MEMBER_ID))
        assert out["awaiting_slot"] == "dob"
        assert out["member_id"] == "M907503"
        assert not out.get("correction_return_to")
