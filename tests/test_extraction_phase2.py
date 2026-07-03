"""Phase 2 offline tests: extraction input context + prompt example contracts.

Covers:
* ``build_worker_input`` renders the ``Pending:`` context line (new kwarg).
* Every quick-example row added to the extraction headers produces a stub
  LLM payload that parses through the ``WorkerResult`` schema.
* Every extraction wrapper (one per agent llm.py) accepts ``pending_slots``.

Run with:  pytest tests/test_extraction_phase2.py
"""

import inspect

import pytest

from agent.llm.extractor import build_worker_input, remaining_slots
from agent.llm.schema import EventType, FollowupDisposition, WorkerResult


def _user_content(**kwargs) -> str:
    messages = build_worker_input("SYSTEM", **kwargs)
    assert messages[0] == {"role": "system", "content": "SYSTEM"}
    return messages[1]["content"]


class TestPendingContextLine:
    def test_pending_line_rendered_after_confirmed_line(self):
        content = _user_content(
            awaiting_slot="zip_code",
            last_agent_message="What is your ZIP?",
            last_user_message="90210",
            confirmed_slots={"first_name": "James", "last_name": "Smith"},
            pending_slots=["zip_code", "delivery_method"],
        )
        lines = content.splitlines()
        confirmed_idx = next(i for i, ln in enumerate(lines) if ln.startswith("Confirmed: "))
        assert lines[confirmed_idx + 1] == "Pending: zip_code, delivery_method"

    def test_pending_line_omitted_when_none_or_empty(self):
        for pending in (None, []):
            content = _user_content(
                awaiting_slot="zip_code",
                last_agent_message="What is your ZIP?",
                last_user_message="90210",
                confirmed_slots={"first_name": "James"},
                pending_slots=pending,
            )
            assert "Pending:" not in content

    def test_pending_line_without_confirmed_slots(self):
        content = _user_content(
            awaiting_slot="first_name",
            last_agent_message="Your first name?",
            last_user_message="James",
            pending_slots=["first_name", "last_name", "member_id", "dob"],
        )
        lines = content.splitlines()
        asking_idx = next(i for i, ln in enumerate(lines) if ln.startswith("Currently asking for: "))
        assert lines[asking_idx + 1] == "Pending: first_name, last_name, member_id, dob"

    def test_remaining_slots_slices_from_current(self):
        order = ["a", "b", "c"]
        assert remaining_slots(order, "b") == ["b", "c"]
        assert remaining_slots(order, "a") == ["a", "b", "c"]
        assert remaining_slots(order, "not_in_order") == ["a", "b", "c"]
        assert remaining_slots(order, "b") is not order  # always a copy


# Stub payloads the extraction LLM is instructed to emit for each of the
# quick-example rows added to header.md in Phase 2. Each must parse through
# the WorkerResult schema — this is the offline contract for the prompt.
_EXAMPLE_ROW_PAYLOADS = {
    "bare wait": {
        "extracted": {},
        "corrections": {},
        "event_type": "wait",
    },
    "wait with excuse": {
        "extracted": {},
        "corrections": {},
        "event_type": "wait",
        "guard": "NONE",
        "guard_confidence": 0.0,
    },
    "wait plus value — value wins": {
        "extracted": {"member_id": "M451982"},
        "corrections": {},
        "event_type": "answered",
    },
    "answer + question about confirmed slot": {
        "extracted": {"zip_code": "90210"},
        "event_type": "answered_with_followup",
        "followup_disposition": "answer_now",
        "followup_query": "what was my member ID again?",
    },
    "answer + question about pending slot": {
        "extracted": {"zip_code": "90210"},
        "event_type": "answered_with_followup",
        "followup_disposition": "park",
        "followup_query": "will I get a text about this?",
    },
    "answer + irrelevant question": {
        "extracted": {"zip_code": "90210"},
        "event_type": "answered_with_followup",
        "followup_disposition": "decline",
        "followup_query": "do you sell car insurance?",
    },
    "answer + repeat request": {
        "extracted": {"zip_code": "90210"},
        "event_type": "answered_with_followup",
        "followup_disposition": "answer_now",
        "followup_query": "say that again?",
    },
    "update shape 1 — new value, no answer": {
        "extracted": {},
        "corrections": {"last_name": "Smith"},
        "event_type": "corrected",
    },
    "update shape 2 — answer plus new value": {
        "extracted": {"zip_code": "90210"},
        "corrections": {"email": "a@b.com"},
        "event_type": "answered_with_followup",
        "followup_disposition": "answer_now",
        "followup_query": "change my email to a@b.com",
    },
    "update shape 3 — no value given": {
        "extracted": {},
        "corrections": {},
        "event_type": "corrected",
        "update_target": "email",
    },
}


class TestExampleRowSchemaRoundTrip:
    @pytest.mark.parametrize("name", list(_EXAMPLE_ROW_PAYLOADS))
    def test_stub_payload_parses(self, name):
        payload = _EXAMPLE_ROW_PAYLOADS[name]
        result = WorkerResult.model_validate(payload)
        assert result.event_type == EventType(payload["event_type"])
        expected_disposition = FollowupDisposition(payload.get("followup_disposition", "none"))
        assert result.followup_disposition == expected_disposition
        assert result.followup_query == payload.get("followup_query")
        assert result.update_target == payload.get("update_target")


class TestAllWrappersAcceptPendingSlots:
    def test_every_extraction_wrapper_has_pending_slots_kwarg(self):
        from agent.agents.benefits.llm import extract_benefits_decision
        from agent.agents.care_wellness.llm import extract_care_wellness_decision
        from agent.agents.claim_adjustment.llm import extract_claim_adjustment_decision
        from agent.agents.delivery_management.llm import extract_delivery_management_decision
        from agent.agents.follow_up.llm import extract_follow_up_decision
        from agent.agents.intake.llm import extract_intake_intent
        from agent.agents.notification_setup.llm import extract_notification_decision
        from agent.agents.provider_search.llm import extract_provider_search_decision
        from agent.agents.records_coordination.llm import extract_records_decision
        from agent.agents.verification.llm import (
            extract_name_confirmation,
            extract_verification_decision,
        )

        wrappers = [
            extract_benefits_decision,
            extract_care_wellness_decision,
            extract_claim_adjustment_decision,
            extract_delivery_management_decision,
            extract_follow_up_decision,
            extract_intake_intent,
            extract_name_confirmation,
            extract_notification_decision,
            extract_provider_search_decision,
            extract_records_decision,
            extract_verification_decision,
        ]
        for fn in wrappers:
            assert "pending_slots" in inspect.signature(fn).parameters, fn.__qualname__


class TestHeaderPromptContent:
    """The new prompt rules actually ship in the packaged header files."""

    def test_wait_rule_in_all_three_headers(self):
        from agent.utils import read_prompt

        for header in ("header.md", "header_extraction.md", "header_core.md"):
            text = read_prompt(f"extraction/{header}")
            assert "## WAIT" in text, header
            assert 'event_type:"wait"' in text, header

    def test_update_and_disposition_rules_in_full_headers(self):
        from agent.utils import read_prompt

        header = read_prompt("extraction/header.md")
        assert "## UPDATE REQUESTS" in header
        assert "## FOLLOWUP DISPOSITION" in header
        extraction = read_prompt("extraction/header_extraction.md")
        assert "## Update requests" in extraction
        assert "## Followup disposition" in extraction
        core = read_prompt("extraction/header_core.md")
        assert "update_target" not in core  # core (intake) only gets the WAIT rule
