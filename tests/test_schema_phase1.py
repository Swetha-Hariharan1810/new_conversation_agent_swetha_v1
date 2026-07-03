"""Schema round-trip tests for the Phase 1 follow-up groundwork:

* ``EventType.WAIT``
* ``FollowupDisposition``
* ``WorkerResult.followup_disposition`` / ``followup_query`` / ``update_target``

Run with:  pytest tests/test_schema_phase1.py
"""

import json

import pytest
from pydantic import ValidationError

from agent.llm.schema import EventType, FollowupDisposition, WorkerResult


class TestWorkerResultDefaults:
    def test_constructs_with_no_arguments(self):
        result = WorkerResult()
        assert result.extracted is None
        assert result.corrections is None
        assert result.event_type == EventType.ANSWERED
        assert result.followup_disposition == FollowupDisposition.NONE
        assert result.followup_query is None
        assert result.update_target is None

    def test_old_prompt_payload_without_new_fields_still_validates(self):
        """A JSON payload shaped like the pre-Phase-1 prompt output must parse."""
        old_payload = {
            "extracted": {"first_name": "James"},
            "corrections": None,
            "event_type": "answered",
            "guard": "NONE",
            "guard_confidence": 0.0,
        }
        result = WorkerResult.model_validate(old_payload)
        assert result.followup_disposition == FollowupDisposition.NONE
        assert result.followup_query is None
        assert result.update_target is None

    def test_extra_fields_still_forbidden(self):
        with pytest.raises(ValidationError):
            WorkerResult.model_validate({"event_type": "answered", "bogus_field": "x"})


class TestNewFieldRoundTrip:
    def test_round_trip_all_new_fields_and_wait_event(self):
        payload = {
            "event_type": "wait",
            "followup_disposition": "park",
            "followup_query": "what's my deductible?",
            "update_target": "zip_code",
        }
        result = WorkerResult.model_validate_json(json.dumps(payload))
        assert result.event_type == EventType.WAIT
        assert result.followup_disposition == FollowupDisposition.PARK
        assert result.followup_query == "what's my deductible?"
        assert result.update_target == "zip_code"

        dumped = json.loads(result.model_dump_json())
        assert dumped["event_type"] == "wait"
        assert dumped["followup_disposition"] == "park"
        assert dumped["followup_query"] == "what's my deductible?"
        assert dumped["update_target"] == "zip_code"

        assert WorkerResult.model_validate(dumped) == result

    @pytest.mark.parametrize("value", ["answer_now", "park", "decline", "none"])
    def test_every_disposition_value_round_trips(self, value):
        result = WorkerResult.model_validate({"followup_disposition": value})
        assert result.followup_disposition == FollowupDisposition(value)
        assert json.loads(result.model_dump_json())["followup_disposition"] == value

    def test_wait_is_a_valid_event_type_member(self):
        assert EventType("wait") is EventType.WAIT

    def test_invalid_disposition_rejected(self):
        with pytest.raises(ValidationError):
            WorkerResult.model_validate({"followup_disposition": "later"})


class TestStateKeys:
    def test_new_keys_declared_on_state(self):
        from agent.state import State

        annotations = State.__annotations__
        assert annotations["parked_followups"] == list[str]
        assert annotations["wait_count"] is int

    def test_reset_for_new_intent_clears_new_keys(self):
        from agent.state import reset_for_new_intent

        updates = reset_for_new_intent({}, "provider_search")
        assert updates["parked_followups"] == []
        assert updates["wait_count"] == 0
