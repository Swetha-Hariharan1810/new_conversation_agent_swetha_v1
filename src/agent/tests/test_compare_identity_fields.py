"""Unit tests for compare_identity_fields (pure field comparison helper).

Phase 1: storage-layer building block only — no agent behavior is exercised.
"""

from agent.storage.queries.members import compare_identity_fields

# A representative member record as returned by get_member_contact():
# names stored Title-Cased, dob stored in Salesforce ISO (YYYY-MM-DD) form.
RECORD = {
    "member_id": "M123456",
    "first_name": "James",
    "last_name": "Anderson",
    "dob": "1977-07-13",
}


def test_all_fields_match():
    result = compare_identity_fields(
        RECORD, first_name="James", last_name="Anderson", dob="07/13/1977"
    )
    assert result == {"first_name": True, "last_name": True, "dob": True}


def test_single_dob_mismatch():
    result = compare_identity_fields(
        RECORD, first_name="James", last_name="Anderson", dob="08/13/1977"
    )
    assert result == {"first_name": True, "last_name": True, "dob": False}


def test_single_last_name_mismatch():
    result = compare_identity_fields(
        RECORD, first_name="James", last_name="Andersen", dob="07/13/1977"
    )
    assert result == {"first_name": True, "last_name": False, "dob": True}


def test_multi_field_mismatch():
    result = compare_identity_fields(
        RECORD, first_name="Robert", last_name="Smith", dob="01/01/1980"
    )
    assert result == {"first_name": False, "last_name": False, "dob": False}


def test_case_insensitive_name_equality():
    # "JAMES" vs stored "James" must not be flagged as a mismatch.
    result = compare_identity_fields(
        RECORD, first_name="JAMES", last_name="anderson", dob="07/13/1977"
    )
    assert result == {"first_name": True, "last_name": True, "dob": True}


def test_format_insensitive_dob_equality():
    # "7/13/1977" (no leading zero) vs stored ISO "1977-07-13" must match.
    result = compare_identity_fields(
        RECORD, first_name="James", last_name="Anderson", dob="7/13/1977"
    )
    assert result["dob"] is True


def test_empty_field_is_not_a_mismatch():
    # An unprovided field is "not yet provided", not a mismatch → reports True.
    result = compare_identity_fields(
        RECORD, first_name="James", last_name="", dob=""
    )
    assert result == {"first_name": True, "last_name": True, "dob": True}


def test_none_record_treats_provided_fields_as_mismatch():
    # No record fetched: provided fields cannot match an empty record.
    result = compare_identity_fields(
        None, first_name="James", last_name="Anderson", dob="07/13/1977"
    )
    assert result == {"first_name": False, "last_name": False, "dob": False}
