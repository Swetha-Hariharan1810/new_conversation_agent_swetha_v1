"""Unit tests for the lookup_member storage tool (Phase 2).

Covers the three lookup outcomes while keeping the existing
``{"verified": bool, ...}`` contract intact. The queries layer is mocked so
no Salesforce I/O occurs; the tool lazily imports its query functions from
``agent.storage.queries.members``, so patches are applied there.
"""

import agent.storage.queries.members as members_q
from agent.storage.tools import lookup_member

# Member-ID-only record as returned by get_member_contact():
# stored Title-Cased names, dob in Salesforce ISO (YYYY-MM-DD) form.
ID_RECORD = {
    "member_id": "M123456",
    "first_name": "James",
    "last_name": "Anderson",
    "dob": "1977-07-13",
    "phone_number": "4155551234",
    "zip_code": "94105",
    "fax": "",
    "email": "james@example.com",
    "relationship": "plan_holder",
}


async def _async_return(value):
    return value


def _patch_queries(mocker, *, full_match, id_record):
    """Patch the two query functions the tool calls."""

    async def fake_find(*, member_id, first_name, last_name, dob):
        return full_match

    async def fake_get_contact(member_id):
        return id_record

    mocker.patch.object(members_q, "find_member_by_identity", side_effect=fake_find)
    mocker.patch.object(members_q, "get_member_contact", side_effect=fake_get_contact)


async def test_full_match_returns_verified_true(mocker):
    _patch_queries(mocker, full_match=ID_RECORD, id_record=ID_RECORD)

    result = await lookup_member.ainvoke(
        {
            "member_id": "M123456",
            "first_name": "James",
            "last_name": "Anderson",
            "dob": "07/13/1977",
        }
    )

    assert result["verified"] is True
    assert result["member_id"] == "M123456"
    assert result["phone_number"] == "4155551234"
    assert result["record"] == ID_RECORD
    # Failure-only keys are not present on the success path.
    assert "member_id_found" not in result
    assert "field_matches" not in result


async def test_failed_match_with_record_reports_field_matches(mocker):
    # Full match fails, but the Member ID exists — wrong last name + dob.
    _patch_queries(mocker, full_match=None, id_record=ID_RECORD)

    result = await lookup_member.ainvoke(
        {
            "member_id": "M123456",
            "first_name": "James",
            "last_name": "Andersen",
            "dob": "08/13/1977",
        }
    )

    assert result["verified"] is False
    assert result["member_id_found"] is True
    assert result["field_matches"] == {
        "first_name": True,
        "last_name": False,
        "dob": False,
    }
    assert result["record"] == ID_RECORD


async def test_failed_match_no_record_reports_member_id_not_found(mocker):
    # Full match fails and no record exists for the Member ID at all.
    _patch_queries(mocker, full_match=None, id_record=None)

    result = await lookup_member.ainvoke(
        {
            "member_id": "M000000",
            "first_name": "James",
            "last_name": "Anderson",
            "dob": "07/13/1977",
        }
    )

    assert result == {"verified": False, "member_id_found": False}


async def test_verified_only_callers_unaffected(mocker):
    # The historical contract: every branch always carries a "verified" key,
    # so callers that only read result["verified"] keep working unchanged.
    _patch_queries(mocker, full_match=None, id_record=ID_RECORD)

    result = await lookup_member.ainvoke(
        {
            "member_id": "M123456",
            "first_name": "Wrong",
            "last_name": "Name",
            "dob": "01/01/2000",
        }
    )

    assert result.get("verified") is False
