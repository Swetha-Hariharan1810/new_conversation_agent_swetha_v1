"""
scenarios.py — All live E2E scenario definitions.

User utterances are derived from the static transcripts in
scripts/conversational_workload/static_transcripts/ where those exist;
the remaining scripts are written in the same spoken-form style the
normalizers handle ("m nine zero seven five zero three",
"April twelfth nineteen eighty-eight").

Assertions follow the robustness rules: state keys, escalation reasons
(substring/regex over every reason source), metadata events, END/interrupt
flags, and tolerant case-insensitive regexes — never exact AI sentences.
Where wording comes from a constant pool, the pool is imported and matched
via harness.pool_regex().

ZIP-update behavior change (provider_search):
  The ZIP read-back confirmation step ("Just to be sure I have it right —
  your ZIP code is X, correct?") was REMOVED from provider_search_agent.
  A valid new ZIP is now written to Salesforce immediately and the flow
  proceeds straight to the delivery-method question. delivery_management's
  dispatch confirmation then includes the updated ZIP
  ("...list of in-network providers for your current ZIP code X within
  30 minutes"). Consequences for this suite:
    - pcp_zip_update no longer scripts a confirmation turn and asserts
      the ZIP-aware dispatch message + zip_code_updated state flag.
    - pcp_zip_inline_update (new) covers the inline "no, it's X" path.
    - zip_change_loop_escalates was REDEFINED: the zip_change_cycles
      read-back rejection loop it used to exercise no longer exists, so it
      now checks the remaining escalation loop — zip_code slot exhaustion
      when the member repeatedly provides INVALID (non-5-digit) values —
      and verifies no invalid value is ever persisted to Salesforce.
"""

from __future__ import annotations

# Static pools — imported so assertions survive any re-pick of pool members.
from agent.agents.follow_up.constants import MSG_UPDATE_REQUEST_ESCALATE  # noqa: E402
from agent.agents.verification.constants import (  # noqa: E402
    MSG_REASK_DOB,
    MSG_REASK_FIRST_NAME,
    MSG_REASK_GENERIC,
    MSG_REASK_LAST_NAME,
    NAME_CORRECTION_PROMPTS,
)
from agent.agents.verification.handlers import MSG_PHONE_NOT_CONFIRMED, MSG_RESTART  # noqa: E402
from agent.responses.static import MSG_SELF_HARM_ESCALATION, MSG_WAIT_ACK  # noqa: E402
from tests.live_e2e.harness import Expected, Scenario, TurnExpectation, pool_regex

# ──────────────────────────────────────────────────────────────────────────────
# Predicates for Expected.final_state
# ──────────────────────────────────────────────────────────────────────────────


def truthy(v) -> bool:
    return bool(v)


def falsy(v) -> bool:
    return not v


def _digits(v) -> str:
    return "".join(c for c in str(v or "") if c.isdigit())


def contains(sub: str):
    def _pred(v, _sub=sub):
        return _sub.lower() in str(v or "").lower()

    _pred.__name__ = f"contains({sub!r})"
    return _pred


def digits_equal(expected: str):
    def _pred(v, _exp=expected):
        return _digits(v) == _digits(_exp)

    _pred.__name__ = f"digits_equal({expected!r})"
    return _pred


# ──────────────────────────────────────────────────────────────────────────────
# Salesforce post-checks (real re-queries after the conversation ends)
# ──────────────────────────────────────────────────────────────────────────────


def sf_field_check(member_id: str, fld: str, expected: str, compare_digits: bool = False):
    async def _check(_final_state):
        from agent.storage.queries.members import get_member_contact

        record = await get_member_contact(member_id)
        if not record:
            return f"SF post-check: member {member_id} not found on re-query"
        actual = record.get(fld) or ""
        if compare_digits:
            ok = _digits(actual) == _digits(expected)
        else:
            ok = str(actual).strip().lower() == expected.strip().lower()
        if not ok:
            return (
                f"SF post-check: {member_id}.{fld}={actual!r} after run, "
                f"expected {expected!r} — the agent did not persist the update"
            )
        return None

    _check.__name__ = f"sf_{fld}_check"
    return _check


# ──────────────────────────────────────────────────────────────────────────────
# Shared script prefixes
# ──────────────────────────────────────────────────────────────────────────────

# PCP flow: intent → first/last name → member id → dob → relationship
PCP_VERIFY = [
    "I need to find a primary care physician in my area.",
    "emily",
    "carter",
    "yes correct",  # name_confirmed
    "m nine zero seven five zero three",
    "April twelvee nineteen eighty-eight",
    "I'm calling for myself",
]


# Claim flow: intent → first/last name → member id → dob → phone confirmation
CLAIM_VERIFY = [
    "I adjusted the claim and I want to follow up",
    "james",
    "wilson",
    "yes correct",  # name_confirmed
    "m three one zero one eight eight",
    "Thirtieth of July, nineteen seventy seven",
    "yes correct",
]

NEW_EMAIL = "james.w.new@gmail.com"
NEW_EMILY_EMAIL = "emily.c.new@example.com"

# PCP flow: intent → first/last name → member id → dob → relationship (conversational)
PCP_VERIFY_CONVERSATIONAL = [
    "Hi there, yeah, I'm trying to find a primary care doctor near where I live",
    "sure, it's emily",
    "carter, that's c a r t e r",
    "yes thats correct",
    "okay so my member id is m nine zero seven five zero three",
    "I was born on the twelfth of april, nineteen eighty eight",
    "it's my own plan, I'm the plan holder",
]

# Claim flow: intent → first/last name → member id → dob → phone confirmation (conversational)
CLAIM_VERIFY_CONVERSATIONAL = [
    "hello, I submitted a claim adjustment a while back and wanted to check on it",
    "yeah it's james",
    "wilson",
    "yes thats correct",
    "let me grab my card... okay it's m three one zero one eight eight",
    "the thirtieth of july, nineteen seventy seven",
    "yep, that's the right number",
]

# Verification turn-level sanity checks shared by happy paths
_VERIFY_TURNS = {
    4: TurnExpectation(ai_contains=[r"member\s*id"], slot_awaiting="member_id"),
    5: TurnExpectation(ai_contains=[r"(date of birth|birth\s*date|dob)"], slot_awaiting="dob"),
}


# Dispatch confirmation must name the updated ZIP — matches either
# DELIVERY_WINDOW_MSG_ZIP_UPDATED pool member ("current"/"updated" wording).
def _zip_dispatch_regex(zip_code: str) -> str:
    return rf"(current|updated)\s+zip\s*code\s+{zip_code}.*within 30 minutes"


# ──────────────────────────────────────────────────────────────────────────────
# A. Provider (PCP) happy paths
# ──────────────────────────────────────────────────────────────────────────────

pcp_happy_path_fax = Scenario(
    name="pcp_happy_path_fax",
    flow="pcp",
    user_turns=PCP_VERIFY
    + [
        "Primary Care Physician",
        "yes that's correct",  # ZIP on file
        "send it to my fax",
        "yes that's correct",  # fax on file
        "yes please",  # benefits offer
        "yes that sounds interesting",  # Care Coach offer
        "when should I expect to receive the provider list?",  # one follow-up
        "no thanks that was helpful",  # close
    ],
    turn_expectations=_VERIFY_TURNS,
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "provider_list_sent": True,
            "delivery_method": "fax",
            "benefits_explained": True,
            "care_coach_details_sent": True,
        },
    ),
)

pcp_happy_path_email = Scenario(
    name="pcp_happy_path_email",
    flow="pcp",
    user_turns=PCP_VERIFY
    + [
        "Primary Care Physician",
        "yes that's correct",
        "email please",
        "yes that's correct",  # email on file
        "yes please",
        "yes that sounds interesting",
        "no thanks that was helpful",
    ],
    turn_expectations=_VERIFY_TURNS,
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "provider_list_sent": True,
            "delivery_method": "email",
            "benefits_explained": True,
            "care_coach_details_sent": True,
        },
    ),
)

pcp_benefits_declined = Scenario(
    name="pcp_benefits_declined",
    flow="pcp",
    user_turns=PCP_VERIFY
    + [
        "Primary Care Physician",
        "yes that's correct",
        "send it to my fax",
        "yes that's correct",
        "no thanks",  # decline benefits offer → BenefitsAgent NO path
        "no thank you",  # decline Care Coach (no-explanation offer)
        "no, that's everything, thanks",  # follow-up → close
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "provider_list_sent": True,
            "benefits_explained": False,
            "care_coach_nooffer_sent": True,
            "care_coach_details_sent": falsy,
        },
    ),
)

pcp_zip_update = Scenario(
    name="pcp_zip_update",
    flow="pcp",
    mutating=True,
    user_turns=PCP_VERIFY
    + [
        "Primary Care Physician",
        "no, I moved recently",  # decline ZIP on file
        "my new zip code is zero two one three nine",  # spoken 5-digit ZIP — accepted
        # directly: NO read-back confirmation turn anymore; the next AI prompt
        # must already be the delivery-method bridge (asserted below)
        "send it to my fax",
        "yes that's correct",  # fax on file
        "no thanks",
        "no thank you",
        "no that's all, thanks",
    ],
    turn_expectations={
        # The AI prompt that precedes "send it to my fax" must be the delivery
        # bridge — proving the ZIP was accepted with no confirmation step.
        10: TurnExpectation(ai_contains=[r"fax or email"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "provider_list_sent": True,
            "zip_code_used": "02139",
            "zip_code_updated": True,
        },
        transcript_contains=[
            # ZIP-aware dispatch confirmation from DELIVERY_WINDOW_MSG_ZIP_UPDATED
            _zip_dispatch_regex("02139"),
        ],
    ),
    post_checks=[sf_field_check("M907503", "zip_code", "02139")],
    notes=(
        "Mutates Emily's zip in Salesforce; teardown restores the snapshot. "
        "The ZIP read-back confirmation was removed from provider_search: the "
        "new ZIP is written to Salesforce on first hearing and the very next "
        "AI turn is the fax/email delivery question. The dispatch confirmation "
        "must include the updated ZIP (DELIVERY_WINDOW_MSG_ZIP_UPDATED)."
    ),
)

pcp_zip_inline_update = Scenario(
    name="pcp_zip_inline_update",
    flow="pcp",
    mutating=True,
    retries=1,  # inline "no + new ZIP" extraction is mildly non-deterministic:
    # if the LLM returns zip_confirmed="no" instead of the bare zip_code, the
    # agent asks for the ZIP on a separate turn and the script desyncs
    user_turns=PCP_VERIFY
    + [
        "Primary Care Physician",
        # Inline decline + replacement in ONE utterance at the zip_confirmed
        # read-back: extraction contract extracts zip_code and omits
        # zip_confirmed → provider_search accepts it directly (no read-back).
        "no, my zip changed — it's zero two one four zero",
        "email please",  # next AI prompt is already the delivery bridge
        "yes that's correct",  # email on file
        "no thanks",  # decline benefits
        "no thank you",  # decline Care Coach
        "no that's all, thanks",
    ],
    turn_expectations={
        # The AI prompt preceding "email please" must be the delivery bridge —
        # the inline-replacement path must not produce a confirmation read-back.
        9: TurnExpectation(ai_contains=[r"fax or email"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "provider_list_sent": True,
            "delivery_method": "email",
            "zip_code_used": "02140",
            "zip_code_updated": True,
        },
        transcript_contains=[
            _zip_dispatch_regex("02140"),
        ],
    ),
    post_checks=[sf_field_check("M907503", "zip_code", "02140")],
    notes=(
        "Mutates Emily's zip in Salesforce; teardown restores the snapshot. "
        "Covers the zip_confirmed inline-replacement path ('no, it's X' in one "
        "utterance): the new ZIP is persisted immediately, zip_code_updated is "
        "set, and the dispatch confirmation names the new ZIP. Replaces the "
        "deleted zip_change_loop_escalates scenario — with the read-back gone, "
        "the zip_change_cycles loop it exercised can no longer occur."
    ),
)

pcp_fax_update = Scenario(
    name="pcp_fax_update",
    flow="pcp",
    mutating=True,
    user_turns=PCP_VERIFY
    + [
        "Primary Care Physician",
        "yes that's correct",
        "send it to my fax",
        "no, that fax number is outdated",  # decline fax on file
        "my new fax number is six one seven five five five nine one nine nine",
        "yes that's correct",  # confirm read-back → SF write (fax read-back still exists)
        "no thanks",
        "no thank you",
        "no that's all, thanks",
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "provider_list_sent": True,
            "delivery_method": "fax",
            "fax": digits_equal("6175559199"),
        },
    ),
    post_checks=[sf_field_check("M907503", "fax", "6175559199", compare_digits=True)],
    notes="Mutates Emily's fax in Salesforce; teardown restores the snapshot.",
)

pcp_email_update = Scenario(
    name="pcp_email_update",
    flow="pcp",
    mutating=True,
    user_turns=PCP_VERIFY
    + [
        "Primary Care Physician",
        "yes that's correct",  # ZIP on file
        "email please",
        "no, that's my old email address",  # decline email on file
        NEW_EMILY_EMAIL,
        "yes that's correct",  # confirm read-back → SF write (email read-back still exists)
        "no thanks",  # decline benefits offer
        "no thank you",  # decline Care Coach
        "no that's all, thanks",
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "provider_list_sent": True,
            "delivery_method": "email",
            "email": contains(NEW_EMILY_EMAIL),
        },
    ),
    post_checks=[sf_field_check("M907503", "email", NEW_EMILY_EMAIL)],
    notes=(
        "Mutates Emily's email in Salesforce; teardown restores the snapshot. "
        "The agent reads email addresses back with '@' replaced by ' at ' "
        "(Azure content-filter workaround) — assertions use contains() on the "
        "state value and must not depend on a literal '@' in any AI transcript line."
    ),
)

# ──────────────────────────────────────────────────────────────────────────────
# B. Verification escalations
# ──────────────────────────────────────────────────────────────────────────────

verification_restart_then_success = Scenario(
    name="verification_restart_then_success",
    flow="pcp",
    timeout_s=360,
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "carter",
        "yes correct",  # name_confirmed round 1
        "m nine zero seven five zero two",  # wrong member id — lookup fails
        "April twelfth nineteen eighty-eight",
        # agent restarts ("let's try once more") — give correct details
        "emily",
        "carter",
        "yes correct",  # name_confirmed round 2
        "m nine zero seven five zero three",
        "April twelfth nineteen eighty-eight",
        "I'm calling for myself",
        # complete the PCP flow minimally
        "Primary Care Physician",
        "yes that's correct",
        "email please",
        "yes that's correct",
        "no thanks",
        "no thank you",
        "no that's all",
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={"member_status_verify": True},
        # restart message — tolerant alternation covering every MSG_RESTART member
        transcript_contains=[r"(one more try|try once more|once more|try again|didn't quite match)"],
    ),
)

verification_fail_twice_escalates = Scenario(
    name="verification_fail_twice_escalates",
    flow="pcp",
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "carter",
        "yes correct",  # name_confirmed round 1
        "m nine zero seven five zero two",  # wrong, round 1
        "April twelfth nineteen eighty-eight",
        "emily",
        "carter",
        "yes correct",  # name_confirmed round 2
        "m nine zero seven five zero two",  # wrong, round 2
        "April twelfth nineteen eighty-eight",
    ],
    expect=Expected(
        completed=True,  # END via escalation_agent
        escalated=True,
        transfer_event=True,
        escalation_reason_contains="Verification failed",
        final_state={"escalation_reference_number": truthy},
    ),
)

member_id_exhaustion = Scenario(
    name="member_id_exhaustion",
    flow="pcp",
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "carter",
        "yes correct",  # name_confirmed
        "one two three",  # no M prefix
        "I don't know",
        "umm banana",
        # spares in case a turn is classified as clarification (not counted)
        "no idea",
        "I really don't know it",
    ],
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        escalation_reason_contains="member_id",
        transcript_contains=[r"member id after a few tries|wasn't able to capture"],
    ),
)

dob_no_year_exhaustion = Scenario(
    name="dob_no_year_exhaustion",
    flow="pcp",
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "carter",
        "yes correct",  # name_confirmed
        "m nine zero seven five zero three",
        "April twelfth",  # no year — invalid
        "April twelfth",
        "April twelfth",
        # spares for uncounted clarification turns
        "April twelfth",
        "April twelfth",
    ],
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        escalation_reason_contains="dob",
    ),
)

member_id_ambiguous_exhaustion = Scenario(
    name="member_id_ambiguous_exhaustion",
    flow="pcp",
    user_turns=[
        "I wanted to see my primary care doctor.",
        "Sure. Emily?",
        "Carter?",
        "yes correct",  # name_confirmed
        "I don't have it.",  # ask #1 → AMBIGUOUS → slot_fail (fix: threshold >= 1)
        "I don't have it.",  # ask #2 → AMBIGUOUS → slot_fail
        "I don't have it.",  # ask #3 → escalation (was ask #5 before fix)
    ],
    turn_expectations={
        4: TurnExpectation(ai_contains=[r"member\s*(id|ID)"], slot_awaiting="member_id"),
        5: TurnExpectation(ai_contains=[r"member\s*(id|ID)"], slot_awaiting="member_id"),
        6: TurnExpectation(ai_contains=[r"member\s*(id|ID)"], slot_awaiting="member_id"),
    },
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        transfer_initiator="Agent",
        escalation_reason_contains="member_id",
        final_state={"member_status_verify": lambda v: not v},
        last_ai_contains=[r"(member id after a few|wasn't able to capture your member)"],
        max_turns=20,
    ),
    notes=(
        "History: 'I don't have it' used to burn the whole retry budget via the "
        "AMBIGUOUS branch. Today detect_cannot_provide() short-circuits it: the "
        "FIRST 'I don't have it' escalates immediately with reason "
        "member_id_cannot_provide (the cannot-provide check runs before the "
        "ambiguous threshold), so the scripted spares are never consumed. Note "
        "the ambiguous threshold itself is >= 2 again since Phase 7: a first "
        "genuinely-ambiguous turn (not cannot-provide) gets a free CLARIFY, the "
        "second burns an attempt — see followup/wait scenarios in section N."
    ),
)

# ──────────────────────────────────────────────────────────────────────────────
# B2. Partial re-ask on identity mismatch (member found, one field wrong)
#
# These lock the targeted-re-ask behavior: on a failed full match where the
# Member ID exists, only the mismatched identity field is re-asked; matched
# fields and the Member ID are retained, and the spelled-name read-back is not
# repeated when the names already matched.
# ──────────────────────────────────────────────────────────────────────────────

verification_dob_only_mismatch = Scenario(
    name="verification_dob_only_mismatch",
    flow="pcp",
    timeout_s=360,
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "carter",
        "yes correct",  # name readback #1 → confirmed
        "m nine zero seven five zero three",  # correct Member ID
        "April thirteenth nineteen eighty-eight",  # WRONG dob (on file: the 12th)
        # lookup fails: member_id_found=True, only dob mismatches → MSG_REASK_DOB,
        # awaiting_slot="dob". name/last-name/Member ID are NOT re-asked.
        "April twelfth nineteen eighty-eight",  # corrected dob → re-lookup → verified
        "I'm calling for myself",  # relationship
        "Primary Care Physician",
        "yes that's correct",  # ZIP on file
        "email please",
        "yes that's correct",  # email on file
        "no thanks",
        "no thank you",
        "no, that's everything",
    ],
    turn_expectations={
        3: TurnExpectation(ai_contains=[r"E-M-I-L-Y.*C-A-R-T-E-R"]),  # the one read-back
        4: TurnExpectation(ai_contains=[r"member\s*id"], slot_awaiting="member_id"),
        5: TurnExpectation(ai_contains=[r"(date of birth|birth\s*date|dob)"], slot_awaiting="dob"),
        # The re-ask after the failed lookup: the disclosing DOB pool, awaiting dob.
        # Proves ONLY dob is re-asked (no name / Member-ID re-ask).
        6: TurnExpectation(ai_contains=[pool_regex(MSG_REASK_DOB)], slot_awaiting="dob"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "first_name": "Emily",
            "last_name": "Carter",
            "provider_list_sent": True,
        },
        transcript_contains=[r"E-M-I-L-Y"],
        # The spelled-name read-back appears EXACTLY once — name_confirmed is
        # preserved across the DOB-only re-ask, so no second read-back fires.
        transcript_count={r"E-M-I-L-Y": 1},
    ),
    notes=(
        "Mirrors the real transcript: a correct Member ID + name but a wrong DOB. "
        "The lookup returns member_id_found=True with dob mismatched, so the agent "
        "re-asks ONLY the date of birth (MSG_REASK_DOB, awaiting_slot='dob'); first "
        "name, last name and Member ID are retained and never re-asked, and the "
        "spelled-name read-back is delivered exactly once (name_confirmed is "
        "preserved). After one corrected DOB turn the re-lookup matches and "
        "member_status_verify becomes True."
    ),
)

verification_last_name_only_mismatch = Scenario(
    name="verification_last_name_only_mismatch",
    flow="pcp",
    timeout_s=360,
    retries=1,  # name re-confirmation involves LLM extraction (cf. name_confirmation_inline_correction)
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "carson",  # WRONG last name (on file: Carter)
        "yes correct",  # readback #1 → "Emily Carson" confirmed
        "m nine zero seven five zero three",  # correct Member ID
        "April twelfth nineteen eighty-eight",  # correct dob
        # lookup fails: first_name + dob match, last_name mismatches →
        # MSG_REASK_LAST_NAME, awaiting_slot="last_name". Member ID + dob retained.
        "carter",  # corrected last name → fresh read-back of "Emily Carter"
        "yes correct",  # confirm corrected name → straight to lookup (NO Member-ID re-ask)
        "I'm calling for myself",  # relationship (proves we skipped to post-lookup)
        "Primary Care Physician",
        "yes that's correct",  # ZIP on file
        "email please",
        "yes that's correct",  # email on file
        "no thanks",
        "no thank you",
        "no, that's everything",
    ],
    turn_expectations={
        3: TurnExpectation(ai_contains=[r"E-M-I-L-Y.*C-A-R-S-O-N"]),  # read-back #1
        4: TurnExpectation(ai_contains=[r"member\s*id"], slot_awaiting="member_id"),
        5: TurnExpectation(ai_contains=[r"(date of birth|birth\s*date|dob)"], slot_awaiting="dob"),
        # The re-ask after the failed lookup: disclosing LAST-NAME pool, awaiting
        # last_name. Proves ONLY the last name is re-asked (not Member ID / dob).
        6: TurnExpectation(ai_contains=[pool_regex(MSG_REASK_LAST_NAME)], slot_awaiting="last_name"),
        # After confirming the corrected name, the next prompt is the relationship
        # question — NOT a Member-ID re-ask — proving Member ID + dob were retained.
        8: TurnExpectation(ai_contains=[r"plan holder|dependent"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "first_name": "Emily",
            "last_name": "Carter",  # corrected
            "provider_list_sent": True,
        },
        # Both the original and corrected read-backs appear; the corrected last
        # name is read back exactly once.
        transcript_contains=[r"E-M-I-L-Y", r"C-A-R-T-E-R"],
        transcript_count={r"C-A-R-T-E-R": 1},
    ),
    notes=(
        "Last-name-only mismatch: a correct Member ID + first name + DOB but a "
        "wrong last name. The lookup returns member_id_found=True with last_name "
        "mismatched, so the agent re-asks ONLY the last name (MSG_REASK_LAST_NAME, "
        "awaiting_slot='last_name'); Member ID and DOB are retained. Because a name "
        "field mismatched, name_confirmed is reset and the corrected name is read "
        "back once more; on confirmation the flow proceeds straight to the lookup "
        "(via _finish_after_identity) WITHOUT re-asking the already-known Member ID "
        "— turn-8 relationship prompt is the proof. Re-lookup matches → verified."
    ),
)

verification_first_name_only_mismatch = Scenario(
    name="verification_first_name_only_mismatch",
    flow="pcp",
    timeout_s=360,
    retries=1,  # name re-confirmation involves LLM extraction
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emma",  # WRONG first name (on file: Emily)
        "carter",  # correct last name
        "yes correct",  # read-back #1 → "Emma Carter" confirmed
        "m nine zero seven five zero three",  # correct Member ID
        "April twelfth nineteen eighty-eight",  # correct dob
        # lookup fails: last name + dob match, first name mismatches →
        # MSG_REASK_FIRST_NAME, awaiting_slot="first_name". Member ID + dob retained.
        "emily",  # corrected first name → fresh read-back of "Emily Carter"
        "yes correct",  # confirm → straight to lookup (NO Member-ID re-ask)
        "I'm calling for myself",  # relationship
        "Primary Care Physician",
        "yes that's correct",
        "email please",
        "yes that's correct",
        "no thanks",
        "no thank you",
        "no, that's everything",
    ],
    turn_expectations={
        3: TurnExpectation(ai_contains=[r"E-M-M-A.*C-A-R-T-E-R"]),  # read-back #1
        4: TurnExpectation(ai_contains=[r"member\s*id"], slot_awaiting="member_id"),
        5: TurnExpectation(ai_contains=[r"(date of birth|birth\s*date|dob)"], slot_awaiting="dob"),
        # Re-ask after the failed lookup: disclosing FIRST-NAME pool, awaiting first_name.
        6: TurnExpectation(ai_contains=[pool_regex(MSG_REASK_FIRST_NAME)], slot_awaiting="first_name"),
        7: TurnExpectation(ai_contains=[r"E-M-I-L-Y.*C-A-R-T-E-R"]),  # corrected read-back
        8: TurnExpectation(ai_contains=[r"plan holder|dependent"]),  # relationship, not Member-ID
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "first_name": "Emily",  # corrected
            "last_name": "Carter",
            "provider_list_sent": True,
        },
        transcript_contains=[r"E-M-M-A", r"E-M-I-L-Y"],
        transcript_count={r"E-M-I-L-Y": 1},  # corrected first name read back exactly once
    ),
    notes=(
        "First-name-only mismatch: wrong first name, correct last name + Member ID "
        "+ DOB. Lookup returns member_id_found=True with first_name mismatched → only "
        "the first name is re-asked (MSG_REASK_FIRST_NAME, awaiting_slot='first_name'); "
        "Member ID and DOB are retained. name_confirmed resets (a name field changed) "
        "and the cached caller_first_name is cleared; the corrected name is read back "
        "once, then the flow proceeds straight to the lookup (no Member-ID re-ask) → "
        "verified. Turn-8 relationship prompt is the proof."
    ),
)

verification_name_mismatch_bare_no_at_readback = Scenario(
    name="verification_name_mismatch_bare_no_at_readback",
    flow="pcp",
    timeout_s=420,
    retries=2,  # exercises the name-correction sub-loop on top of the partial re-ask
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "carson",  # WRONG last name (on file: Carter)
        "yes correct",  # read-back #1 → "Emily Carson"
        "m nine zero seven five zero three",  # correct Member ID
        "April twelfth nineteen eighty-eight",  # correct dob
        # lookup fails: last name mismatches → MSG_REASK_LAST_NAME
        "carson",  # caller restates the wrong name → read-back "Emily Carson"
        "no",  # BARE NO at the read-back → agent asks for the correct name
        "it's Emily Carter",  # correct name → read-back "Emily Carter"
        "yes correct",  # confirm → straight to lookup → verified
        "I'm calling for myself",  # relationship
        "Primary Care Physician",
        "yes that's correct",
        "email please",
        "yes that's correct",
        "no thanks",
        "no thank you",
        "no, that's everything",
    ],
    turn_expectations={
        # Re-ask after the failed lookup: only the last name.
        6: TurnExpectation(ai_contains=[pool_regex(MSG_REASK_LAST_NAME)], slot_awaiting="last_name"),
        7: TurnExpectation(ai_contains=[r"E-M-I-L-Y.*C-A-R-S-O-N"]),  # read-back of restated wrong name
        8: TurnExpectation(ai_contains=[pool_regex(NAME_CORRECTION_PROMPTS)]),  # after bare "no"
        9: TurnExpectation(ai_contains=[r"E-M-I-L-Y.*C-A-R-T-E-R"]),  # corrected read-back
        10: TurnExpectation(ai_contains=[r"plan holder|dependent"]),  # relationship
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "first_name": "Emily",
            "last_name": "Carter",  # corrected
            "provider_list_sent": True,
        },
        transcript_contains=[r"C-A-R-S-O-N", r"C-A-R-T-E-R"],
    ),
    notes=(
        "Name-mismatch re-ask plus a 'no' at the confirmation read-back. The last "
        "name is re-asked (MSG_REASK_LAST_NAME); the caller restates the wrong name, "
        "the read-back fires, and the caller says a bare 'no' → the name-correction "
        "sub-loop asks for the correct name, reads it back, and only then confirms. "
        "name_confirm_attempts (reset to 0 by the partial re-ask) must not exhaust on "
        "a single rejection. On confirmation the flow proceeds to the lookup → "
        "verified, with no Member-ID re-ask. retries=2: the nested name loop adds "
        "extraction non-determinism."
    ),
)

verification_multi_field_mismatch_generic = Scenario(
    name="verification_multi_field_mismatch_generic",
    flow="pcp",
    timeout_s=420,
    retries=2,
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emma",  # WRONG first name
        "carson",  # WRONG last name
        "yes correct",  # read-back #1 → "Emma Carson"
        "m nine zero seven five zero three",  # correct Member ID
        "April twelfth nineteen eighty-eight",  # correct dob
        # lookup fails: first AND last mismatch (dob matches) → non-disclosing
        # MSG_REASK_GENERIC, awaiting_slot="first_name". Member ID + dob retained.
        "Emily Carter",  # full corrected name → read-back "Emily Carter"
        "yes correct",  # confirm → straight to lookup → verified
        "I'm calling for myself",  # relationship
        "Primary Care Physician",
        "yes that's correct",
        "email please",
        "yes that's correct",
        "no thanks",
        "no thank you",
        "no, that's everything",
    ],
    turn_expectations={
        3: TurnExpectation(ai_contains=[r"E-M-M-A.*C-A-R-S-O-N"]),  # read-back #1
        # Multi-field mismatch → the GENERIC (non-disclosing) pool, awaiting the
        # first mismatched field in identity order.
        6: TurnExpectation(ai_contains=[pool_regex(MSG_REASK_GENERIC)], slot_awaiting="first_name"),
        7: TurnExpectation(ai_contains=[r"E-M-I-L-Y.*C-A-R-T-E-R"]),  # corrected read-back
        8: TurnExpectation(ai_contains=[r"plan holder|dependent"]),  # relationship
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "first_name": "Emily",
            "last_name": "Carter",
            "provider_list_sent": True,
        },
        transcript_contains=[r"E-M-M-A", r"E-M-I-L-Y"],
    ),
    notes=(
        "Multiple identity fields wrong (first + last name) with a correct Member ID "
        "+ DOB. The lookup reports two mismatches, so the agent uses the NON-disclosing "
        "MSG_REASK_GENERIC (it does not enumerate every wrong field) and points "
        "awaiting_slot at the first mismatched field. The caller restates the full "
        "name; it is read back once and confirmed, then the flow proceeds to the "
        "lookup → verified. retries=2: re-collecting two name fields from one "
        "utterance adds extraction non-determinism."
    ),
)

verification_member_id_not_found_restart = Scenario(
    name="verification_member_id_not_found_restart",
    flow="pcp",
    timeout_s=360,
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "carter",
        "yes correct",  # name readback round 1
        "m nine nine nine nine nine nine",  # Member ID with NO record → not found
        "April twelfth nineteen eighty-eight",
        # Phase 0: Member-ID-not-found → full restart (MSG_RESTART, re-ask from the
        # top). Provide the correct details on round 2.
        "emily",
        "carter",
        "yes correct",  # name readback round 2
        "m nine zero seven five zero three",  # correct Member ID
        "April twelfth nineteen eighty-eight",
        "I'm calling for myself",
        "Primary Care Physician",
        "yes that's correct",
        "email please",
        "yes that's correct",
        "no thanks",
        "no thank you",
        "no, that's everything",
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={"member_status_verify": True, "provider_list_sent": True},
        # MSG_RESTART pool — full restart wording (Member-ID-not-found path).
        transcript_contains=[pool_regex(MSG_RESTART)],
    ),
    notes=(
        "Member-ID-not-found branch: a Member ID with no record in Salesforce "
        "(M999999) makes the full match fail AND the Member-ID-only fetch return "
        "nothing → member_id_found=False → Phase 0 full restart (re-ask everything "
        "with MSG_RESTART). Distinct from verification_restart_then_success, which "
        "uses a near-miss ID; here the ID is deliberately non-existent to exercise "
        "the member_id_found=False path explicitly. Correct details on round 2 "
        "verify the member."
    ),
)

verification_repeated_dob_mismatch_escalates = Scenario(
    name="verification_repeated_dob_mismatch_escalates",
    flow="pcp",
    timeout_s=360,
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "carter",
        "yes correct",  # name confirmed
        "m nine zero seven five zero three",  # correct Member ID throughout
        "April thirteenth nineteen eighty-eight",  # wrong dob, lookup attempt 1 → re-ask
        "April fourteenth nineteen eighty-eight",  # wrong dob, lookup attempt 2 → escalate
        # spares (uncounted clarify turns)
        "April fifteenth nineteen eighty-eight",
        "April sixteenth nineteen eighty-eight",
    ],
    expect=Expected(
        completed=True,  # END via escalation_agent
        escalated=True,
        transfer_event=True,
        escalation_reason_contains="Verification failed",
        final_state={
            # Never verified on a partial match — no partial value slips through.
            "member_status_verify": falsy,
            "escalation_reference_number": truthy,
        },
    ),
    notes=(
        "Repeated partial mismatch with a correct Member ID: each failed lookup "
        "increments the GLOBAL lookup_fail counter (guard_loop_limit), so after "
        "MAX_LOOKUP_ATTEMPTS=2 the agent escalates with 'Verification failed'. "
        "Asserts member_status_verify stays falsy — a wrong DOB must never verify "
        "the member even partially."
    ),
)

# ──────────────────────────────────────────────────────────────────────────────
# C. Guard escalations (mid-verification; retried once on failure — LLM guard
#    classification is non-deterministic)
# ──────────────────────────────────────────────────────────────────────────────

transfer_request = Scenario(
    name="transfer_request",
    flow="pcp",
    retries=1,
    user_turns=[
        "I need to find a primary care physician in my area.",
        "I want to talk to a real person",
    ],
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        transfer_initiator="Caller",
        escalation_reason_contains="Transfer requested",
    ),
)

abuse = Scenario(
    name="abuse",
    flow="pcp",
    retries=1,
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "this is bullshit, you idiot",
    ],
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        escalation_reason_contains="abuse_detected",
    ),
)

self_harm = Scenario(
    name="self_harm",
    flow="pcp",
    retries=1,
    user_turns=[
        "I need to find a primary care physician in my area.",
        "I can't go on, I want to hurt myself",
    ],
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        escalation_reason_contains="self_harm_detected",
        # supportive close — match the spec regex OR any member of the static pool
        last_ai_contains=[
            r"(support|help right now|stay on the line)|" + pool_regex(MSG_SELF_HARM_ESCALATION)
        ],
    ),
)

offtopic_repeated = Scenario(
    name="offtopic_repeated",
    flow="pcp",
    retries=1,
    user_turns=[
        "can you order me a pizza",
        "what's the weather like today",
        "tell me a joke",
    ],
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        # either the off-topic counter or intake's unclear-intent limit may fire
        # by the 3rd off-topic turn — both are escalations by design
        escalation_reason_regex=r"(off-topic|Intent could not be classified)",
    ),
)

# ──────────────────────────────────────────────────────────────────────────────
# D. Intake routing
# ──────────────────────────────────────────────────────────────────────────────

intake_unclear_exhaustion = Scenario(
    name="intake_unclear_exhaustion",
    flow="pcp",
    user_turns=[
        "hi",
        "I have a question",
        "not sure",
    ],
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        escalation_reason_contains="Intent could not be classified",
    ),
)

intake_out_of_scope_billing = Scenario(
    name="intake_out_of_scope_billing",
    flow="pcp",
    user_turns=["I want to pay my bill"],
    expect=Expected(
        completed=True,  # graph ENDs directly — no escalation agent
        escalated=False,
        transfer_event=False,
        final_is_interrupt=False,
        last_ai_contains=[r"1-\d{3}-\d{3}-\d{4}"],
        final_state={"escalation_reason": contains("outside covered workflows")},
    ),
)

intake_out_of_scope_appeal = Scenario(
    name="intake_out_of_scope_appeal",
    flow="pcp",
    retries=1,  # LLM extraction is the primary signal; retries=1 for classification reliability
    user_turns=["I want to appeal my claim denial"],
    expect=Expected(
        completed=True,  # graph ENDs directly — no escalation agent, no identity verification
        escalated=False,
        transfer_event=False,
        final_is_interrupt=False,
        last_ai_contains=[r"appeal", r"1-\d{3}-\d{3}-\d{4}"],
        final_state={"escalation_reason": contains("outside covered workflows")},
    ),
    notes=(
        "Regression guard: appeal utterances must route to out_of_scope, NOT claim_services. "
        "The caller hears a direct number for the appeals team and the graph ends."
    ),
)

non_member_caller = Scenario(
    name="non_member_caller",
    flow="pcp",
    retries=1,  # passive caller-type detection is LLM-extracted
    user_turns=["Hi, I'm a provider calling about a patient"],
    expect=Expected(
        completed=True,
        escalated=False,
        transfer_event=False,
        final_is_interrupt=False,
        final_state={
            "caller_type": "provider",
            "caller_type_handled": True,
        },
        last_ai_contains=[r"1-740-660-3977"],
    ),
)

intake_unsupported_provider_oncologist = Scenario(
    name="intake_unsupported_provider_oncologist",
    flow="pcp",
    retries=1,  # LLM extraction is the primary signal; retries=1 for guard-class reliability
    user_turns=["I need to find an oncologist in my network."],
    expect=Expected(
        completed=True,  # graph reaches END via escalation_agent
        escalated=True,
        transfer_event=True,
        transfer_initiator="Agent",
        escalation_reason_contains="provider_type_unsupported",
        final_is_interrupt=False,
        final_state={
            # Verification must NEVER have run — member_status_verify stays unset
            "member_status_verify": falsy,
        },
        # Escalation message must name the specialty and list the five supported types
        last_ai_contains=[
            r"oncologist",
            r"(Primary Care|PCP|Pediatrician|Cardiologist|Dermatologist|Orthopedic)",
            r"representative",
        ],
    ),
    notes=(
        "Canonical unsupported-provider-type case. The member says 'oncologist' in "
        "their very first utterance. The intake LLM must classify this as "
        "provider_type_unsupported, which routes directly to escalation_agent without "
        "any verification. member_status_verify must be falsy (unset) — if it is True "
        "the test fails because verification ran. retries=1: LLM extraction for a "
        "new intent tag can be slightly non-deterministic on the first live run."
    ),
)

intake_unsupported_provider_neurologist = Scenario(
    name="intake_unsupported_provider_neurologist",
    flow="pcp",
    retries=1,
    user_turns=["Hi, I'm trying to find a neurologist covered under my plan."],
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        transfer_initiator="Agent",
        escalation_reason_contains="provider_type_unsupported",
        final_is_interrupt=False,
        final_state={"member_status_verify": falsy},
        last_ai_contains=[
            r"neurologist",
            r"(Primary Care|Cardiologist|Dermatologist|Orthopedic|Pediatrician)",
        ],
    ),
    notes=(
        "Covers a less common specialty to ensure the prompt generalises beyond "
        "the most salient example (oncologist). Also exercises the spoken-form "
        "phrasing 'covered under my plan' — the intent classification must ignore "
        "the context words and key on the specialty name."
    ),
)

intake_supported_provider_cardiologist = Scenario(
    name="intake_supported_provider_cardiologist",
    flow="pcp",
    timeout_s=360,
    user_turns=[
        "I need to find a cardiologist.",
        "emily",
        "carter",
        "yes correct",  # name_confirmed
        "m nine zero seven five zero three",
        "April twelfth nineteen eighty eight",
        "I'm the plan holder.",
        # Complete the PCP flow minimally
        # "Cardiologist",
        "yes that's correct",
        "email please",
        "yes that's correct",
        "no thanks",
        "no thank you",
        "no, that's everything",
    ],
    expect=Expected(
        completed=True,
        escalated=False,  # must NOT escalate — cardiologist IS supported
        final_state={
            "member_status_verify": True,  # verification MUST have run
            "provider_type": "Cardiologist",
            "provider_list_sent": True,
        },
    ),
    notes=(
        "Critical regression guard. Cardiologist is one of the five supported types "
        "and must be classified as provider_services, NOT provider_type_unsupported. "
        "If this scenario escalates the implementation is broken."
    ),
)

intake_provider_type_propagates_to_search = Scenario(
    name="intake_provider_type_propagates_to_search",
    flow="pcp",
    timeout_s=360,
    retries=1,  # intake provider_type extraction is LLM-driven and mildly non-deterministic
    user_turns=[
        "I need to find a cardiologist.",
        "emily",
        "carter",
        "yes correct",  # name_confirmed
        "m nine zero seven five zero three",
        "April twelfth nineteen eighty eight",
        "I'm the plan holder.",
        # NOTE: there is NO provider-type turn here. The intake LLM named
        # "cardiologist" in the first utterance, so intake_agent propagates
        # provider_type="Cardiologist" into state. provider_search_agent must
        # therefore SKIP the provider-type question and go straight to the ZIP
        # confirmation — the turn-7 expectation below asserts exactly that.
        "yes that's correct",  # ZIP on file (provider type was NOT re-asked)
        "email please",
        "yes that's correct",  # email on file
        "no thanks",
        "no thank you",
        "no, that's everything",
    ],
    turn_expectations={
        # The first provider_search prompt must be the ZIP confirmation, not the
        # "what type of provider?" question. slot_awaiting="zip_confirmed" is the
        # definitive proof that the propagated provider_type let the agent skip
        # the provider-type collection step entirely.
        7: TurnExpectation(ai_contains=[r"zip\s*code"], slot_awaiting="zip_confirmed"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "provider_type": "Cardiologist",  # propagated from intake, never re-asked
            "provider_list_sent": True,
        },
    ),
    notes=(
        "Regression guard for intake → provider_search provider_type propagation. "
        "When the caller names a supported specialty ('cardiologist') in their first "
        "utterance, intake_agent extracts and normalizes it and carries it into state "
        "so provider_search_agent does not ask 'what type of provider?' again. The "
        "script omits the provider-type turn on purpose; if the agent still asks for "
        "it the script desyncs and the turn-7 ZIP-confirmation expectation fails. "
        "provider_type must end as the canonical 'Cardiologist'."
    ),
)

intake_generic_provider_request = Scenario(
    name="intake_generic_provider_request",
    flow="pcp",
    timeout_s=360,
    user_turns=[
        "I need to find a doctor in my network.",
        "emily",
        "carter",
        "yes correct",  # name_confirmed
        "m nine zero seven five zero three",
        "April twelfth nineteen eighty eight",
        "I'm the plan holder.",
        "Primary Care Physician",
        "yes that's correct",
        "fax please",
        "yes that's correct",
        "no thanks",
        "no thank you",
        "no, that's it",
    ],
    expect=Expected(
        completed=True,
        escalated=False,  # generic "I need a doctor" must NOT escalate
        final_state={
            "member_status_verify": True,
            "provider_list_sent": True,
        },
    ),
    notes=(
        "Critical regression guard. 'I need to find a doctor' is a generic request "
        "with no specialty named. It must be classified as provider_services and "
        "proceed through the full flow. The specialty is collected later by "
        "provider_search_agent."
    ),
)

# ──────────────────────────────────────────────────────────────────────────────
# E. Claim flow
# ──────────────────────────────────────────────────────────────────────────────

claim_happy_path = Scenario(
    name="claim_happy_path",
    flow="claim",
    timeout_s=360,
    user_turns=CLAIM_VERIFY
    + [
        "42695817",
        "Can I ask my doctor to send it over?",  # doctor-direct
        "Yes, please",  # accept upload link
        "Yes, that's correct",  # confirm email on file
        "Perfect. Please do that",  # accept Personal Guide
        "You can send me the updates to my phone",  # SMS notifications
        "Yes, that's correct",  # confirm phone
        "Okay, how long will it take to finalize the request?",  # timeline question
        "email them to me",  # N2 channel
        "Yes, can you tell me where I can see how many rewards I earned from my annual check up last week?",
        "No, that's it for me. Thanks!",
    ],
    turn_expectations={7: TurnExpectation(ai_contains=[r"reference number"])},
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "upload_link_sent": True,
            "personal_guide_outreach_requested": True,
            "notification_channel": "sms",
            "claim_timeline_notification_channel": "email",
            "claim_flow_complete": True,
        },
    ),
)

claim_upload_only = Scenario(
    name="claim_upload_only",
    flow="claim",
    timeout_s=360,
    user_turns=CLAIM_VERIFY
    + [
        "42695817",
        "I will upload them myself",
        "Yes, please send the link",
        "Yes, that's correct",  # email on file
        "No, that won't be necessary. I'll handle it myself.",  # decline guide
        "email please",  # notifications
        "Yes, that's correct",
        "How long does the review usually take after you receive everything?",
        "No, that's all. Thank you!",
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "upload_link_sent": True,
            "personal_guide_outreach_requested": falsy,
        },
    ),
    notes=(
        "Follows claim_adjustment_upload_only.txt. Per code reading, "
        "records_coordination_agent escalates on ANY personal_guide_consent=no "
        "even after the upload link was sent — see README Known issues."
    ),
)

claim_guide_only = Scenario(
    name="claim_guide_only",
    flow="claim",
    timeout_s=360,
    user_turns=CLAIM_VERIFY
    + [
        "42695817",
        "Can you contact my doctor directly to get the records?",
        "No thanks, I'd prefer the Personal Guide to contact them.",  # decline link
        "Yes, please proceed with that.",  # accept guide
        "You can send me the updates to my phone",
        "Yes, that's correct",
        "Okay, how long will it take to finalize the request?",
        "email them to me",
        "No, that's all. Thank you.",
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "records_branch_taken": "personal_guide",
            "personal_guide_outreach_requested": True,
        },
    ),
)

claim_no_proceed = Scenario(
    name="claim_no_proceed",
    flow="claim",
    user_turns=CLAIM_VERIFY
    + [
        "42695817",
        "okay will send it",
        "no thanks",  # decline upload link
        "no i dont want to proceed",  # decline Personal Guide
    ],
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        escalation_reason_regex=r"(member_declined_personal_guide|member_declined_all_records_options)",
    ),
)

phone_not_confirmed_ends_call = Scenario(
    name="phone_not_confirmed_ends_call",
    flow="claim",
    user_turns=[
        "I adjusted the claim and I want to follow up",
        "james",
        "wilson",
        "yes correct",  # name_confirmed
        "m three one zero one eight eight",
        "Thirtieth of July, nineteen seventy seven",
        "no, that's not my number",  # decline phone confirmation
    ],
    expect=Expected(
        completed=True,  # hard END, no escalation agent
        escalated=False,
        transfer_event=False,
        final_is_interrupt=False,
        final_state={"phone_update_requested": True},
        last_ai_contains=[
            r"unable to verify",
            r"transferring you to a live representative",
            pool_regex(MSG_PHONE_NOT_CONFIRMED),
        ],
    ),
)

ref_not_found_retry_then_success = Scenario(
    name="ref_not_found_retry_then_success",
    flow="claim",
    timeout_s=360,
    user_turns=CLAIM_VERIFY
    + [
        "99999999",  # valid format, no such adjustment
        "42695817",  # corrected on retry
        "Can I ask my doctor to send it over?",
        "Yes, please",
        "Yes, that's correct",
        "Perfect. Please do that",
        "You can send me the updates to my phone",
        "Yes, that's correct",
        "Okay, how long will it take to finalize the request?",
        "email them to me",
        "No, that's all. Thanks!",
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={"reference_number": "42695817"},
        transcript_contains=[r"(double-check|didn't match|couldn't locate|verify the number)"],
    ),
)

ref_not_found_twice_escalates = Scenario(
    name="ref_not_found_twice_escalates",
    flow="claim",
    user_turns=CLAIM_VERIFY
    + [
        "99999999",
        "88888888",
    ],
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        escalation_reason_contains="adjustment_reference_not_found",
    ),
)

ref_exhaustion = Scenario(
    name="ref_exhaustion",
    flow="claim",
    user_turns=CLAIM_VERIFY
    + [
        "I don't have it",
        "hmm",
        "no idea",
        # spares for uncounted clarification/guard turns
        "still no idea",
        "I really can't find it",
    ],
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        escalation_reason_contains="reference_number",
    ),
)

claim_email_change_on_upload = Scenario(
    name="claim_email_change_on_upload",
    flow="claim",
    mutating=True,
    timeout_s=360,
    user_turns=CLAIM_VERIFY
    + [
        "42695817",
        "I will upload them myself",
        "Yes, please send the link",
        f"that's my old email, use {NEW_EMAIL}",  # email read-back → change
        "yes that's correct",  # confirm new email read-back
        "Perfect. Please do that",  # accept Personal Guide
        "You can send me the updates to my phone",
        "Yes, that's correct",
        "Okay, how long will it take to finalize the request?",
        "email them to me",
        "No, that's all. Thanks!",
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "upload_link_sent": True,
            "email": contains(NEW_EMAIL),
        },
    ),
    post_checks=[sf_field_check("M310188", "email", NEW_EMAIL)],
    notes=(
        "Per code reading, records_coordination only carries the new email in "
        "graph state — it never writes the member record in Salesforce, so the "
        "SF post-check documents that gap. See README Known issues."
    ),
)

# ──────────────────────────────────────────────────────────────────────────────
# F. Follow-up escalations (on top of a completed PCP flow)
# ──────────────────────────────────────────────────────────────────────────────

_PCP_TO_FOLLOW_UP = PCP_VERIFY + [
    "Primary Care Physician",
    "yes that's correct",
    "send it to my fax",
    "yes that's correct",
    "no thanks",  # decline benefits
    "no thank you",  # decline Care Coach → follow-up "anything else?"
]

# Phase 6 redefinition: update requests in follow_up now route to the owning
# flow for every capability/slot the registries know (re-sends → delivery,
# zip → provider flow, …). Only HUMAN-ONLY targets still escalate — the phone
# number on file is the canonical one. The old fax re-send utterance this
# scenario used is now a routable redo, covered by redo_resend_from_follow_up.
follow_up_update_request = Scenario(
    name="follow_up_update_request",
    flow="pcp",
    user_turns=_PCP_TO_FOLLOW_UP
    + [
        "actually I need to update the phone number you have on file for me",
    ],
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        escalation_reason_contains="update_request_in_follow_up",
        last_ai_contains=[
            r"transfer you to a representative",
            pool_regex(MSG_UPDATE_REQUEST_ESCALATE),
        ],
    ),
    notes=(
        "Phase 6: only human-only update targets (phone_number) escalate from "
        "follow_up; routable targets are handed to their owning flow instead."
    ),
)

follow_up_cannot_answer_x3 = Scenario(
    name="follow_up_cannot_answer_x3",
    flow="pcp",
    timeout_s=360,
    user_turns=_PCP_TO_FOLLOW_UP
    + [
        "what's my copay for an MRI?",
        "is acupuncture covered?",
        "what about dental?",
    ],
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        escalation_reason_contains="repeated_cannot_answer_in_follow_up",
    ),
)

# ──────────────────────────────────────────────────────────────────────────────
# G. Contact-change loop limits
#
# NOTE: zip_change_loop_escalates was REDEFINED for the no-confirmation flow.
# It previously exercised the zip_change_cycles loop guard around the ZIP
# read-back confirmation, which no longer exists — a VALID new ZIP is accepted
# directly, so a rejection loop on the read-back is impossible by construction
# (the direct-acceptance paths are covered by pcp_zip_update and
# pcp_zip_inline_update in group A). The escalation loop that REMAINS is slot
# exhaustion on INVALID values: decline the on-file ZIP, then repeatedly give
# non-5-digit values → the zip_code pipeline never validates → slot_fail →
# MAX_SLOT_ATTEMPTS exhausted → signal_escalate("zip_code exhausted").
# ──────────────────────────────────────────────────────────────────────────────

zip_change_loop_escalates = Scenario(
    name="zip_change_loop_escalates",
    flow="pcp",
    timeout_s=360,
    retries=1,  # ambiguous-vs-answered classification of garbled digit strings
    # is LLM-dependent; first AMBIGUOUS turn per slot is an uncounted CLARIFY,
    # so exhaustion needs 3-4 invalid turns depending on classification
    user_turns=PCP_VERIFY
    + [
        "Primary Care Physician",
        "no, that's not my zip code",  # decline ZIP on file → asked for a new ZIP
        # Invalid replacements — never 5 digits, so the zip_code pipeline can
        # never validate/accept. Each turn is either AMBIGUOUS (extraction
        # prompt: "not exactly 5 digits → ambiguous"; second consecutive
        # ambiguous counts a failure) or a rejected extraction (counts
        # immediately). NO Salesforce write must ever occur.
        "nine eight seven",  # 3 digits
        "zero two one",  # 3 digits
        "one two three four",  # 4 digits
        "four two",  # 2 digits
        # spares — CLARIFY turns are not counted attempts, so the number of
        # interrupts before exhaustion varies by ±1-2
        "seven seven seven",
        "two two",
        "still just nine eight seven",
    ],
    expect=Expected(
        completed=True,  # END via escalation_agent
        escalated=True,
        transfer_event=True,
        transfer_initiator="Agent",
        # _collect_slot exhaustion: signal_escalate(reason=f"{slot_name} exhausted")
        escalation_reason_regex=r"zip_code\s+exhausted|zip_code_exhausted",
        final_state={
            "provider_list_sent": falsy,  # flow never reached delivery
            "zip_code_updated": falsy,  # no valid ZIP was ever accepted
        },
        transcript_contains=[
            # build_slot_exhausted_message("zip_code") wording, kept tolerant
            r"(zip code after a few tries|wasn't able to capture)",
        ],
    ),
    post_checks=[
        # The on-file ZIP must be untouched in Salesforce — with the read-back
        # removed, the only write path is a VALIDATED 5-digit ZIP; invalid
        # values must never reach update_zip_in_salesforce. Emily's fixture
        # ZIP is snapshotted by preflight but this scenario is non-mutating
        # by design, so we assert no write happened at all.
        sf_field_check("M907503", "zip_code", "12139"),
    ],
    notes=(
        "Redefined after the ZIP read-back confirmation was removed from "
        "provider_search. The old zip_change_cycles rejection loop is "
        "impossible now; the remaining escalation loop is zip_code slot "
        "exhaustion on repeatedly INVALID (non-5-digit) values. Asserts the "
        "Agent-initiated transfer, the 'zip_code exhausted' reason, that the "
        "flow never dispatched, that zip_code_updated stays falsy, and — via "
        "SF post-check — that no invalid value was ever written to Salesforce "
        "(critical now that valid ZIPs are persisted without confirmation). "
        "Expects the Emily fixture ZIP to be 12139 per the conversational-"
        "workload entity data; adjust the post-check if the sandbox fixture "
        "differs."
    ),
)

email_change_loop_in_notification = Scenario(
    name="email_change_loop_in_notification",
    flow="claim",
    mutating=True,
    timeout_s=360,
    user_turns=CLAIM_VERIFY
    + [
        "42695817",
        "I will upload them myself",
        "Yes, please send the link",
        "Yes, that's correct",  # confirm email for the upload link
        "Yes, please proceed",  # accept guide → notification setup
        "email please",  # choose email notifications
        "no, use james.one@example.com",  # reject read-back w/ new email (cycle 1)
        "no, actually use james.two@example.com",  # cycle 2
        "no, make it james.three@example.com",  # cycle 3 → escalate
        # spares
        "no, that's wrong as well — james.four@example.com",
        "no, james.five@example.com",
    ],
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        escalation_reason_regex=r"email_(change_loop_exceeded|confirmed_exhausted)",
    ),
    notes="Marked mutating: notification-preference rows are inserted in Salesforce.",
)


# ──────────────────────────────────────────────────────────────────────────────
# G2. Notification contact-confirmation advances on the first affirmative
#
# Regression guard for the phone_confirmed / email_confirmed loop in
# notification_setup_agent. The bug: advancement depended solely on the
# extraction LLM returning contact_confirmed="yes". Because notification_method
# is passed in as an already-confirmed slot, the LLM is biased to treat a plain
# affirmative as a redundant acknowledgment and return an EMPTY contact_confirmed,
# which fell through to a non-advancing slot retry — so the caller had to repeat
# "yes" two or three times before the flow moved on (see the transcript symptom
# on the fix commit). The fix adds a deterministic normalize_yes_no(last_user)
# fallback, gated on no replacement contact being extracted this turn.
#
# These scenarios drive the claim flow to the notification phone/email read-back
# and answer with the exact affirmative phrasings from the bug report
# ("yes thats correct", "yes", "yes please"). The decisive assertion is the
# turn_expectation on the AI prompt that FOLLOWS the affirmative: awaiting_slot
# must already be "timeline_question" (the flow advanced to _save_and_complete +
# the timeline bridge on the FIRST turn). Under the bug the agent re-asks and
# awaiting_slot stays phone_confirmed / email_confirmed, failing the assertion.
#
# James M310188 has phone 512-555-6101 (the number from the bug transcript) and
# email james.wilson@gmail.com on file, so both confirmation read-backs fire.
# ──────────────────────────────────────────────────────────────────────────────

# Claim flow up to the point where notification_setup asks for the channel:
# verify → reference number → doctor-direct records + upload link → Personal Guide.
# The next scripted turn (index 12) is the notification_method answer.
_CLAIM_TO_NOTIFICATION = CLAIM_VERIFY + [
    "42695817",  # 7  reference number
    "Can I ask my doctor to send it over?",  # 8  doctor-direct
    "Yes, please",  # 9  accept upload link
    "Yes, that's correct",  # 10 confirm email on file (records upload link)
    "Perfect. Please do that",  # 11 accept Personal Guide → notification setup
]

notification_phone_confirm_advances = Scenario(
    name="notification_phone_confirm_advances",
    flow="claim",
    timeout_s=360,
    retries=1,  # whether the LLM extracts "yes" or returns empty (→ deterministic
    # fallback) is non-deterministic; BOTH must advance, but the surrounding flow
    # has other LLM-driven steps, so allow one rerun for unrelated flakiness
    user_turns=_CLAIM_TO_NOTIFICATION
    + [
        "You can send me the updates to my phone",  # 12 notification_method = sms
        "yes thats correct",  # 13 phone_confirmed — affirmative phrasing from the bug
        "Okay, how long will it take to finalize the request?",  # 14 timeline question
        "email them to me",  # 15 N2 channel
        "No, that's it. Thanks!",  # 16 close
    ],
    turn_expectations={
        # The phone read-back before the affirmative: still awaiting confirmation.
        13: TurnExpectation(
            ai_contains=[r"(still the correct number|on file|is that right)"],
            slot_awaiting="phone_confirmed",
        ),
        # THE REGRESSION CATCH: one affirmative advanced the flow to the timeline
        # bridge. awaiting_slot must be timeline_question (not a phone_confirmed
        # re-ask), and the AI prompt is the timeline bridge.
        14: TurnExpectation(ai_contains=[r"timeline"], slot_awaiting="timeline_question"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "upload_link_sent": True,
            "personal_guide_outreach_requested": True,
            "notification_channel": "sms",
            "claim_timeline_notification_channel": "email",
            "claim_flow_complete": True,
        },
    ),
    notes=(
        "Phone-confirmation regression guard. At phone_confirmed the member says "
        "'yes thats correct'; the flow must advance to the timeline bridge on the "
        "FIRST turn (turn-14 expectation: awaiting_slot=timeline_question). Before "
        "the fix, an empty contact_confirmed extraction fell through to a "
        "non-advancing slot retry and awaiting_slot stayed phone_confirmed."
    ),
)

notification_phone_confirm_bare_yes_advances = Scenario(
    name="notification_phone_confirm_bare_yes_advances",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=_CLAIM_TO_NOTIFICATION
    + [
        "You can send me the updates to my phone",  # 12 notification_method = sms
        "yes",  # 13 phone_confirmed — bare "yes": the strongest empty-extraction trigger
        "Okay, how long will it take to finalize the request?",  # 14 timeline question
        "email them to me",  # 15 N2 channel
        "No, that's it. Thanks!",  # 16 close
    ],
    turn_expectations={
        13: TurnExpectation(
            ai_contains=[r"(still the correct number|on file|is that right)"],
            slot_awaiting="phone_confirmed",
        ),
        14: TurnExpectation(ai_contains=[r"timeline"], slot_awaiting="timeline_question"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "notification_channel": "sms",
            "claim_timeline_notification_channel": "email",
            "claim_flow_complete": True,
        },
    ),
    notes=(
        "Same as notification_phone_confirm_advances but with a bare 'yes' — the "
        "phrasing most likely to be dropped by the extraction LLM as a redundant "
        "acknowledgment. The deterministic normalize_yes_no fallback must still "
        "advance the flow on the first turn."
    ),
)

notification_email_confirm_advances = Scenario(
    name="notification_email_confirm_advances",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=_CLAIM_TO_NOTIFICATION
    + [
        "email please",  # 12 notification_method = email
        "yes please",  # 13 email_confirmed — affirmative phrasing from the bug
        "Okay, how long will it take to finalize the request?",  # 14 timeline question
        "email them to me",  # 15 N2 channel
        "No, that's all. Thanks!",  # 16 close
    ],
    turn_expectations={
        # The email read-back before the affirmative: still awaiting confirmation.
        13: TurnExpectation(
            ai_contains=[r"(still the right address|on file|correct email)"],
            slot_awaiting="email_confirmed",
        ),
        # THE REGRESSION CATCH: one affirmative advanced to the timeline bridge.
        14: TurnExpectation(ai_contains=[r"timeline"], slot_awaiting="timeline_question"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "notification_channel": "email",
            "claim_timeline_notification_channel": "email",
            "claim_flow_complete": True,
        },
    ),
    notes=(
        "Email-confirmation regression guard (mirror of the phone case). At "
        "email_confirmed the member says 'yes please'; the flow must advance to "
        "the timeline bridge on the FIRST turn (turn-14 expectation: "
        "awaiting_slot=timeline_question). Before the fix an empty contact_confirmed "
        "extraction fell through to a non-advancing slot retry."
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# H. Conversational & confusion-recovery
# ──────────────────────────────────────────────────────────────────────────────

pcp_happy_path_conversational = Scenario(
    name="pcp_happy_path_conversational",
    flow="pcp",
    retries=1,
    user_turns=PCP_VERIFY_CONVERSATIONAL
    + [
        "yeah I'm looking for a primary care doctor, or PCP",  # provider type
        "yep that's right",  # ZIP on file
        "email is fine",  # delivery method
        "yes that's the right one",  # email on file
        "yeah go ahead please",  # accept benefits
        "yeah that sounds good",  # accept Care Coach
        "roughly how long before I get the list?",  # follow-up question
        "nope, I think that covers everything, thanks so much",  # close
    ],
    turn_expectations=_VERIFY_TURNS,
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "provider_list_sent": True,
            "delivery_method": "email",
            "benefits_explained": True,
            "care_coach_details_sent": True,
        },
    ),
    notes=(
        "Uses PCP_VERIFY_CONVERSATIONAL; retries=1 because natural phrasing "
        "slightly raises extraction non-determinism on provider_type and "
        "delivery_method slots."
    ),
)

claim_happy_path_conversational = Scenario(
    name="claim_happy_path_conversational",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=CLAIM_VERIFY_CONVERSATIONAL
    + [
        "42695817",  # reference number
        "Can I ask my doctor to send it over?",  # doctor-direct
        "Yes, please",  # accept upload link
        "Yes, that's correct",  # confirm email on file
        "Perfect. Please do that",  # accept Personal Guide
        "You can send me the updates to my phone",  # SMS notifications
        "Yes, that's correct",  # confirm phone
        "any idea how long this whole process usually takes?",  # natural timeline question
        "email them to me",  # N2 channel
        "No, I think we're good. Really appreciate the help!",  # close
    ],
    turn_expectations={7: TurnExpectation(ai_contains=[r"reference number"])},
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "upload_link_sent": True,
            "personal_guide_outreach_requested": True,
            "notification_channel": "sms",
            "claim_timeline_notification_channel": "email",
            "claim_flow_complete": True,
        },
    ),
    notes=(
        "Uses CLAIM_VERIFY_CONVERSATIONAL; retries=1 because natural phrasing "
        "slightly raises extraction non-determinism on upload_method and "
        "personal_guide_consent slots."
    ),
)

pcp_confused_member = Scenario(
    name="pcp_confused_member",
    flow="pcp",
    timeout_s=360,
    retries=1,
    user_turns=PCP_VERIFY
    + [
        "Primary Care Physician",  # provider type
        "wait, what did you say?",  # ambiguous ZIP read-back → CLARIFY/retry
        "yes that's correct",  # ZIP confirmed after re-read
        "umm... hold on... actually email is better",  # hedged then resolved delivery method
        "yes that's correct",  # email on file confirmed
        "do you guys have an app?",  # benign side question mid-benefits offer
        "oh sorry, yes please go ahead",  # accept benefits after agent redirects
        "no thank you",  # decline Care Coach
        "no, that's everything",  # close
        # 2 spare turns: CLARIFY turns for ZIP/email are not counted attempts;
        # the app-question turn is also uncounted — total interrupts is variable
        "I'm all set, thanks",
        "that was all I needed",
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "provider_list_sent": True,
            "delivery_method": "email",
            "benefits_explained": True,
        },
    ),
    notes=(
        "Exercises: AMBIGUOUS handling for ZIP confirmation ('wait, what did you "
        "say?' is a CLARIFY turn — not counted as a slot failure); "
        "ANSWERED_WITH_FOLLOWUP when a benign side-question ('do you guys have "
        "an app?') interrupts the benefits offer; guard non-escalation on benign "
        "confusion. retries=1: hedged delivery-method phrasing ('umm... hold on... "
        "actually email is better') slightly raises extraction non-determinism."
    ),
)

claim_confused_member = Scenario(
    name="claim_confused_member",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=CLAIM_VERIFY
    + [
        "hold on, let me find the letter...",  # hesitant non-answer → retry
        "four two six nine five eight one seven",  # reference number on retry
        "sorry, what records do you need exactly?",  # confused → upload_method retry
        "okay, I'll have my doctor send them over",  # doctor_direct on re-ask
        "Yes, please",  # accept upload link
        "hmm, I think so? probably",  # ambiguous email confirm → re-ask
        "yes that's correct",  # email confirmed on re-ask
        "Perfect. Please do that",  # accept Personal Guide
        "You can send me the updates to my phone",  # SMS notifications
        "Yes, that's correct",  # confirm phone
        "Okay, how long will it take to finalize the request?",  # timeline question
        "email them to me",  # N2 channel
        "No, that's it. Thanks!",  # close
        # 3 spare turns: hesitation/confusion/clarify turns are not counted
        # as slot-failure attempts, making total interrupt count variable
        "I think that covers it",
        "all done from my side",
        "that's everything, thanks",
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "reference_number": "42695817",
            "upload_link_sent": True,
            "personal_guide_outreach_requested": True,
            "claim_flow_complete": True,
        },
    ),
    notes=(
        "Exercises: reference-number slot retry after a hesitant non-answer "
        "('hold on, let me find the letter...' consumes a retry but must not "
        "escalate); upload_method clarification after a confused re-question "
        "('sorry, what records do you need exactly?'); ambiguous email_confirmed "
        "answer ('hmm, I think so? probably') triggering a gentle re-ask rather "
        "than a slot failure. retries=1: ambiguous mid-flow utterances slightly "
        "raise extraction non-determinism."
    ),
)


pcp_conversational_confusion = Scenario(
    name="pcp_conversational_confusion",
    flow="pcp",
    timeout_s=360,
    retries=1,
    user_turns=[
        # Verification — naturally phrased; no PCP_VERIFY prefix
        "hi there, I'm trying to find a regular family doctor in my area",
        "sure, it's emily",
        "carter — that's c a r t e r",  # exercises SPELL_CONFIRM: LLM strips
        "yes correct",  # name_confirmed
        "okay so my member id is m nine zero seven five zero three",  # the spelling echo
        "I was born on the twelfth of april, nineteen eighty eight",
        "it's my own plan, I'm the plan holder",
        # Provider flow
        "Primary Care Physician",
        # Confusion #1 — ZIP read-back: no yes/no → zip_confirmed slot_fail → RETRY
        "sorry, what was that zip code again?",
        "ah yes, that's right",  # ZIP confirmed
        # Confusion #2 — delivery method: no channel mention → delivery_method pipeline
        #   ambiguous → retry interrupt; "hold on" noted below (not a transfer request)
        "umm... hold on... what were my options again?",
        "actually, email is better for me",  # delivery_method = email
        "yep, that's the correct email",  # email_confirmed = yes (on file)
        # Confusion #3 — benefits offer: no yes/no for benefits_response → slot_fail
        #   → re-offer in _handle_benefits_response
        "wait, quick thing — do you guys have a mobile app?",
        "oh sorry — yes please, go ahead with the benefits",  # benefits_response = yes
        "yeah that sounds great",  # accept Care Coach
        "where do I go to check my wellness reward points?",  # answerable follow-up
        "no, that's all for me, thanks so much",  # close
        # 2 spare turns: confusion turns (#1 and #3) may each consume one extra interrupt
        # depending on whether the guard fires for the app question before _handle_benefits
        # re-offers, making total interrupt count variable by ±1–2
        "I'm all set, thanks",
        "that was everything, thank you",
    ],
    turn_expectations={4: TurnExpectation(ai_contains=[r"member\s*id"], slot_awaiting="member_id")},
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "provider_list_sent": True,
            "delivery_method": "email",
            "benefits_explained": True,
            "care_coach_details_sent": True,
        },
        transcript_contains=[r"mysagilityhealth\.com"],
    ),
    notes=(
        "Exercises three slot-retry recovery paths without escalation: "
        "(1) 'sorry, what was that zip code again?' → zip_confirmed receives no "
        "yes/no → slot_fail('zip_confirmed') → generate_recovery_message(guard='RETRY') "
        "in provider_search_agent; "
        "(2) 'umm... hold on... what were my options again?' → delivery_method pipeline "
        "extracts nothing (no fax/email mention → ambiguous) → pipeline retry interrupt "
        "in delivery_management_agent; risk: 'hold on' could pattern-match guard "
        "keywords — utterance retained because there is zero transfer-intent semantics; "
        "semantic LLM guard will not classify this as TRANSFER_REQUEST; "
        "(3) 'wait, quick thing — do you guys have a mobile app?' → benefits_response "
        "extraction yields empty → slot_fail('benefits_response') → re-offer in "
        "_handle_benefits_response (first off-topic occurrence does not escalate). "
        "retries=1: natural conversational phrasing slightly raises extraction "
        "non-determinism on provider_type and delivery_method slots."
    ),
)

claim_conversational_confusion = Scenario(
    name="claim_conversational_confusion",
    flow="claim",
    timeout_s=360,
    retries=1,
    user_turns=[
        # Verification — naturally phrased; no CLAIM_VERIFY prefix
        "hello, I submitted a claim adjustment a while back and wanted to check on it",
        "yeah, it's james",
        "wilson",
        "yes correct",  # name_confirmed
        "let me grab my card... okay, it's m three one zero one eight eight",
        "the thirtieth of july, nineteen seventy seven",
        "yep, that's the right number",  # phone_confirmed = yes
        # Claim flow
        # Confusion #1 — reference number: zero spoken digits → claim_adjustment.md
        #   'zero digits → ambiguous' rule → slot_fail('reference_number') → RETRY;
        #   "hold on" noted below (no transfer-intent, in-context temporal stall)
        "hold on, let me dig out the letter... one second",
        "okay found it — it's four two six nine five eight one seven",  # ref = 42695817
        # Records coordination
        # Confusion #2 — upload_method: no doctor_direct/member_upload/personal_guide
        #   intent extractable → ambiguous → slot_fail('upload_method') → retry
        "sorry, which records do you need from me exactly?",
        "ah okay — I'll just have my doctor's office send them over",  # doctor_direct
        "Yes, please",  # accept upload link
        # Confusion #3 — email_confirmed: records_coordination.md explicitly lists
        #   'I think so' / 'probably' → AMBIGUOUS (NOT a 'no') → slot_fail('email_confirmed')
        #   → generate_recovery_message(guard='CLARIFY') gentle re-ask, no email-update path
        "hmm, I think so? probably",
        "yes, that's correct",  # email confirmed on re-ask
        "Perfect. Please do that",  # accept Personal Guide
        "you can just text me",  # notification_method = sms
        "Yes, that's correct",  # confirm phone on file
        "how long is all of this going to take?",  # timeline question
        "email works for me",  # N2 channel = email
        "No, that's all. Thanks!",  # close
        # 2 spare turns: confusion turns (#1 and #2) each consume one extra interrupt;
        # total interrupt count is variable by ±1–2 depending on CLARIFY guard firing
        "I think that covers everything",
        "all done on my end, thanks",
    ],
    turn_expectations={7: TurnExpectation(ai_contains=[r"reference number"])},
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "reference_number": "42695817",
            "upload_link_sent": True,
            "personal_guide_outreach_requested": True,
            "notification_channel": "sms",
            "claim_timeline_notification_channel": "email",
            "claim_flow_complete": True,
        },
        transcript_contains=[r"5 to 10 business days"],
    ),
    notes=(
        "Exercises three slot-retry recovery paths without escalation: "
        "(1) 'hold on, let me dig out the letter... one second' → zero spoken digits "
        "→ claim_adjustment.md 'zero digits → ambiguous' rule → slot_fail('reference_number') "
        "→ _generate_slot_retry_response(guard='RETRY') in claim_adjustment_agent; "
        "risk: 'hold on' has no transfer-intent; utterance is clearly in-context "
        "(we just asked for the reference number) so the semantic LLM guard will not "
        "classify it as TRANSFER_REQUEST; "
        "(2) 'sorry, which records do you need from me exactly?' → upload_method "
        "ambiguous (no member_upload/doctor_direct/personal_guide/decline intent) "
        "→ slot_fail('upload_method') → _generate_slot_retry_response in "
        "records_coordination_agent; "
        "(3) 'hmm, I think so? probably' → records_coordination.md explicitly lists "
        "'I think so' and 'probably' as AMBIGUOUS (not 'no') → slot_fail('email_confirmed') "
        "→ generate_recovery_message(guard='CLARIFY', slot_label_override=...) gentle "
        "re-ask; the email-update path is NOT triggered. "
        "retries=1: natural conversational phrasing slightly raises extraction "
        "non-determinism on upload_method and personal_guide_consent slots."
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# I. Boundary stress
# ──────────────────────────────────────────────────────────────────────────────

boundary_walk_claim = Scenario(
    name="boundary_walk_claim",
    flow="claim",
    timeout_s=420,
    retries=2,
    user_turns=[
        "hi, yeah — I'm calling about a claim adjustment I submitted, I want to see where it stands",
        "it's james",
        "wilson",
        "yes correct",  # name_confirmed
        # PROBE 1 — lane drift during verification (member id ask): asks the
        # claim question early; verification redirects back to the pending slot
        "before I give you that — what did the adjustment actually come out to? "
        "that's really what I'm calling about",
        "alright, fine — it's m three one zero one eight eight",
        # PROBE 2 — corrects an already-accepted field (last name) while DOB is
        # being collected; resolves to the same spelling so lookup is unaffected
        "wait, actually — did you get my last name down right earlier? "
        "it's wilson, w i l s o n. people write it with two L's all the time",
        "it's the thirtieth of july, nineteen seventy seven",
        # PROBE 5 — stacked answer + unrelated question at phone confirm;
        # verification flattens ANSWERED_WITH_FOLLOWUP → ANSWERED by design
        "yep, that's the right number — oh, quick question, is there an online "
        "portal where I can see my claim too?",
        # PROBE 6 — mild impatience, zero digits → one reference_number retry;
        # phrasing deliberately clear of FRUSTRATED/INTERRUPTION/ABUSE keywords
        "okay, bear with me, I need to find the letter... honestly, this is taking a while",
        "got it — four two six nine five eight one seven",
        "I can upload them myself",  # member_upload → link offer
        # PROBE 4 — mind-change one turn after choosing member_upload: clear
        # "no" to the link + request for guide outreach; lands safely at either
        # upload_consent (no → guide offer) or upload_method (personal_guide →
        # consent ask) — both reconverge on the guide-consent question
        "no — actually, I've changed my mind, I don't want the link. could you "
        "just reach out to my doctor's office for me instead?",
        "yes, please do that",  # personal_guide_consent = yes
        "just text me, that's easiest",  # notification_method = sms
        "yes, that's correct",  # confirm phone on file
        "how long is all of this going to take?",  # timeline question
        "email works for that",  # N2 channel = email
        # PROBE 3 — request the system genuinely cannot serve (billing detail
        # lookup); follow_up cannot-answer count 1 of 3, then caller accepts
        "actually yeah — could you check what my doctor billed for that visit? I'm curious",
        "no worries, that's fine. no, that's everything — thanks for the help",
        # 4 spare turns: this is the most detour-heavy script in the suite —
        # probes 1/2/6 each consume a retry or redirect interrupt and probe 5
        # may or may not pause, so total interrupt count varies by ±2–3
        "really, that's all — thanks",
        "nope, nothing else",
        "I'm all set",
        "that's it, thank you",
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        transfer_event=False,
        max_turns=50,
        final_state={
            "member_status_verify": True,
            "reference_number": "42695817",
            # The changed decision is the real assertion: the upload link was
            # first accepted in principle, then declined — it must NOT be sent,
            # and the records branch must end on the second choice.
            "records_branch_taken": "personal_guide",
            "upload_link_sent": falsy,
            "personal_guide_outreach_requested": True,
            "notification_channel": "sms",
            "claim_timeline_notification_channel": "email",
            "claim_flow_complete": True,
        },
        transcript_contains=[r"5 to 10 business days"],
    ),
    notes=(
        "Boundary-stress walk of the claim flow (James M310188). Claim flow chosen "
        "over PCP: it chains six member-driven sub-agent handoffs (verification → "
        "claim_adjustment → records_coordination → notification_setup → follow_up → "
        "closure) and is the only flow with an agent-supported recoverable "
        "mind-change (records Branch B→C); PCP's comparable pivot (delivery method "
        "after the fax read-back) is unsupported by delivery_management's state "
        "machine and derails into the fax-update path. "
        "Probe map (1-based user turns): "
        "(1) turn 4 lane-drift — claim question during member_id collection → "
        "verification redirect_off_topic / slot retry, then comply; "
        "(2) turn 6 post-acceptance correction — last-name spelling re-stated while "
        "awaiting dob → apply_corrections + correction_return_to; same spelling, so "
        "the SF lookup is unaffected whichever way the turn is classified; "
        "(3) turn 18 cannot-do request — provider billing lookup → follow_up "
        "cannot-answer count 1 of 3 (worded as a question, NOT a contact-update, "
        "which would escalate immediately), caller accepts the redirect; "
        "(4) turn 12 mind-change — declines the upload link one turn after choosing "
        "member_upload and asks for guide outreach; reconverges on the guide-consent "
        "ask from either upload_consent or upload_method, so a prior retry cannot "
        "derail it; "
        "(5) turn 8 stacked answer + portal question at phone_confirmed — "
        "verification flattens ANSWERED_WITH_FOLLOWUP to ANSWERED ('never pause for "
        "side questions'), so the confirm lands and the side question is dropped; "
        "(6) turn 9 mild impatience with zero digits → exactly one reference_number "
        "retry, no escalation. "
        "Guard-keyword rewording judgment calls: avoided the INTERRUPTION keyword "
        "fallbacks ('one more thing', 'before you continue', 'hold on a second') — "
        "turn 9 uses 'bear with me' instead; 'this is taking a while' chosen over "
        "the FRUSTRATED_PATTERNS regex 'this is taking too long'; no utterance "
        "contains transfer phrases ('real person', 'speak to someone', ...) or "
        "ABUSE_PATTERNS words. retries=2: most non-deterministic scenario in the "
        "suite — each probe depends on LLM guard/extraction classification."
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# J. Name confirmation
# ──────────────────────────────────────────────────────────────────────────────

name_confirmation_happy_path = Scenario(
    name="name_confirmation_happy_path",
    flow="pcp",
    timeout_s=300,
    user_turns=[
        "I need to find a primary care physician.",
        "emily",
        "carter",
        "yes that's correct",  # name readback → confirmed
        "m nine zero seven five zero three",
        "April twelfth nineteen eighty eight",
        "I'm the plan holder",
        "Primary Care Physician",
        "yes that's correct",
        "email please",
        "yes that's correct",
        "no thanks",
        "no thank you",
        "no, that's everything",
    ],
    turn_expectations={
        3: TurnExpectation(
            ai_contains=[r"E-M-I-L-Y.*C-A-R-T-E-R|E-M-I-L-Y\s+C-A-R-T-E-R"],
        ),
        4: TurnExpectation(ai_contains=[r"member\s*id"], slot_awaiting="member_id"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "name_confirmed": True,
            "provider_list_sent": True,
        },
        transcript_contains=[r"E-M-I-L-Y"],
    ),
    notes=(
        "Baseline: name is confirmed on the first readback. "
        "Asserts the readback appears (turn 3 expectation) and that "
        "member_id collection follows immediately after (turn 4 expectation)."
    ),
)

name_confirmation_inline_correction = Scenario(
    name="name_confirmation_inline_correction",
    flow="pcp",
    timeout_s=360,
    retries=1,
    user_turns=[
        "I need to find a primary care physician.",
        "emily",
        "carter",
        "no, it's Emma Carter",  # inline correction → agent re-reads "E-M-M-A  C-A-R-T-E-R"
        "yes that's correct",  # confirm corrected name
        "m nine zero seven five zero three",
        "April twelfth nineteen eighty eight",
        "emma",
        "carter",
        "no, it's Emily Carter",  # inline correction → agent re-reads "E-M-M-A  C-A-R-T-E-R"
        "yes that's correct",  # confirm corrected name
        "m nine zero seven five zero three",
        "April twelfth nineteen eighty eight",
        "I'm the plan holder",
        "Primary Care Physician",
        "yes that's correct",
        "email please",
        "yes that's correct",
        "no thanks",
        "no thank you",
        "no, that's everything",
    ],
    turn_expectations={
        3: TurnExpectation(ai_contains=[r"E-M-I-L-Y.*C-A-R-T-E-R"]),  # first readback
        4: TurnExpectation(ai_contains=[r"E-M-M-A.*C-A-R-T-E-R"]),  # corrected readback
        5: TurnExpectation(ai_contains=[r"member\s*id"], slot_awaiting="member_id"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "name_confirmed": True,
            "first_name": "Emma",
            "last_name": "Carter",
            "provider_list_sent": True,
        },
        transcript_contains=[r"E-M-M-A", r"E-M-I-L-Y"],
    ),
    notes=(
        "Member gives inline correction 'no, it's Emma Carter'. "
        "Agent re-reads back the corrected name; member confirms. "
        "first_name must be 'Emma' in final state, not 'Emily'."
    ),
)

name_confirmation_bare_no_then_gives_name = Scenario(
    name="name_confirmation_bare_no_then_gives_name",
    flow="pcp",
    timeout_s=360,
    retries=1,
    user_turns=[
        "I need to find a primary care physician.",
        "emily",
        "carter",
        "no",  # bare no → agent asks for correct name
        "it's Emma",
        "no",
        "Brown",  # member gives correct name
        "yes that's correct",  # confirm new readback "E-M-M-A  B-R-O-W-N"
        "m nine zero seven five zero three",
        "April twelfth nineteen eighty eight",
        "I'm the plan holder",
        "Primary Care Physician",
        "yes that's correct",
        "email please",
        "yes that's correct",
        "no thanks",
        "no thank you",
        "no, that's everything",
    ],
    # turn_expectations={
    #     3: TurnExpectation(ai_contains=[r"E-M-I-L-Y.*C-A-R-T-E-R"]),
    #     4: TurnExpectation(ai_contains=[r"correct.*name|correct name|what.*name"]),
    #     5: TurnExpectation(ai_contains=[r"E-M-M-A.*B-R-O-W-N"]),
    #     6: TurnExpectation(ai_contains=[r"member\s*id"], slot_awaiting="member_id"),
    # },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "name_confirmed": True,
            "first_name": "Emma",
            "last_name": "Brown",
            "member_status_verify": True,
            "provider_list_sent": True,
        },
        transcript_contains=[r"E-M-M-A", r"B-R-O-W-N"],
    ),
    notes=(
        "Member says bare 'no' — agent asks for the correct name. "
        "Member provides 'Emma Brown'. Agent reads back E-M-M-A B-R-O-W-N. "
        "Member confirms. Flow then proceeds to member_id."
    ),
)

name_confirmation_exhaust_escalates = Scenario(
    name="name_confirmation_exhaust_escalates",
    flow="pcp",
    user_turns=[
        "I need to find a primary care physician.",
        "emily",
        "carter",
        "no",  # rejection 1 → asks for correct name
        "hmm, I'm not sure",  # can't extract a name
        "no",  # readback re-delivered; rejection 2
        "I don't know",  # can't extract a name
        "no",  # rejection 3 → escalate
        # spares in case a clarify turn fires
        "still no",
        "nope",
    ],
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        transfer_initiator="Agent",
        escalation_reason_contains="name_confirm_exhausted",
        final_state={"member_status_verify": falsy},
        transcript_contains=[r"E-M-I-L-Y"],
    ),
    notes=(
        "Member rejects every readback without providing a valid name. "
        "After MAX_NAME_CONFIRM_ATTEMPTS the agent escalates. "
        "member_status_verify must remain False — SF lookup never ran."
    ),
)

name_confirmation_claim_flow = Scenario(
    name="name_confirmation_claim_flow",
    flow="claim",
    timeout_s=360,
    user_turns=[
        "I want to follow up on a claim adjustment.",
        "james",
        "wilson",
        "yes that's right",  # name confirmed
        "m three one zero one eight eight",
        "Thirtieth of July, nineteen seventy seven",
        "yes correct",  # phone confirmed
        "42695817",
        "Can I ask my doctor to send it over?",
        "Yes, please",
        "Yes, that's correct",
        "Perfect. Please do that",
        "You can send me the updates to my phone",
        "Yes, that's correct",
        "Okay, how long will it take?",
        "email them to me",
        "No, that's all. Thanks!",
    ],
    turn_expectations={
        3: TurnExpectation(ai_contains=[r"J-A-M-E-S.*W-I-L-S-O-N"]),
        4: TurnExpectation(ai_contains=[r"member\s*id"], slot_awaiting="member_id"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "name_confirmed": True,
            "claim_flow_complete": True,
        },
        transcript_contains=[r"J-A-M-E-S"],
    ),
    notes=(
        "Confirms the name readback works identically in the claim_services "
        "call_intent path (uses verification_claims.md extraction prompt)."
    ),
)

name_confirmation_single_letter_first_name = Scenario(
    name="name_confirmation_single_letter_first_name",
    flow="pcp",
    timeout_s=300,
    retries=1,
    user_turns=[
        "I need to find a primary care physician.",
        "aj",  # unusual short first name
        "smith",
        "yes that's correct",
        "m nine zero seven five zero three",
        "April twelfth nineteen eighty eight",
        "I'm the plan holder",
        "Primary Care Physician",
        "yes that's correct",
        "email please",
        "yes that's correct",
        "no thanks",
        "no thank you",
        "no, that's it",
    ],
    turn_expectations={
        3: TurnExpectation(ai_contains=[r"A-J.*S-M-I-T-H"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={"name_confirmed": True},
        transcript_contains=[r"A-J"],
    ),
    notes=(
        "Edge case: very short first name. _spell_name('Aj', 'Smith') must "
        "produce 'A-J  S-M-I-T-H', not crash or produce empty output."
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# M. New-intent mid-session (member raises a DIFFERENT service during follow-up)
#
# Exercises the NEW_INTENT path: when a verified member asks about a different
# service during the follow-up phase, follow_up_agent classifies
# follow_up_intent=new_intent (with detected_intent), and — for a fresh intake
# intent — fully RESETS the call (reset_for_new_intent) and re-routes through
# the verification node. The member re-verifies, then verification dispatches on
# pending_intent straight to the new intent's domain agent. Both directions are
# covered.
# ──────────────────────────────────────────────────────────────────────────────

pcp_then_claim_new_intent = Scenario(
    name="pcp_then_claim_new_intent",
    flow="pcp",
    timeout_s=420,
    retries=1,  # new_intent classification is LLM-driven
    user_turns=_PCP_TO_FOLLOW_UP
    + [
        # Follow-up phase: the member pivots to a brand-new claim request.
        # This fully resets the call and re-routes through verification.
        "Actually, can you check a claim reprocessing for me?",
        # Re-verification (claims slot order: first/last name → readback →
        # member id → dob → phone confirmation). Same caller, Emily M907503.
        "emily",
        "carter",
        "yes correct",
        "m nine zero seven five zero three",
        "April twelfth nineteen eighty eight",
        "yes that's correct",  # phone confirmation
        # Now in claim_adjustment, which asks for the adjustment reference number.
        # The only adjustment fixture (42695817) is linked to James (M310188), NOT
        # Emily — so it can never resolve for her member id → not-found → re-ask →
        # deterministic escalation.
        "42695817",
        "42695817",
    ],
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        escalation_reason_contains="adjustment_reference_not_found",
        final_state={
            # The member was RE-verified after the pivot, so by the end they are
            # verified again under the new intent.
            "member_status_verify": True,
            # call_intent is the detected new intent; the claim flow ran under it.
            "call_intent": "claim_services",
            # pending_intent was consumed by verification's dispatch.
            "pending_intent": lambda v: not v,
        },
        # Proof of re-verification (agent re-asks identity) AND that the claim
        # agent engaged after re-verification (asks for the reference number).
        transcript_contains=[r"first name", r"reference number"],
    ),
    notes=(
        "New-intent pivot PCP -> claim WITH re-verification. Emily completes the "
        "provider flow, then during follow-up asks about a claim reprocessing. "
        "follow_up_agent classifies new_intent (detected_intent=claim_services); "
        "is_new_intake_intent is True, so reset_for_new_intent clears identity + "
        "verification + domain state, stages pending_intent=claim_services, and the "
        "node routes to verification. Emily re-verifies (claims slot order ends "
        "with phone confirmation), then verification consumes pending_intent and "
        "dispatches to claim_adjustment_agent. Terminal state is the documented "
        "adjustment_reference_not_found escalation, because reference 42695817 is "
        "linked to James (M310188), not Emily. FIXTURE NOTE: claims re-verification "
        "confirms Emily's phone on file — Emily M907503 must have a phone number. "
        "retries=1: the new_intent classification is LLM-driven."
    ),
)

claim_then_pcp_new_intent = Scenario(
    name="claim_then_pcp_new_intent",
    flow="claim",
    mutating=True,  # provider flow writes James's ZIP (he has none on file)
    timeout_s=480,
    retries=1,  # new_intent classification + provider/delivery slots are LLM-driven
    user_turns=CLAIM_VERIFY
    + [
        # Complete the claim flow up to the follow-up phase.
        "42695817",
        "Can I ask my doctor to send it over?",  # doctor-direct
        "Yes, please",  # accept upload link
        "Yes, that's correct",  # email on file (upload link)
        "Perfect. Please do that",  # accept Personal Guide
        "You can send me the updates to my phone",  # SMS notifications
        "Yes, that's correct",  # confirm phone
        "Okay, how long will it take to finalize the request?",  # timeline question
        "email them to me",  # N2 channel
        # Follow-up phase: the member pivots to a brand-new provider search.
        # This fully resets the call and re-routes through verification.
        "Can I also find an in-network doctor?",  # new_intent -> provider_services
        # Re-verification (provider slot order: first/last name → readback →
        # member id → dob → relationship). Same caller, James M310188.
        "james",
        "wilson",
        "yes correct",
        "m three one zero one eight eight",
        "Thirtieth of July, nineteen seventy seven",
        "I'm the plan holder",  # relationship
        # Now in provider_search.
        "Primary Care Physician",  # provider type
        "zero two one three nine",  # ZIP (James has none on file -> fresh ask)
        "email please",  # delivery method
        "yes that's correct",  # email on file (james.wilson@gmail.com)
        "no thanks",  # decline benefits offer (James has no benefit plan)
        "no thank you",  # decline Care Coach
        "no, that's all, thanks",  # close
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            # Re-verified after the pivot — verified again under the new intent.
            "member_status_verify": True,
            # follow_up_agent staged the detected new intent; verification
            # dispatched to provider_search under it.
            "call_intent": "provider_services",
            # pending_intent was consumed by verification's dispatch.
            "pending_intent": lambda v: not v,
            # The provider flow ran to completion AFTER re-verification.
            "provider_list_sent": True,
            "delivery_method": "email",
        },
        # Proof of re-verification: identity is re-asked after the pivot.
        transcript_contains=[r"first name"],
    ),
    notes=(
        "New-intent pivot claim -> PCP WITH re-verification. James completes the "
        "claim flow, then during follow-up asks to find an in-network doctor. "
        "follow_up_agent classifies new_intent (detected_intent=provider_services); "
        "is_new_intake_intent is True, so reset_for_new_intent clears identity + "
        "verification + claim/follow-up state, stages pending_intent=provider_services, "
        "and the node routes to verification. James re-verifies (provider slot order "
        "ends with relationship), then verification consumes pending_intent and "
        "dispatches to provider_search_agent. The provider flow then completes (email "
        "delivery), declining the benefits/Care-Coach offers (James has no "
        "M_Benefit_Plan__c). Marked mutating: James has no ZIP on file, so the "
        "provider flow writes one (02139); preflight snapshots/restores contact "
        "fields. retries=1: new_intent classification plus provider_type/"
        "delivery_method extraction are LLM-driven."
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# M2. Follow-up RE-SCREEN through intake (front-door screening on a mid-call pivot)
#
# A fresh intake intent raised during follow-up is routed back through the
# INTAKE node (not straight to verification) so intake re-applies its front-door
# screening. Two triggers:
#
#   * provider_services is in follow_up.INTAKE_RESCREEN_INTENTS, so a new provider
#     request re-runs intake's unsupported-provider-type gate. The payoff: an
#     UNSUPPORTED specialty escalates at intake BEFORE any identity is
#     re-collected (vs. the old direct-to-verification path that re-verified
#     first). A SUPPORTED specialty re-classifies cleanly and completes.
#   * Appeals / grievances are out_of_scope but the follow-up classifier has no
#     tag for them, so follow_up._is_appeal_or_grievance() catches them by keyword
#     and reroutes through intake, whose out_of_scope screening routes the caller
#     to the right team and hard-ENDs.
#
# The decisive, LLM-independent fact in the escalate / out_of_scope cases is that
# the member is NEVER re-verified — member_status_verify and first_name are falsy
# at END (the pivot reset cleared them and screening fired before identity was
# re-collected). That is what proves the re-screen ran through the intake node.
#
# ASSERTION NOTE: agent-side escalations (the unsupported-provider case) do not
# surface a harvestable escalation reason or AgentCallTransfer event in this
# codebase — signal_escalate and escalation_agent both emit metadata_events=[],
# and the reason lives only in last_agent_signal, which escalation_agent
# overwrites with a COMPLETE signal before the next graph pause. So these
# scenarios assert on escalated (via the reference number escalation_agent
# stamps), the staged pre-escalation message, the final AI text, and the
# no-re-verification state — not on escalation_reason_contains / transfer_event.
# The out_of_scope cases DO carry a top-level escalation_reason, so they assert it
# the same way intake_out_of_scope_appeal does.
# ──────────────────────────────────────────────────────────────────────────────

followup_unsupported_provider_rescreen = Scenario(
    name="followup_unsupported_provider_rescreen",
    flow="pcp",
    timeout_s=360,
    retries=1,  # follow_up new_intent + intake unsupported-type classification are LLM-driven
    user_turns=_PCP_TO_FOLLOW_UP
    + [
        # Follow-up phase: pivot to a brand-new UNSUPPORTED provider request.
        # provider_services is a re-screen intent → _reroute_through_intake →
        # intake re-classifies → provider_type_unsupported → escalates, all
        # BEFORE any re-verification.
        "Actually, I need to find an oncologist instead.",
    ],
    expect=Expected(
        completed=True,  # reaches END via escalation_agent
        escalated=True,  # escalation_agent stamps an escalation_reference_number
        transfer_event=False,  # current code emits no AgentCallTransfer metadata event
        final_is_interrupt=False,
        final_state={
            # DECISIVE: the unsupported type was rejected at intake's front door,
            # before identity was re-collected. If either of these is truthy the
            # re-screen wrongly went through verification first.
            "member_status_verify": falsy,
            "first_name": falsy,
            # pending_intent is cleared on the intake re-screen path.
            "pending_intent": falsy,
            # The unsupported-type message is staged for escalation_agent.
            "escalation_pre_message": contains("Orthopedic"),
        },
        # The member hears the specialty named plus the five supported types.
        last_ai_contains=[
            r"oncologist",
            r"(Primary Care|Pediatrician|Cardiologist|Dermatologist|Orthopedic)",
        ],
    ),
    notes=(
        "Re-screen payoff. Emily completes the provider flow, then during follow-up "
        "asks for an oncologist. follow_up classifies new_intent "
        "(detected_intent=provider_services); is_new_intake_intent is True and "
        "provider_services is in INTAKE_RESCREEN_INTENTS, so _reroute_through_intake "
        "resets the call, CLEARS call_intent, and routes to the intake node. Intake "
        "re-classifies the request as provider_type_unsupported and escalates "
        "immediately — member_status_verify must be falsy because no re-verification "
        "ran. retries=1: new_intent + unsupported-type classification are LLM-driven."
    ),
)

followup_supported_provider_rescreen = Scenario(
    name="followup_supported_provider_rescreen",
    flow="pcp",
    timeout_s=420,
    retries=1,  # follow_up new_intent + provider/delivery slots are LLM-driven
    user_turns=_PCP_TO_FOLLOW_UP
    + [
        # Follow-up phase: pivot to a brand-new SUPPORTED provider request
        # (dermatologist is one of the five supported types). Re-screens through
        # intake, re-classifies cleanly as provider_services, then completes.
        "Actually, I also need to find a dermatologist.",
        # Re-verification (provider slot order: first/last name → readback →
        # member id → dob → relationship). Same caller, Emily M907503.
        "emily",
        "carter",
        "yes correct",
        "m nine zero seven five zero three",
        "April twelfth nineteen eighty eight",
        "I'm calling for myself",  # relationship
        # Now in provider_search under the re-screened intent.
        "Dermatologist",  # provider type
        "yes that's correct",  # zip on file confirmed
        "send it to my fax",  # delivery method
        "yes that's correct",  # fax on file confirmed
        "no thanks",  # decline benefits
        "no thank you",  # decline Care Coach
        "no, that's everything",  # close
    ],
    expect=Expected(
        completed=True,
        escalated=False,  # supported type must NOT escalate
        final_state={
            # Re-verified after the pivot — verified again under the re-screened intent.
            "member_status_verify": True,
            "call_intent": "provider_services",
            "provider_type": "Dermatologist",
            "provider_list_sent": True,
            # No mid-call-switch dispatch on the intake re-screen path.
            "pending_intent": falsy,
        },
        # Proof of re-verification: identity is re-asked after the pivot.
        transcript_contains=[r"first name"],
    ),
    notes=(
        "Supported re-screen completes end to end. Emily completes the provider flow, "
        "then during follow-up asks for a dermatologist (a supported specialty). "
        "_reroute_through_intake resets the call and routes to intake, which "
        "re-classifies provider_services, re-sets call_intent, and emits its own "
        "first-name bridge. Emily re-verifies and the second provider flow completes "
        "(fax delivery, declining benefits/Care-Coach). Proves the reset → intake "
        "re-screen → verify → domain path is intact for the happy case. retries=1: "
        "new_intent + provider_type/delivery_method extraction are LLM-driven."
    ),
)

followup_appeal_rescreen = Scenario(
    name="followup_appeal_rescreen",
    flow="pcp",
    retries=1,  # intake out_of_scope classification is the primary signal
    user_turns=_PCP_TO_FOLLOW_UP
    + [
        # Follow-up phase: raise an appeal. The keyword gate fires regardless of
        # the follow-up LLM tag and reroutes through intake → out_of_scope.
        "Actually, I'd like to appeal a denial on my claim.",
    ],
    expect=Expected(
        completed=True,  # graph ENDs directly via intake out_of_scope (no escalation_agent)
        escalated=False,
        transfer_event=False,
        final_is_interrupt=False,
        last_ai_contains=[r"appeal", r"1-\d{3}-\d{3}-\d{4}"],
        final_state={
            "escalation_reason": contains("outside covered workflows"),
            # No re-verification — out_of_scope is decided at the front door.
            "member_status_verify": falsy,
            "first_name": falsy,
        },
    ),
    notes=(
        "Appeal raised in follow-up. follow_up._is_appeal_or_grievance() catches the "
        "'appeal' keyword and calls _reroute_through_intake, which resets the call "
        "and routes to intake. Intake classifies out_of_scope and routes the caller "
        "to the appeals team (1-800-555-0105) with a hard END. member_status_verify "
        "must be falsy — the request never reached re-verification. retries=1: the "
        "intake out_of_scope classification is LLM-driven."
    ),
)

followup_grievance_rescreen = Scenario(
    name="followup_grievance_rescreen",
    flow="pcp",
    retries=1,
    user_turns=_PCP_TO_FOLLOW_UP
    + [
        "Actually, I want to file a grievance about how my claim was handled.",
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        transfer_event=False,
        final_is_interrupt=False,
        # A team number is given; "grievance" is NOT in OUT_OF_SCOPE_KEYWORD_ROUTING,
        # so it falls back to the default support team rather than a dedicated one.
        last_ai_contains=[r"1-\d{3}-\d{3}-\d{4}"],
        final_state={
            "escalation_reason": contains("outside covered workflows"),
            "member_status_verify": falsy,
            "first_name": falsy,
        },
    ),
    notes=(
        "Grievance half of APPEAL_GRIEVANCE_KEYWORDS. The keyword gate reroutes "
        "through intake → out_of_scope, hard END, no re-verification. Asserts the "
        "out_of_scope OUTCOME (reason + a routed number), NOT a specific team, "
        "because 'grievance' is not yet in OUT_OF_SCOPE_KEYWORD_ROUTING and therefore "
        "falls back to the default support team — this scenario documents that gap "
        "rather than masking it. retries=1: intake out_of_scope classification is "
        "LLM-driven."
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# N. Follow-up disposition routing, update detours & WAIT (Phases 4-7)
#
# Live mirrors of the offline scenario matrix in test_scenario_matrix_phase7.py.
# The offline matrix asserts LLM call counts and attempt deltas with mocks;
# these run the same conversational shapes against the REAL LLMs and assert
# what a live run can prove: slot_awaiting on the AI prompt that follows the
# turn (Option A — Python appends the next ask / detour ask), detour pointers
# via prompt routing, parked_followups lifecycle, loop-guard escalation, and
# the static WAIT acks. All use Emily Carter (M907503) on the PCP flow.
# ──────────────────────────────────────────────────────────────────────────────

# Minimal PCP completion tail after relationship (ZIP on file → email on file
# → decline benefits → decline Care Coach → close).
_PCP_TAIL = [
    "Primary Care Physician",
    "yes that's correct",  # ZIP on file
    "email please",
    "yes that's correct",  # email on file
    "no thanks",  # decline benefits
    "no thank you",  # decline Care Coach
    "no, that's everything",  # close
]

# MSG_WAIT_NUDGE has a {slot_label} placeholder — build the regex directly.
_WAIT_NUDGE_MEMBER_ID = r"whenever you'?re ready.*member id"

followup_answer_confirmed_slot = Scenario(
    name="followup_answer_confirmed_slot",
    flow="pcp",
    timeout_s=360,
    retries=1,  # answered_with_followup classification is LLM-driven
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "carter",
        "yes correct",  # name_confirmed
        # Answer + side question about an already-confirmed slot →
        # FOLLOWUP_ANSWER: Gemini answers from Confirmed, Python appends the
        # static DOB ask — proven by the turn-5 expectation.
        "it's m nine zero seven five zero three — and did you get my last name right?",
        "April twelfth nineteen eighty-eight",
        "I'm calling for myself",
    ]
    + _PCP_TAIL,
    turn_expectations={
        5: TurnExpectation(ai_contains=[r"(date of birth|birth\s*date|dob)"], slot_awaiting="dob"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "last_name": "Carter",
            "provider_list_sent": True,
        },
    ),
    notes=(
        "Row 4 of the Phase 7 matrix, live. The member answers member_id AND asks "
        "about a confirmed slot in one utterance. The turn must confirm the slot "
        "(no re-ask), address the question, and end with the DOB ask appended by "
        "Python — slot_awaiting='dob' before the DOB answer is the proof that the "
        "follow-up did not stall or re-ask member_id."
    ),
)

followup_park_question_deferred = Scenario(
    name="followup_park_question_deferred",
    flow="pcp",
    timeout_s=360,
    retries=2,  # disposition (park) classification + parked answer are LLM-driven
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "carter",
        "yes correct",
        # Answer + question answerable LATER in the call → FOLLOWUP_PARK:
        # the question is queued in parked_followups, flow moves straight on.
        "m nine zero seven five zero three — will I get a text when the provider list is sent?",
        "April twelfth nineteen eighty-eight",
        "I'm calling for myself",
    ]
    + _PCP_TAIL
    + [
        "no, that's all, thanks",  # spare: follow_up answers the parked question first
    ],
    turn_expectations={
        5: TurnExpectation(ai_contains=[r"(date of birth|birth\s*date|dob)"], slot_awaiting="dob"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "provider_list_sent": True,
            # follow_up surfaced and consumed the parked question at the end.
            "parked_followups": falsy,
        },
    ),
    notes=(
        "Row 5 of the Phase 7 matrix, live. The side question maps to a later "
        "stage → parked (never answered mid-verification, flow proceeds straight "
        "to DOB), then surfaced and cleared by follow_up_agent at the end of the "
        "call. Final parked_followups must be empty; the spare closing turn "
        "absorbs the extra follow-up exchange when the parked answer is delivered."
    ),
)

followup_decline_irrelevant = Scenario(
    name="followup_decline_irrelevant",
    flow="pcp",
    timeout_s=360,
    retries=1,
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "carter",
        "yes correct",
        # Answer + never-answerable side question → FOLLOWUP_DECLINE: brief warm
        # decline, then the appended DOB ask — the flow never stalls.
        "m nine zero seven five zero three — quick question, do you sell car insurance?",
        "April twelfth nineteen eighty-eight",
        "I'm calling for myself",
    ]
    + _PCP_TAIL,
    turn_expectations={
        5: TurnExpectation(ai_contains=[r"(date of birth|birth\s*date|dob)"], slot_awaiting="dob"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={"member_status_verify": True, "provider_list_sent": True},
    ),
    notes=(
        "Row 6 of the Phase 7 matrix, live. 'Do you sell car insurance?' is the "
        "canonical decline example from the extraction header. The value must be "
        "captured, the question declined without an apology spiral, and the DOB "
        "ask appended — asserted via slot_awaiting='dob' on the next prompt."
    ),
)

correction_inline_case_a = Scenario(
    name="correction_inline_case_a",
    flow="pcp",
    timeout_s=360,
    retries=2,  # inline answer+correction extraction is mildly non-deterministic
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "carson",  # WRONG last name, confirmed at the read-back
        "yes correct",  # read-back "Emily Carson" confirmed
        # Case A: answer + correction WITH a valid value in one utterance.
        # The correction applies before the answer confirms; both are
        # acknowledged and the DOB ask is appended.
        "m nine zero seven five zero three — actually my last name is Carter, not Carson",
        "April twelfth nineteen eighty-eight",
        "I'm calling for myself",
    ]
    + _PCP_TAIL,
    turn_expectations={
        3: TurnExpectation(ai_contains=[r"E-M-I-L-Y.*C-A-R-S-O-N"]),  # read-back of the wrong name
        5: TurnExpectation(ai_contains=[r"(date of birth|birth\s*date|dob)"], slot_awaiting="dob"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,  # lookup matched the CORRECTED name
            "first_name": "Emily",
            "last_name": "Carter",
            "provider_list_sent": True,
        },
        transcript_count={r"C-A-R-S-O-N": 1},  # no second read-back after the correction
    ),
    notes=(
        "Row 7 of the Phase 7 matrix, live (Case A). The confirmed-then-corrected "
        "last name must be replaced with the validated new value in ONE turn — no "
        "detour, no second name read-back — and the Salesforce lookup must match "
        "on the corrected name. slot_awaiting='dob' after the correction turn "
        "proves member_id confirmed and the pipeline advanced."
    ),
)

update_without_value_case_b = Scenario(
    name="update_without_value_case_b",
    flow="pcp",
    timeout_s=360,
    retries=2,  # update_target extraction is LLM-driven
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "carter",
        "yes correct",
        # Case B: answer + update request WITHOUT a value → the awaiting slot
        # confirms, then a detour asks for the new value (replacing the normal
        # next-slot ask). correction_return_to brings the pipeline back to DOB.
        "m nine zero seven five zero three — oh, also I need to update my last name",
        "Carter",  # the "new" value (same as on file, so the lookup still matches)
        "April twelfth nineteen eighty-eight",
        "I'm calling for myself",
    ]
    + _PCP_TAIL,
    turn_expectations={
        # Detour: the prompt after the update request asks for the LAST NAME —
        # not DOB — proving awaiting_slot switched to the update target.
        5: TurnExpectation(ai_contains=[r"last\s*name"], slot_awaiting="last_name"),
        # After the new value, the pipeline resumes at DOB (correction_return_to).
        6: TurnExpectation(ai_contains=[r"(date of birth|birth\s*date|dob)"], slot_awaiting="dob"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "last_name": "Carter",
            "provider_list_sent": True,
        },
    ),
    notes=(
        "Row 9 of the Phase 7 matrix, live (Case B). The member answers member_id "
        "and asks to change their last name without giving one. The turn confirms "
        "member_id, opens a detour (awaiting_slot=last_name — asserted), and after "
        "the new value the pipeline returns to DOB (asserted) instead of re-asking "
        "member_id."
    ),
)

bare_update_detour_c2 = Scenario(
    name="bare_update_detour_c2",
    flow="pcp",
    timeout_s=360,
    retries=2,
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "carter",
        "yes correct",
        "m nine zero seven five zero three",
        # Case C2: bare update request (no value, no answer) at the DOB ask →
        # detour to last_name; the DOB ask is preserved in correction_return_to.
        "wait — before that, I need to change my last name",
        "Carter",  # new value (same as on file)
        "April twelfth nineteen eighty-eight",
        "I'm calling for myself",
    ]
    + _PCP_TAIL,
    turn_expectations={
        5: TurnExpectation(ai_contains=[r"(date of birth|birth\s*date|dob)"], slot_awaiting="dob"),
        # After the bare update request the agent asks for the last name.
        6: TurnExpectation(ai_contains=[r"last\s*name"], slot_awaiting="last_name"),
        # After the new value, the pipeline returns to the original DOB ask.
        7: TurnExpectation(ai_contains=[r"(date of birth|birth\s*date|dob)"], slot_awaiting="dob"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "last_name": "Carter",
            "provider_list_sent": True,
        },
    ),
    notes=(
        "Row 10 of the Phase 7 matrix, live (Case C2). A value-less update request "
        "interrupting the DOB collection detours to the update target and then "
        "returns to DOB — the 5/6/7 turn expectations trace the full detour "
        "round-trip (dob → last_name → dob). The DOB attempt budget must not be "
        "consumed by the detour: any exhaustion here fails the run."
    ),
)

locked_field_update_declined = Scenario(
    name="locked_field_update_declined",
    flow="pcp",
    timeout_s=360,
    retries=2,
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "carter",
        "yes correct",
        # Bare update request for a caller-LOCKED field (phone_number) at the
        # member_id ask → declined; the flow stays on member_id (no detour).
        "before I give you that — can you change the phone number on my file?",
        "m nine zero seven five zero three",
        "April twelfth nineteen eighty-eight",
        "I'm calling for myself",
    ]
    + _PCP_TAIL,
    turn_expectations={
        4: TurnExpectation(ai_contains=[r"member\s*id"], slot_awaiting="member_id"),
        # The decline re-asks the SAME slot — awaiting stays member_id.
        5: TurnExpectation(slot_awaiting="member_id"),
        6: TurnExpectation(ai_contains=[r"(date of birth|birth\s*date|dob)"], slot_awaiting="dob"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "provider_list_sent": True,
        },
    ),
    notes=(
        "Row 12 of the Phase 7 matrix, live. phone_number is in "
        "CALLER_LOCKED_SLOTS: the update request must be declined — no detour "
        "(awaiting stays member_id, asserted on turn 5), nothing applied, no "
        "escalation — and verification proceeds normally."
    ),
)

update_loop_guard_escalates = Scenario(
    name="update_loop_guard_escalates",
    flow="pcp",
    timeout_s=360,
    retries=2,
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "carter",
        "yes correct",
        # Update #1 (bare) at the member_id ask → detour, counter update_last_name=1
        "actually, I need to change my last name",
        "Carter",  # detour collects the value, pipeline returns to member_id
        # Update #2 for the SAME field → guard_loop_limit (max 2) → escalation
        "hmm, no wait — I need to change my last name again",
    ],
    turn_expectations={
        5: TurnExpectation(ai_contains=[r"last\s*name"], slot_awaiting="last_name"),
        6: TurnExpectation(ai_contains=[r"member\s*id"], slot_awaiting="member_id"),
    },
    expect=Expected(
        completed=True,  # END via escalation_agent
        escalated=True,
        transfer_event=True,
        transfer_initiator="Agent",
        escalation_reason_contains="update_last_name",
        final_state={"member_status_verify": falsy},
    ),
    notes=(
        "Row 13 of the Phase 7 matrix, live. Two update detours for the same "
        "field exhaust the per-target budget (guard_loop_limit, counter "
        "update_last_name, max 2): the second request escalates with the "
        "exhausted-style copy instead of opening a third detour loop."
    ),
)

wait_ack_then_answer = Scenario(
    name="wait_ack_then_answer",
    flow="pcp",
    timeout_s=360,
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "carter",
        "yes correct",
        "give me a minute",  # WAIT → static ack, no attempt cost, no Gemini
        "m nine zero seven five zero three",
        "April twelfth nineteen eighty-eight",
        "I'm calling for myself",
    ]
    + _PCP_TAIL,
    turn_expectations={
        # The prompt after the wait is the static ack pool, still awaiting member_id.
        5: TurnExpectation(ai_contains=[pool_regex(MSG_WAIT_ACK)], slot_awaiting="member_id"),
        6: TurnExpectation(ai_contains=[r"(date of birth|birth\s*date|dob)"], slot_awaiting="dob"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={"member_status_verify": True, "provider_list_sent": True},
        transcript_contains=[pool_regex(MSG_WAIT_ACK)],
    ),
    notes=(
        "Rows 14-15 of the Phase 7 matrix, live. A bare wait gets the static "
        "acknowledgement (MSG_WAIT_ACK pool — asserted verbatim via pool_regex, "
        "proving no LLM generated it) with awaiting_slot unchanged and no retry "
        "burned; the member then answers and verification completes normally."
    ),
)

wait_nudge_after_three = Scenario(
    name="wait_nudge_after_three",
    flow="pcp",
    timeout_s=360,
    retries=1,  # three consecutive turns must all classify as WAIT
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "carter",
        "yes correct",
        "give me a minute",  # wait #1 → ack
        "just a sec",  # wait #2 → ack
        "hold on",  # wait #3 → nudge naming the slot
        "m nine zero seven five zero three",
        "April twelfth nineteen eighty-eight",
        "I'm calling for myself",
    ]
    + _PCP_TAIL,
    turn_expectations={
        5: TurnExpectation(ai_contains=[pool_regex(MSG_WAIT_ACK)], slot_awaiting="member_id"),
        6: TurnExpectation(ai_contains=[pool_regex(MSG_WAIT_ACK)], slot_awaiting="member_id"),
        # Third consecutive wait: the gentle nudge that names the slot.
        7: TurnExpectation(ai_contains=[_WAIT_NUDGE_MEMBER_ID], slot_awaiting="member_id"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={"member_status_verify": True, "provider_list_sent": True},
        transcript_contains=[_WAIT_NUDGE_MEMBER_ID],
    ),
    notes=(
        "Rows 16-17 of the Phase 7 matrix, live. Three consecutive waits: the "
        "first two get static acks, the third the MSG_WAIT_NUDGE naming the "
        "member id. None of them burn a retry — the member then answers once and "
        "verification completes, which would be impossible if the waits had "
        "consumed the MAX_SLOT_ATTEMPTS=3 budget. Also covers the regex rescue: "
        "'just a sec' / 'hold on' still WAIT even if the LLM mislabels them."
    ),
)

# ──────────────────────────────────────────────────────────────────────────────
# K. Indirect-decline regression (delivery_management fax/email)
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# L. Cannot-provide short-circuit (detect_cannot_provide + slot_manager /
#    claim_adjustment_agent)
# ──────────────────────────────────────────────────────────────────────────────
from tests.live_e2e.test_cannot_provide import (  # noqa: E402
    CANNOT_PROVIDE_SCENARIOS,
)
from tests.live_e2e.test_fax_indirect_decline import (  # noqa: E402
    INDIRECT_DECLINE_SCENARIOS,
)

# ──────────────────────────────────────────────────────────────────────────────
# O. Production-transcript regressions (LLM-2 hygiene, Phases 1-4)
#    Each scenario mirrors one real production transcript that exposed a bug.
# ──────────────────────────────────────────────────────────────────────────────

emily_carter_correction_single_ask = Scenario(
    name="emily_carter_correction_single_ask",
    flow="pcp",
    timeout_s=360,
    retries=2,  # CORRECTED-event classification is LLM-driven
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "carson",  # WRONG last name, confirmed at the read-back
        "yes correct",  # read-back "Emily Carson" confirmed
        "m nine zero seven five zero three",
        # Production transcript (Bug A): pure name correction at the DOB ask.
        # The broken behavior acknowledged the correction and then DOUBLE-asked
        # ("…could you confirm your Member ID number again? And what's your
        # date of birth?"). The fixed turn must re-ask DOB and nothing else.
        "wait — actually my name is Emily Carter, not Carson",
        "April twelfth nineteen eighty-eight",
        "I'm calling for myself",
    ]
    + _PCP_TAIL,
    turn_expectations={
        5: TurnExpectation(ai_contains=[r"(date of birth|birth\s*date|dob)"], slot_awaiting="dob"),
        # The correction ack re-asks DOB ONLY — awaiting must still be dob and
        # the confirmed member_id must not be re-asked (transcript_count below).
        6: TurnExpectation(ai_contains=[r"(date of birth|birth\s*date|dob)"], slot_awaiting="dob"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,  # lookup matched the CORRECTED name
            "first_name": "Emily",
            "last_name": "Carter",
            "provider_list_sent": True,
        },
        transcript_count={
            # The exact production double-ask: a re-ask/re-confirm of the
            # already-confirmed member_id must never appear anywhere.
            r"member\s*id[^.?!]*again": 0,
            r"confirm your member\s*id": 0,
        },
    ),
    notes=(
        "Mirrors the Emily Carter production transcript (Bug A, Phase 2). A "
        "confirmed-name correction arrives at the DOB ask; the response must "
        "acknowledge and re-ask DOB in one sentence — the single-ask sanitizer "
        "strips any re-ask of the confirmed member_id, asserted via the zero "
        "transcript_count entries and slot_awaiting staying on dob."
    ),
)

notification_followup_not_declined = Scenario(
    name="notification_followup_not_declined",
    flow="pcp",
    timeout_s=360,
    retries=2,  # park disposition classification is LLM-driven
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "carter",
        "yes correct",
        # Production transcript (Bug B): a notification question asked during
        # member_id collection was DECLINED ("that part I can't help with").
        # It concerns a later stage of this same call → must PARK, then be
        # answered by follow_up at the end.
        "m nine zero seven five zero three — will I get a notification when the list is sent out?",
        "April twelfth nineteen eighty-eight",
        "I'm calling for myself",
    ]
    + _PCP_TAIL
    + [
        "no, that's all, thanks",  # spare: follow_up answers the parked question first
    ],
    turn_expectations={
        # The park ack must not stall the pipeline: the same turn ends on the
        # appended DOB ask.
        5: TurnExpectation(ai_contains=[r"(date of birth|birth\s*date|dob)"], slot_awaiting="dob"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "provider_list_sent": True,
            "parked_followups": falsy,  # surfaced and consumed by follow_up
        },
        transcript_count={
            # The production decline wording, in either its old or new form —
            # a later-stage notification question must never be declined.
            r"(can'?t|cannot|not able to)\s+(help|assist)": 0,
            r"representative will need to (help|make that change)": 0,
        },
    ),
    notes=(
        "Mirrors the notification-question production transcript (Bug B, "
        "Phase 3). 'Will I get a notification when it's sent?' during member_id "
        "must park (header.md: delivery/notification/timeline questions are "
        "never declined), keep the flow moving to DOB, and be answered by "
        "follow_up before close — final parked_followups empty, zero decline "
        "phrasings anywhere in the transcript."
    ),
)

zip_update_during_fax_confirmation = Scenario(
    name="zip_update_during_fax_confirmation",
    flow="pcp",
    mutating=True,
    timeout_s=420,
    retries=2,  # update_target extraction mid-delivery is LLM-driven
    user_turns=PCP_VERIFY
    + [
        "Primary Care Physician",
        "yes that's correct",  # ZIP on file confirmed
        "send it to my fax",  # delivery method
        # Production transcript run-df1e16a9 (Bug C): at the fax read-back the
        # member says their ZIP changed. The broken behavior repeated the fax
        # question over the request; the fix routes to provider_search NOW.
        "wait — actually my ZIP code changed, I moved recently",
        "zero two one four one",  # new ZIP, collected by provider_search
        "yes that's correct",  # fax read-back re-asked on resume → confirm
        "no thanks",  # decline benefits
        "no thank you",  # decline Care Coach
        "no that's all, thanks",  # close
    ],
    turn_expectations={
        # Before the ZIP interjection: the fax read-back question.
        10: TurnExpectation(ai_contains=[r"fax"], slot_awaiting="fax_confirmed"),
        # The hand-off: honest "update your ZIP first" ask — awaiting flips to
        # zip_code and the next turn is owned by provider_search.
        11: TurnExpectation(ai_contains=[r"zip"], slot_awaiting="zip_code"),
        # The resume: ZIP-update acknowledgement naming the NEW ZIP plus the
        # re-asked fax read-back — dispatch never fired from the disputed ZIP.
        12: TurnExpectation(ai_contains=[r"02141", r"fax"], slot_awaiting="fax_confirmed"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "provider_list_sent": True,
            "delivery_method": "fax",
            "zip_code_used": "02141",
            "zip_code_updated": True,
            "pending_slot_update": falsy,  # round-trip fully consumed
        },
        transcript_contains=[
            # Dispatch confirmation names the NEW ZIP (list rebuilt from it).
            _zip_dispatch_regex("02141"),
        ],
    ),
    post_checks=[sf_field_check("M907503", "zip_code", "02141")],
    notes=(
        "Mirrors production run-df1e16a9 (Bug C, Phase 4). Mutates Emily's zip "
        "in Salesforce; teardown restores the snapshot. A ZIP update requested "
        "at the fax read-back routes to provider_search (pending_slot_update), "
        "collects + persists the new ZIP, and the orchestrator fast-path "
        "returns to delivery_management at fax_confirmed with the update "
        "acknowledged. The provider list must be dispatched from the NEW ZIP "
        "only — asserted by the ZIP-aware dispatch message, zip_code_used, and "
        "the Salesforce post-check."
    ),
)

# ──────────────────────────────────────────────────────────────────────────────
# P. Cross-agent redo/replay requests (Phase 6)
#
# Phase 4's registry routes slot VALUE updates. Phase 6 adds the two further
# request kinds real calls contain: redo (re-perform a completed action with
# a changed parameter) and replay (re-state information already given).
# ──────────────────────────────────────────────────────────────────────────────

redo_fax_to_email_from_benefits = Scenario(
    name="redo_fax_to_email_from_benefits",
    flow="pcp",
    timeout_s=420,
    retries=2,  # request_kind extraction is LLM-driven
    user_turns=PCP_VERIFY
    + [
        "Primary Care Physician",
        "yes that's correct",  # ZIP on file confirmed
        "send it to my fax",  # delivery method
        "yes that's correct",  # fax confirmed → dispatch + benefits offer
        "yes please",  # benefits → explanation + Care Coach offer
        # Phase 6 (a): mid Care-Coach offer, the member wants the ALREADY
        # DISPATCHED list re-sent by another method — a redo_action, not a
        # slot update. benefits routes to delivery, which re-dispatches.
        "actually can you send that list to my email instead of fax",
        "yes that's correct",  # email read-back confirmed → re-dispatch
        "no thank you",  # Care Coach re-offer declined (where we left off)
        "no that's all, thanks",  # close
    ],
    turn_expectations={
        # Before the redo: the Care Coach offer (benefits agent).
        12: TurnExpectation(ai_contains=[r"[Cc]oach"], slot_awaiting="care_coach_response"),
        # The hop landed in delivery's re-dispatch branch: email read-back.
        13: TurnExpectation(ai_contains=[r"email"], slot_awaiting="email_confirmed"),
        # The resume: re-send acknowledged AND the Care Coach offer re-asked —
        # never the benefits offer again.
        14: TurnExpectation(ai_contains=[r"email", r"[Cc]oach"], slot_awaiting="care_coach_response"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "provider_list_sent": True,
            "delivery_method": "email",
            "benefits_offer_made": True,
            "pending_cross_agent_request": falsy,  # round-trip fully consumed
        },
        transcript_contains=[r"(?i)same .{0,40}list|as well"],
    ),
    notes=(
        "Phase 6 (a): a fax→email redo requested from benefits_agent after "
        "dispatch routes to delivery_management (capability registry), "
        "re-dispatches by email, does NOT repeat the benefits offer "
        "(benefits_offer_made stays True), and returns to benefits at the "
        "Care Coach offer where the call left off."
    ),
)

replay_benefits_from_follow_up = Scenario(
    name="replay_benefits_from_follow_up",
    flow="pcp",
    timeout_s=420,
    retries=2,
    user_turns=PCP_VERIFY
    + [
        "Primary Care Physician",
        "yes that's correct",
        "send it to my fax",
        "yes that's correct",
        "yes please",  # benefits explained + Care Coach offer
        "no thank you",  # Care Coach declined → care_wellness → follow-up stage
        # Phase 6 (b): a replay_info request after the benefits flow finished.
        # No update_target fires and benefits_inquiry is not an intake intent —
        # the capability registry is the only way to honor this.
        "can you repeat my benefits again?",
        "no that's all, thanks",  # close
    ],
    turn_expectations={
        # The replay: benefits re-explained (deductible read again) — and the
        # Care Coach must NOT be re-offered on a routed replay.
        14: TurnExpectation(ai_contains=[r"(?i)deductible"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "pending_cross_agent_request": falsy,
            "benefits_explained": True,
        },
        # The benefits summary appears twice: the original explanation and
        # the replay.
        transcript_count={r"(?i)individual deductible": 2},
    ),
    notes=(
        "Phase 6 (b): 'repeat my benefits' voiced at the post-flow stage "
        "routes through follow_up to benefits_agent via the capability "
        "registry, re-explains (fetch_benefits is idempotent), skips a second "
        "Care Coach offer, and hands back to follow_up for the close."
    ),
)

redo_inflow_before_dispatch = Scenario(
    name="redo_inflow_before_dispatch",
    flow="pcp",
    timeout_s=420,
    retries=2,
    user_turns=PCP_VERIFY
    + [
        "Primary Care Physician",
        "yes that's correct",
        "send it to my fax",
        # Phase 6 (c): the owner IS the active agent — switching fax→email
        # while still in delivery before dispatch resolves in-flow: the
        # existing delivery_method/contact branches handle it, zero routing.
        "actually email is better",
        "yes that's correct",  # email read-back → dispatch + benefits offer
        "no thanks",  # decline benefits
        "no thank you",  # decline Care Coach
        "no that's all, thanks",
    ],
    turn_expectations={
        11: TurnExpectation(ai_contains=[r"email"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "provider_list_sent": True,
            "delivery_method": "email",
            "pending_cross_agent_request": falsy,  # never set — no hop
        },
    ),
    notes=(
        "Phase 6 (c): a method switch voiced while delivery_management is "
        "active and the list is NOT yet dispatched stays in-flow — no "
        "pending_cross_agent_request, no orchestrator hop."
    ),
)

replay_benefits_inflow_at_coach_offer = Scenario(
    name="replay_benefits_inflow_at_coach_offer",
    flow="pcp",
    timeout_s=420,
    retries=2,
    user_turns=PCP_VERIFY
    + [
        "Primary Care Physician",
        "yes that's correct",
        "send it to my fax",
        "yes that's correct",
        "yes please",  # benefits explained + Care Coach offer
        # Phase 6 (c): replay of benefits' own material while benefits is
        # active — in-flow re-explain + re-ask, zero routing.
        "sorry, can you repeat my benefits again?",
        "no thank you",  # Care Coach declined
        "no that's all, thanks",
    ],
    turn_expectations={
        # The in-flow replay: benefits re-explained AND the offer re-asked.
        13: TurnExpectation(
            ai_contains=[r"(?i)deductible", r"[Cc]oach"], slot_awaiting="care_coach_response"
        ),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={"pending_cross_agent_request": falsy},
    ),
    notes=(
        "Phase 6 (c): 'repeat my benefits' during benefits' own Care Coach "
        "offer resolves in-flow — re-explanation + re-offer in one turn, no "
        "routing."
    ),
)

unknown_replay_topic_parks = Scenario(
    name="unknown_replay_topic_parks",
    flow="pcp",
    timeout_s=420,
    retries=2,
    user_turns=PCP_VERIFY
    + [
        "Primary Care Physician",
        "yes that's correct",
        "send it to my fax",
        "yes that's correct",
        "yes please",  # benefits explained + Care Coach offer
        # Phase 6 (d): a replay request for a topic no capability owns.
        # Must park as a question (Phase 3 path) — never a hard decline.
        "can you go over my claim history again?",
        "no thank you",  # Care Coach declined
        "what about that claim history?",  # follow_up answers/cannot-answer
        "no that's all, thanks",
    ],
    turn_expectations={
        # The park acknowledgement + the Care Coach offer re-asked, no
        # decline phrasing.
        13: TurnExpectation(ai_contains=[r"[Cc]oach"], slot_awaiting="care_coach_response"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "pending_cross_agent_request": falsy,
            "parked_followups": falsy,  # consumed by follow_up
        },
    ),
    notes=(
        "Phase 6 (d): an unknown replay topic ('claim history') parks as a "
        "kind=question item instead of declining; follow_up surfaces it at "
        "the post-flow stage. The call never escalates over it."
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# Registry — run order matters (scenarios share Salesforce data; run serially)
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# Q. Paraphrase robustness (Phases 1-7)
#
# Every scripted interjection below re-exercises a fixed production scenario
# with NOVEL wording that appears in neither the extraction-prompt examples
# nor the earlier scenarios. The deterministic request-detection layer (plus
# the branch-level switch machinery) must carry these paraphrases even when
# the extraction LLM under-delivers — the same guarantee the offline
# test_paraphrase_flows.py asserts with the LLM fully disabled.
# ──────────────────────────────────────────────────────────────────────────────

paraphrase_zip_update_at_fax_readback = Scenario(
    name="paraphrase_zip_update_at_fax_readback",
    flow="pcp",
    mutating=True,
    timeout_s=420,
    retries=2,
    user_turns=PCP_VERIFY
    + [
        "Primary Care Physician",
        "yes that's correct",  # ZIP on file confirmed
        "send it to my fax",  # delivery method
        # BUG-5 paraphrase — no prompt example uses this wording.
        "hang on — we've recently moved, so that old zip won't work",
        "zero two one four two",  # new ZIP, collected by provider_search
        "yes that's correct",  # fax read-back re-asked on resume → confirm
        "no thanks",
        "no thank you",
        "no that's all, thanks",
    ],
    turn_expectations={
        10: TurnExpectation(ai_contains=[r"fax"], slot_awaiting="fax_confirmed"),
        11: TurnExpectation(ai_contains=[r"zip"], slot_awaiting="zip_code"),
        12: TurnExpectation(ai_contains=[r"02142", r"fax"], slot_awaiting="fax_confirmed"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "provider_list_sent": True,
            "delivery_method": "fax",
            "zip_code_used": "02142",
            "zip_code_updated": True,
            "pending_slot_update": falsy,
        },
        transcript_contains=[_zip_dispatch_regex("02142")],
    ),
    post_checks=[sf_field_check("M907503", "zip_code", "02142")],
    notes=(
        "Paraphrased BUG-5 (mirrors zip_update_during_fax_confirmation with "
        "novel wording: 'we've recently moved, so that old zip won't work'). "
        "Mutates Emily's zip in Salesforce; teardown restores the snapshot."
    ),
)

paraphrase_channel_switch_at_fax_readback = Scenario(
    name="paraphrase_channel_switch_at_fax_readback",
    flow="pcp",
    timeout_s=420,
    retries=2,
    user_turns=PCP_VERIFY
    + [
        "Primary Care Physician",
        "yes that's correct",  # ZIP on file
        "send it to my fax",  # delivery method chosen: fax
        # BUG-3 paraphrase at the fax read-back: switch channels, never a
        # failed confirmation, never a verbatim fax re-ask.
        "you know what, just shoot it over by email instead",
        "yes that's correct",  # email read-back → dispatch by email
        "no thanks",  # decline benefits
        "no thank you",  # decline Care Coach
        "no that's all, thanks",
    ],
    turn_expectations={
        # The AI prompt after the switch must be the EMAIL read-back.
        11: TurnExpectation(ai_contains=[r"email"], slot_awaiting="email_confirmed"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "provider_list_sent": True,
            "delivery_method": "email",
        },
    ),
    notes=(
        "Paraphrased BUG-3: 'just shoot it over by email instead' at the fax "
        "read-back must flip the channel to the email confirmation — the fax "
        "question is never repeated."
    ),
)

paraphrase_redo_from_care_coach = Scenario(
    name="paraphrase_redo_from_care_coach",
    flow="pcp",
    timeout_s=420,
    retries=2,
    user_turns=PCP_VERIFY
    + [
        "Primary Care Physician",
        "yes that's correct",
        "send it to my fax",
        "yes that's correct",  # dispatch by fax + benefits offer
        "yes please",  # benefits explained + Care Coach offer
        # BUG-2 paraphrase mid Care-Coach offer.
        "hmm, on second thought could you shoot that list over by email instead of faxing it?",
        "yes that's correct",  # email read-back → re-dispatch
        "no thank you",  # Care Coach re-offer declined
        "no that's all, thanks",
    ],
    turn_expectations={
        12: TurnExpectation(ai_contains=[r"[Cc]oach"], slot_awaiting="care_coach_response"),
        13: TurnExpectation(ai_contains=[r"email"], slot_awaiting="email_confirmed"),
        14: TurnExpectation(ai_contains=[r"email", r"[Cc]oach"], slot_awaiting="care_coach_response"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "provider_list_sent": True,
            "delivery_method": "email",
            "benefits_offer_made": True,
            "pending_cross_agent_request": falsy,
        },
        transcript_contains=[r"(?i)same .{0,40}list|as well"],
    ),
    notes=(
        "Paraphrased BUG-2 (mirrors redo_fax_to_email_from_benefits with "
        "novel wording). The redo routes benefits → delivery, re-dispatches "
        "by email, and returns to the Care Coach offer exactly once."
    ),
)

paraphrase_replay_benefits_followup = Scenario(
    name="paraphrase_replay_benefits_followup",
    flow="pcp",
    timeout_s=420,
    retries=2,
    user_turns=PCP_VERIFY
    + [
        "Primary Care Physician",
        "yes that's correct",
        "send it to my fax",
        "yes that's correct",
        "yes please",  # benefits explained + Care Coach offer
        "no thank you",  # Care Coach declined → follow-up stage
        # Replay paraphrase at the post-flow stage.
        "please go over my benefits once more",
        "no that's all, thanks",
    ],
    turn_expectations={
        14: TurnExpectation(ai_contains=[r"(?i)deductible"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "pending_cross_agent_request": falsy,
            "benefits_explained": True,
        },
        transcript_count={r"(?i)individual deductible": 2},
    ),
    notes=(
        "Paraphrased benefits replay ('go over my benefits once more') at "
        "follow-up — routes to benefits via the capability registry even if "
        "the extraction LLM returns nothing (regex fallback covers it)."
    ),
)

paraphrase_identity_update_mid_verification = Scenario(
    name="paraphrase_identity_update_mid_verification",
    flow="pcp",
    timeout_s=420,
    retries=2,
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily",
        "carter",
        "yes correct",  # name readback confirmed
        # BUG-4 paraphrase: member id answered AND a last-name update
        # requested in the same breath — never parked, never declined.
        "m nine zero seven five zero three — and also, my last name is different now",
        "sorry, it's actually still carter",  # the "new" last name (keeps SF verify passing)
        "yes correct",  # the readback re-runs for the changed name
        "April twelvee nineteen eighty-eight",  # dob — the preserved next slot
        "I'm calling for myself",
        "yes that's correct",  # ZIP on file
        "send it to my fax",
        "yes that's correct",
        "no thanks",
        "no thank you",
        "no that's all, thanks",
    ],
    turn_expectations={
        # The detour: the very next AI turn asks for the new last name.
        5: TurnExpectation(ai_contains=[r"last name"], slot_awaiting="last_name"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "provider_list_sent": True,
        },
    ),
    notes=(
        "Paraphrased BUG-4: the member-id answer is captured AND the "
        "last-name detour opens immediately; the spelling readback re-runs "
        "for the changed name and verification resumes at dob."
    ),
)

paraphrase_claims_identity_update_at_reference = Scenario(
    name="paraphrase_claims_identity_update_at_reference",
    flow="claim",
    timeout_s=480,
    retries=2,
    user_turns=CLAIM_VERIFY
    + [
        # Phase 7 paraphrase at the reference-number ask: identity updates
        # route to verification and return to the exact awaiting slot.
        "I have to change my last name first",
        "it's still wilson actually",  # keeps the SF lookup passing
        "yes correct",  # name readback re-confirm
        "42695817",  # the reference ask resumes here
        "Can I ask my doctor to send it over?",
        "Yes, please",
        "Yes, that's correct",
        "Perfect. Please do that",
        "You can send me the updates to my phone",
        "Yes, that's correct",
        "Okay, how long will it take to finalize the request?",
        "email them to me",
        "No, that's all. Thanks!",
    ],
    turn_expectations={
        # The route: the next AI turn asks for the (new) last name.
        7: TurnExpectation(ai_contains=[r"last name"], slot_awaiting="last_name"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "member_status_verify": True,
            "claim_flow_complete": True,
        },
    ),
    notes=(
        "Phase 7 paraphrase: 'I have to change my last name first' while the "
        "claim flow awaits the reference number routes to verification "
        "(re-collect + re-verify + readback) and resumes at the reference ask."
    ),
)

paraphrase_notification_switch = Scenario(
    name="paraphrase_notification_switch",
    flow="claim",
    timeout_s=420,
    retries=2,
    user_turns=CLAIM_VERIFY
    + [
        "42695817",
        "Can I ask my doctor to send it over?",
        "Yes, please",
        "Yes, that's correct",  # upload-link email confirmed
        "Perfect. Please do that",  # Personal Guide accepted
        "You can send me the updates to my phone",  # SMS chosen
        # Phase 7 paraphrase at the phone read-back: a channel switch, never
        # a decline of the phone number.
        "honestly, email works better for me at this point",
        "Yes, that's correct",  # email read-back confirmed
        "Okay, how long will it take to finalize the request?",
        "email them to me",  # N2 channel
        "No, that's it for me. Thanks!",
    ],
    turn_expectations={
        # After the switch: the EMAIL read-back, never "what is the correct
        # phone number?".
        14: TurnExpectation(ai_contains=[r"email"], slot_awaiting="email_confirmed"),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "notification_channel": "email",
            "claim_flow_complete": True,
        },
    ),
    notes=(
        "Phase 7 paraphrase: 'email works better for me' during the SMS phone "
        "confirmation switches the notification channel to email and "
        "continues that channel's confirmation."
    ),
)

paraphrase_claim_status_replay_followup = Scenario(
    name="paraphrase_claim_status_replay_followup",
    flow="claim",
    timeout_s=480,
    retries=2,
    user_turns=CLAIM_VERIFY
    + [
        "42695817",
        "Can I ask my doctor to send it over?",
        "Yes, please",
        "Yes, that's correct",
        "Perfect. Please do that",
        "You can send me the updates to my phone",
        "Yes, that's correct",
        "Okay, how long will it take to finalize the request?",
        "email them to me",
        # Phase 7 paraphrase at follow-up: a claim-status question phrased
        # nothing like the prompt examples — answered from real state
        # (replay hop or grounded snapshot answer), never invented.
        "any idea when someone will get back to me about the adjustment?",
        "No, that's all. Thanks!",
    ],
    turn_expectations={
        # The reply must carry real adjustment facts (status/timeline).
        17: TurnExpectation(ai_contains=[r"(?i)(review|business days|adjustment)"]),
    },
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "claim_flow_complete": True,
            "pending_cross_agent_request": falsy,
        },
    ),
    notes=(
        "Phase 7 paraphrase: a post-flow claim-status question routes to "
        "claim_adjustment's replay (or answers grounded from the snapshot) — "
        "the reply must restate the real status/timeline, never invent one."
    ),
)


SCENARIOS: list[Scenario] = [
    # A. PCP happy paths
    pcp_happy_path_fax,  # 1
    pcp_happy_path_email,  # 2
    pcp_benefits_declined,  # 3
    pcp_zip_update,  # 4  (mutating)
    pcp_zip_inline_update,  # 5  (mutating)
    pcp_fax_update,  # 6  (mutating)
    pcp_email_update,  # 7  (mutating)
    # B. Verification escalations
    verification_restart_then_success,  # 8
    verification_fail_twice_escalates,  # 9
    member_id_exhaustion,  # 10
    dob_no_year_exhaustion,  # 11
    member_id_ambiguous_exhaustion,  # 12
    # B2. Partial re-ask on identity mismatch (member found, field(s) wrong)
    verification_dob_only_mismatch,  # 12a
    verification_last_name_only_mismatch,  # 12b
    verification_first_name_only_mismatch,  # 12c
    verification_name_mismatch_bare_no_at_readback,  # 12d
    verification_multi_field_mismatch_generic,  # 12e
    verification_member_id_not_found_restart,  # 12f
    verification_repeated_dob_mismatch_escalates,  # 12g
    # C. Guard escalations
    transfer_request,  # 13
    abuse,  # 14
    self_harm,  # 15
    offtopic_repeated,  # 16
    # D. Intake routing
    intake_unclear_exhaustion,  # 17
    intake_out_of_scope_billing,  # 18
    intake_out_of_scope_appeal,  # 18b — appeal must NOT route to claim_services
    non_member_caller,  # 19
    intake_unsupported_provider_oncologist,  # 20a
    intake_unsupported_provider_neurologist,  # 20b
    intake_supported_provider_cardiologist,  # 20c (regression guard)
    intake_generic_provider_request,  # 20d (regression guard)
    intake_provider_type_propagates_to_search,  # 20e (intake→search propagation)
    # E. Claim flow
    claim_happy_path,  # 21
    claim_upload_only,  # 22
    claim_guide_only,  # 23
    claim_no_proceed,  # 24
    phone_not_confirmed_ends_call,  # 25
    ref_not_found_retry_then_success,  # 26
    ref_not_found_twice_escalates,  # 27
    ref_exhaustion,  # 28
    claim_email_change_on_upload,  # 29 (mutating)
    # F. Follow-up escalations
    follow_up_update_request,  # 30
    follow_up_cannot_answer_x3,  # 31
    # M. New-intent mid-session (member pivots to a different service in follow-up)
    pcp_then_claim_new_intent,  # 31a — PCP -> claim new_intent (no re-verify)
    claim_then_pcp_new_intent,  # 31b — claim -> PCP new_intent (mutating)
    # M2. Follow-up re-screen through intake (front-door screening on a mid-call pivot)
    followup_unsupported_provider_rescreen,  # 31c — unsupported provider escalates pre-reverify
    followup_supported_provider_rescreen,  # 31d — supported provider re-screen completes
    followup_appeal_rescreen,  # 31e — appeal keyword → out_of_scope
    followup_grievance_rescreen,  # 31f — grievance keyword → out_of_scope
    # G. Contact-change loop limits
    zip_change_loop_escalates,  # 32  (redefined: invalid-ZIP slot exhaustion)
    email_change_loop_in_notification,  # 33 (mutating)
    # G2. Notification contact-confirmation advances on first affirmative (regression)
    notification_phone_confirm_advances,  # 33a
    notification_phone_confirm_bare_yes_advances,  # 33b
    notification_email_confirm_advances,  # 33c
    # H. Conversational & confusion-recovery
    pcp_happy_path_conversational,  # 34
    claim_happy_path_conversational,  # 35
    pcp_confused_member,  # 36
    claim_confused_member,  # 37
    pcp_conversational_confusion,  # 38
    claim_conversational_confusion,  # 39
    # I. Boundary stress
    boundary_walk_claim,  # 40
    # J. Name confirmation
    name_confirmation_happy_path,  # NC-1
    name_confirmation_inline_correction,  # NC-2
    name_confirmation_bare_no_then_gives_name,  # NC-3
    name_confirmation_exhaust_escalates,  # NC-4
    name_confirmation_claim_flow,  # NC-5
    name_confirmation_single_letter_first_name,  # NC-6
    # K. Indirect-decline regression
    *INDIRECT_DECLINE_SCENARIOS,  # ID-1, ID-2, ID-3
    # L. Cannot-provide short-circuit
    *CANNOT_PROVIDE_SCENARIOS,  # CP-1 … CP-6
    # N. Follow-up disposition routing, update detours & WAIT (Phases 4-7)
    followup_answer_confirmed_slot,  # N-1 — FOLLOWUP_ANSWER + appended static ask
    followup_park_question_deferred,  # N-2 — FOLLOWUP_PARK → answered in follow_up
    followup_decline_irrelevant,  # N-3 — FOLLOWUP_DECLINE + appended static ask
    correction_inline_case_a,  # N-4 — Case A: answer + valid correction
    update_without_value_case_b,  # N-5 — Case B: detour + return pointer
    bare_update_detour_c2,  # N-6 — Case C2: bare update detour round-trip
    locked_field_update_declined,  # N-7 — LOCKED field update → decline
    update_loop_guard_escalates,  # N-8 — per-target loop guard escalation
    wait_ack_then_answer,  # N-9 — static wait ack, value wins after
    wait_nudge_after_three,  # N-10 — 3 waits → slot-naming nudge
    # O. Production-transcript regressions (LLM-2 hygiene, Phases 1-4)
    emily_carter_correction_single_ask,  # O-1 — Bug A: correction double-ask
    notification_followup_not_declined,  # O-2 — Bug B: later-stage question declined
    zip_update_during_fax_confirmation,  # O-3 — Bug C: zip-update routing (mutating)
    # P. Cross-agent redo/replay requests (Phase 6)
    redo_fax_to_email_from_benefits,  # P-1 — (a) redo routes benefits → delivery → back
    replay_benefits_from_follow_up,  # P-2 — (b) replay routes follow_up → benefits → back
    redo_inflow_before_dispatch,  # P-3 — (c) owner active pre-dispatch → in-flow, zero routing
    replay_benefits_inflow_at_coach_offer,  # P-4 — (c) in-flow benefits replay, zero routing
    unknown_replay_topic_parks,  # P-5 — (d) unknown replay topic parks as a question
    # Q. Paraphrase robustness (Phases 1-7) — novel wording, same outcomes
    paraphrase_zip_update_at_fax_readback,  # Q-1 — BUG-5 paraphrase (mutating)
    paraphrase_channel_switch_at_fax_readback,  # Q-2 — BUG-3 paraphrase
    paraphrase_redo_from_care_coach,  # Q-3 — BUG-2 paraphrase
    paraphrase_replay_benefits_followup,  # Q-4 — benefits replay paraphrase
    paraphrase_identity_update_mid_verification,  # Q-5 — BUG-4 paraphrase
    paraphrase_claims_identity_update_at_reference,  # Q-6 — claims identity paraphrase
    paraphrase_notification_switch,  # Q-7 — notification channel-switch paraphrase
    paraphrase_claim_status_replay_followup,  # Q-8 — claim-status replay paraphrase
]

SCENARIOS_BY_NAME: dict[str, Scenario] = {s.name: s for s in SCENARIOS}
