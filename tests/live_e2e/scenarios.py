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
"""

from __future__ import annotations

# Static pools — imported so assertions survive any re-pick of pool members.
from agent.agents.follow_up.constants import MSG_UPDATE_REQUEST_ESCALATE  # noqa: E402
from agent.agents.verification.handlers import MSG_PHONE_NOT_CONFIRMED  # noqa: E402
from agent.responses.static import MSG_SELF_HARM_ESCALATION  # noqa: E402
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
    "m nine zero seven five zero three",
    "April twelfth nineteen eighty-eight",
    "I'm calling for myself",
]

# Claim flow: intent → first/last name → member id → dob → phone confirmation
CLAIM_VERIFY = [
    "I adjusted the claim and I want to follow up",
    "james",
    "wilson",
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
    "okay so my member id is m nine zero seven five zero three",
    "I was born on the twelfth of april, nineteen eighty eight",
    "it's my own plan, I'm the plan holder",
]

# Claim flow: intent → first/last name → member id → dob → phone confirmation (conversational)
CLAIM_VERIFY_CONVERSATIONAL = [
    "hello, I submitted a claim adjustment a while back and wanted to check on it",
    "yeah it's james",
    "wilson",
    "let me grab my card... okay it's m three one zero one eight eight",
    "the thirtieth of july, nineteen seventy seven",
    "yep, that's the right number",
]

# Verification turn-level sanity checks shared by happy paths
_VERIFY_TURNS = {
    3: TurnExpectation(ai_contains=[r"member\s*id"], slot_awaiting="member_id"),
    4: TurnExpectation(ai_contains=[r"(date of birth|birth\s*date|dob)"], slot_awaiting="dob"),
}


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
        "my new zip code is zero two one three nine",  # spoken 5-digit ZIP
        "yes that's correct",  # confirm read-back → SF write
        "send it to my fax",
        "yes that's correct",
        "no thanks",
        "no thank you",
        "no that's all, thanks",
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={
            "provider_list_sent": True,
            "zip_code_used": "02139",
        },
    ),
    post_checks=[sf_field_check("M907503", "zip_code", "02139")],
    notes="Mutates Emily's zip in Salesforce; teardown restores the snapshot.",
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
        "yes that's correct",  # confirm read-back → SF write
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
        "yes that's correct",  # confirm read-back → SF write
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
        "m nine zero seven five zero two",  # wrong member id — lookup fails
        "April twelfth nineteen eighty-eight",
        # agent restarts ("let's try once more") — give correct details
        "emily",
        "carter",
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
        "m nine zero seven five zero two",  # wrong, round 1
        "April twelfth nineteen eighty-eight",
        "emily",
        "carter",
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
        "Yes, can you tell me where I can see how many rewards I earned "
        "from my annual check up last week?",
        "No, that's it for me. Thanks!",
    ],
    turn_expectations={6: TurnExpectation(ai_contains=[r"reference number"])},
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

follow_up_update_request = Scenario(
    name="follow_up_update_request",
    flow="pcp",
    user_turns=_PCP_TO_FOLLOW_UP
    + [
        "actually can you send it to a different fax number, "
        "six one seven five five five nine nine nine nine",
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
# ──────────────────────────────────────────────────────────────────────────────

zip_change_loop_escalates = Scenario(
    name="zip_change_loop_escalates",
    flow="pcp",
    timeout_s=360,
    user_turns=PCP_VERIFY
    + [
        "Primary Care Physician",
        "no, that's not my zip code",  # reject ZIP on file (cycle 1)
        "zero two one three nine",  # provide new ZIP
        "no wait, it's actually zero two one four zero",  # reject read-back w/ new ZIP (cycle 2)
        "no, actually make that zero two one four one",  # cycle 3 → escalate
        # spares in case a cycle is consumed differently
        "no, that's wrong too — zero two one four two",
        "no, it's zero two one four three",
    ],
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        escalation_reason_regex=r"(zip_change_loop_exceeded|zip_confirmed_exhausted)",
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
# H. Conversational & confusion-recovery
# ──────────────────────────────────────────────────────────────────────────────

pcp_happy_path_conversational = Scenario(
    name="pcp_happy_path_conversational",
    flow="pcp",
    retries=1,
    user_turns=PCP_VERIFY_CONVERSATIONAL
    + [
        "yeah I'm looking for a primary care doctor, or PCP",    # provider type
        "yep that's right",                                      # ZIP on file
        "email is fine",                                         # delivery method
        "yes that's the right one",                              # email on file
        "yeah go ahead please",                                  # accept benefits
        "yeah that sounds good",                                 # accept Care Coach
        "roughly how long before I get the list?",               # follow-up question
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
        "42695817",                                                     # reference number
        "Can I ask my doctor to send it over?",                         # doctor-direct
        "Yes, please",                                                  # accept upload link
        "Yes, that's correct",                                          # confirm email on file
        "Perfect. Please do that",                                      # accept Personal Guide
        "You can send me the updates to my phone",                      # SMS notifications
        "Yes, that's correct",                                          # confirm phone
        "any idea how long this whole process usually takes?",          # natural timeline question
        "email them to me",                                             # N2 channel
        "No, I think we're good. Really appreciate the help!",         # close
    ],
    turn_expectations={6: TurnExpectation(ai_contains=[r"reference number"])},
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
        "Primary Care Physician",                       # provider type
        "wait, what did you say?",                      # ambiguous ZIP read-back → CLARIFY/retry
        "yes that's correct",                           # ZIP confirmed after re-read
        "umm... hold on... actually email is better",   # hedged then resolved delivery method
        "yes that's correct",                           # email on file confirmed
        "do you guys have an app?",                     # benign side question mid-benefits offer
        "oh sorry, yes please go ahead",                # accept benefits after agent redirects
        "no thank you",                                 # decline Care Coach
        "no, that's everything",                        # close
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
        "hold on, let me find the letter...",                    # hesitant non-answer → retry
        "four two six nine five eight one seven",                # reference number on retry
        "sorry, what records do you need exactly?",             # confused → upload_method retry
        "okay, I'll have my doctor send them over",             # doctor_direct on re-ask
        "Yes, please",                                          # accept upload link
        "hmm, I think so? probably",                            # ambiguous email confirm → re-ask
        "yes that's correct",                                   # email confirmed on re-ask
        "Perfect. Please do that",                              # accept Personal Guide
        "You can send me the updates to my phone",              # SMS notifications
        "Yes, that's correct",                                  # confirm phone
        "Okay, how long will it take to finalize the request?", # timeline question
        "email them to me",                                     # N2 channel
        "No, that's it. Thanks!",                               # close
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
        "carter — that's c a r t e r",             # exercises SPELL_CONFIRM: LLM strips
        "okay so my member id is m nine zero seven five zero three",  # the spelling echo
        "I was born on the twelfth of april, nineteen eighty eight",
        "it's my own plan, I'm the plan holder",
        # Provider flow
        "Primary Care Physician",
        # Confusion #1 — ZIP read-back: no yes/no → zip_confirmed slot_fail → RETRY
        "sorry, what was that zip code again?",
        "ah yes, that's right",                     # ZIP confirmed
        # Confusion #2 — delivery method: no channel mention → delivery_method pipeline
        #   ambiguous → retry interrupt; "hold on" noted below (not a transfer request)
        "umm... hold on... what were my options again?",
        "actually, email is better for me",         # delivery_method = email
        "yep, that's the correct email",            # email_confirmed = yes (on file)
        # Confusion #3 — benefits offer: no yes/no for benefits_response → slot_fail
        #   → re-offer in _handle_benefits_response
        "wait, quick thing — do you guys have a mobile app?",
        "oh sorry — yes please, go ahead with the benefits",  # benefits_response = yes
        "yeah that sounds great",                   # accept Care Coach
        "where do I go to check my wellness reward points?",  # answerable follow-up
        "no, that's all for me, thanks so much",    # close
        # 2 spare turns: confusion turns (#1 and #3) may each consume one extra interrupt
        # depending on whether the guard fires for the app question before _handle_benefits
        # re-offers, making total interrupt count variable by ±1–2
        "I'm all set, thanks",
        "that was everything, thank you",
    ],
    turn_expectations={3: TurnExpectation(ai_contains=[r"member\s*id"], slot_awaiting="member_id")},
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
        "let me grab my card... okay, it's m three one zero one eight eight",
        "the thirtieth of july, nineteen seventy seven",
        "yep, that's the right number",             # phone_confirmed = yes
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
        "ah okay — I'll just have my doctor's office send them over",   # doctor_direct
        "Yes, please",                              # accept upload link
        # Confusion #3 — email_confirmed: records_coordination.md explicitly lists
        #   'I think so' / 'probably' → AMBIGUOUS (NOT a 'no') → slot_fail('email_confirmed')
        #   → generate_recovery_message(guard='CLARIFY') gentle re-ask, no email-update path
        "hmm, I think so? probably",
        "yes, that's correct",                      # email confirmed on re-ask
        "Perfect. Please do that",                  # accept Personal Guide
        "you can just text me",                     # notification_method = sms
        "Yes, that's correct",                      # confirm phone on file
        "how long is all of this going to take?",   # timeline question
        "email works for me",                       # N2 channel = email
        "No, that's all. Thanks!",                  # close
        # 2 spare turns: confusion turns (#1 and #2) each consume one extra interrupt;
        # total interrupt count is variable by ±1–2 depending on CLARIFY guard firing
        "I think that covers everything",
        "all done on my end, thanks",
    ],
    turn_expectations={6: TurnExpectation(ai_contains=[r"reference number"])},
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
# Registry — run order matters (scenarios share Salesforce data; run serially)
# ──────────────────────────────────────────────────────────────────────────────

SCENARIOS: list[Scenario] = [
    # A. PCP happy paths
    pcp_happy_path_fax,  # 1
    pcp_happy_path_email,  # 2
    pcp_benefits_declined,  # 3
    pcp_zip_update,  # 4  (mutating)
    pcp_fax_update,  # 5  (mutating)
    pcp_email_update,  # 6  (mutating)
    # B. Verification escalations
    verification_restart_then_success,  # 7
    verification_fail_twice_escalates,  # 8
    member_id_exhaustion,  # 9
    dob_no_year_exhaustion,  # 10
    # C. Guard escalations
    transfer_request,  # 11
    abuse,  # 12
    self_harm,  # 13
    offtopic_repeated,  # 14
    # D. Intake routing
    intake_unclear_exhaustion,  # 15
    intake_out_of_scope_billing,  # 16
    non_member_caller,  # 17
    # E. Claim flow
    claim_happy_path,  # 18
    claim_upload_only,  # 19
    claim_guide_only,  # 20
    claim_no_proceed,  # 21
    phone_not_confirmed_ends_call,  # 22
    ref_not_found_retry_then_success,  # 23
    ref_not_found_twice_escalates,  # 24
    ref_exhaustion,  # 25
    claim_email_change_on_upload,  # 26 (mutating)
    # F. Follow-up escalations
    follow_up_update_request,  # 27
    follow_up_cannot_answer_x3,  # 28
    # G. Contact-change loop limits
    zip_change_loop_escalates,  # 29
    email_change_loop_in_notification,  # 30 (mutating)
    # H. Conversational & confusion-recovery
    pcp_happy_path_conversational,  # 31
    claim_happy_path_conversational,  # 32
    pcp_confused_member,  # 33
    claim_confused_member,  # 34
    pcp_conversational_confusion,  # 35
    claim_conversational_confusion,  # 36
    # I. Boundary stress
    boundary_walk_claim,  # 37
]

SCENARIOS_BY_NAME: dict[str, Scenario] = {s.name: s for s in SCENARIOS}