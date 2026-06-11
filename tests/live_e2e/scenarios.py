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
# Registry — run order matters (scenarios share Salesforce data; run serially)
# ──────────────────────────────────────────────────────────────────────────────

SCENARIOS: list[Scenario] = [
    # A. PCP happy paths
    pcp_happy_path_fax,  # 1
    pcp_happy_path_email,  # 2
    pcp_benefits_declined,  # 3
    pcp_zip_update,  # 4  (mutating)
    pcp_fax_update,  # 5  (mutating)
    # B. Verification escalations
    verification_restart_then_success,  # 6
    verification_fail_twice_escalates,  # 7
    member_id_exhaustion,  # 8
    dob_no_year_exhaustion,  # 9
    # C. Guard escalations
    transfer_request,  # 10
    abuse,  # 11
    self_harm,  # 12
    offtopic_repeated,  # 13
    # D. Intake routing
    intake_unclear_exhaustion,  # 14
    intake_out_of_scope_billing,  # 15
    non_member_caller,  # 16
    # E. Claim flow
    claim_happy_path,  # 17
    claim_upload_only,  # 18
    claim_guide_only,  # 19
    claim_no_proceed,  # 20
    phone_not_confirmed_ends_call,  # 21
    ref_not_found_retry_then_success,  # 22
    ref_not_found_twice_escalates,  # 23
    ref_exhaustion,  # 24
    claim_email_change_on_upload,  # 25 (mutating)
    # F. Follow-up escalations
    follow_up_update_request,  # 26
    follow_up_cannot_answer_x3,  # 27
    # G. Contact-change loop limits
    zip_change_loop_escalates,  # 28
    email_change_loop_in_notification,  # 29 (mutating)
]

SCENARIOS_BY_NAME: dict[str, Scenario] = {s.name: s for s in SCENARIOS}
