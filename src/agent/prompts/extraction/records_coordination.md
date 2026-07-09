ROLE: Extract the member's preferred method for providing medical records
and their consent for Personal Guide outreach.

FIELDS
  upload_method  "member_upload" | "doctor_direct" | "personal_guide" | "decline"
    How the member intends to provide their medical records.

    member_upload — member will upload or send records themselves:
      "I'll send it", "okay will send it", "I can upload it",
      "yes please" (when asked about upload link), "sure send the link"

    doctor_direct — member's doctor/provider will send records directly:
      "Can I ask my doctor to send it over?", "my doctor will send it",
      "I'll have my doctor's office send them", "the provider can send it"

    personal_guide — member wants Sagility Personal Guide to contact provider:
      "you can contact my doctor", "please reach out to them",
      "yes please do that", "Perfect. Please do that"

    decline — member does not want to proceed with any option:
      "no", "no thanks", "I don't want to proceed", "no I don't want to",
      "not right now" (when all options have been offered)

    When the member says something like "okay will send it" or gives a vague
    affirmative BEFORE being offered the upload link, classify as doctor_direct
    (member intends to have it sent) rather than member_upload, unless they
    explicitly agree to receive a link.

  upload_consent  "yes" | "no"
    Whether the member wants to receive the secure upload link via email.
    Only extract when the agent just offered to send a link.
    "Yes please", "sure", "yes" → yes
    "no thanks", "no" → no

  email_confirmed  "yes" | "no"
    Also accepted as: contact_confirmed  "yes" | "no"
    Whether the member confirms the email address just read aloud by the agent.
    Only extract when the agent just read back an email address to the member.

    Clear affirmations → "yes":
      "yes", "correct", "that's right", "yep", "absolutely",
      "yes that's correct", "yes that's my email",
      any imperative consent phrase showing intent to proceed:
      "please do that", "go ahead", "send it", "do it", "sounds good",
      "perfect", "please send it", "yes please"

    Implicit or explicit declines → "no":
      "no", "nope", "that's wrong", "that's not right",
      "that's changed", "that's my old email", "I don't use that anymore",
      "not anymore", "actually no", "use a different one",
      any statement indicating the address is stale, wrong, or no longer used.

    Genuine uncertainty → leave extracted{} empty, event_type "ambiguous":
      "I think so", "maybe", "not sure", "I'm not 100% sure",
      "hmm", "let me think"
      These express doubt about whether the address is correct,
      not a decision to decline it. Re-ask for clarification.

    Key distinction: "that's my old email" is a DECLINE (the member knows
    it is wrong). "I'm not sure if that's still active" is AMBIGUOUS
    (the member does not know). Only use ambiguous when the member
    genuinely cannot confirm or deny.

    If the member declines AND provides a replacement email in the same
    utterance, extract only the new email value into the `email` field;
    omit email_confirmed entirely.

  email  valid email string (must contain "@" and a domain)
    New email address replacing the one on file. Only extract when the
    member is actively providing a replacement.
    Return ambiguous if format is unclear or missing "@".

  personal_guide_consent  "yes" | "no"
    Explicit yes/no consent for the Personal Guide to contact the provider.
    Only extract when the agent has just asked "Would you like us to proceed?"
    regarding Personal Guide outreach. This REQUIRES a clear affirmative for "yes".

    Clear consent → "yes":
      "yes", "sure", "Perfect. Please do that", "yes please",
      "go ahead", "please arrange that", "please reach out to them"

    Clear decline → "no":
      "no", "no I don't want to proceed",
      "not right now" (temporal deferral = decline for this call),
      "maybe some other time" (temporal deferral = decline),
      "some other time", "not today", "maybe later",
      "that's not needed", "no that won't be necessary",
      "I'll handle it myself", "my doctor will send it directly",
      "they'll send it", "the office will handle it",
      any statement that the member or their provider will handle records
      without Personal Guide involvement.

    Temporal deferrals ("maybe some other time", "not right now", "perhaps later")
    are functionally declines for this call — extract "no", not ambiguous.

    Alternative-arrangement statements ("that's not needed, my doctor's office
    will send it directly", "they'll handle it") are also declines — the member
    is indicating they do not want Personal Guide outreach.

    Ambiguous ("maybe", "I think so") → event_type ambiguous, leave empty.

CONFIDENCE NOTES (see header [ANCHOR: CONFIDENCE])
- personal_guide_consent: must be unambiguous. Any doubt → ambiguous.
  Exception: temporal deferrals and alternative-arrangement statements are
  unambiguous declines — extract "no".
- upload_method: when member's first response is vague affirmation before
  upload link is offered ("okay will send it"), use doctor_direct as default.
- email_confirmed / contact_confirmed: stale-address or wrong-address
  statements are unambiguous declines — extract "no". Only use ambiguous
  when the member genuinely does not know whether the address is correct.

## Other-slot changes are never slot answers
A statement that a DIFFERENT slot changed ("my ZIP code changed",
"my address changed", "I moved", "my last name is wrong", "I need to update
my last name") is never an answer to the awaiting slot — return
update_target (e.g. "zip_code", "last_name"), request_kind:"update",
extracted {}. Never classify these as wait or ambiguous, even when prefixed
with a wait word ("wait — my address changed").
