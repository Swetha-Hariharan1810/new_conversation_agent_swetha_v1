ROLE: Extract the member's notification channel preference (SMS or email)
and confirm their contact detail.

FIELDS
  timeline_response  "yes" | "no" | "question"
    Whether the member is responding to the timeline offer.
    "how long", "when will", "how many days", "how much longer",
    "what is the timeline", "when will it be done" → question
    "yes", "sure", "go ahead", "okay", "please" → yes
    "no", "no thanks", "that's all", "skip", "I'm done",
    "not right now", "no need", "I don't need that" → no
    Ambiguous or unrelated (e.g. "what is my deductible?") → leave empty,
    event_type "ambiguous"

  notification_method  "sms" | "email" | "none"
    The member's preferred channel for claim status update notifications.
    "text me", "send a text", "my phone", "SMS", "phone" → sms
    "email", "send an email", "email me", "by email" → email
    "You can send me to my phone" → sms
    "email them to me" → email
    "no", "no thanks", "no that's all", "that's all", "I'm done",
    "not right now", "no need", "skip", "I don't need that",
    "nothing", "neither" → none
    Return ambiguous only if channel is genuinely indeterminate
    and not a clear opt-out.

  contact_confirmed  "yes" | "no"
    Whether the member confirms the contact detail (phone or email) just
    read aloud by the agent.
    Bias rule (identical to delivery_management): anything other than a
    clear affirmation → "no".
    "yes", "correct", "that's right", "yep", "yes, that's correct" → yes
    "no", "nope", "that's changed", "use a different one" → no
    If member declines AND provides a replacement in the same utterance,
    extract only the new phone/email; omit contact_confirmed entirely.

  phone  10-digit string
    Updated phone number if member declines the one on file.
    Normalize spoken digits. Return ambiguous if not exactly 10 digits.

  email  valid email string
    Updated email if member declines the one on file.
    Must contain "@" and a domain. Return ambiguous if format unclear.

CONFIDENCE NOTES (see header [ANCHOR: CONFIDENCE])
- contact_confirmed: bias rule — non-clear-affirmation → no.
- phone: not exactly 10 digits → ambiguous.
- email: missing "@" or valid domain → ambiguous.
