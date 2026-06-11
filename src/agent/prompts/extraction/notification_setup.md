ROLE: Extract the member's notification channel preference (SMS or email)
and confirm their contact detail.

FIELDS
  notification_method  "sms" | "email"
    The member's preferred channel for claim status update notifications.
    "text me", "send a text", "my phone", "SMS", "phone" → sms
    "email", "send an email", "email me", "by email", "mail" → email
    "You can send me to my phone" → sms
    "email them to me" → email
    Return ambiguous if channel is genuinely indeterminate.

  contact_confirmed  "yes" | "no"
    Whether the member confirms the contact detail (phone or email) just
    read aloud by the agent.
    Bias rule (identical to delivery_management): anything other than a
    clear affirmation → "no".
    "yes", "correct", "that's right", "yep", "yes, that's correct" → yes
    "no", "nope", "that's changed", "use a different one",
    "that's my old number", "I changed my email", "that's my old email",
    "I don't use that anymore", "that's outdated",
    "it needs to be updated" → no
    If member declines AND provides a replacement in the same utterance,
    extract only the new phone/email; omit contact_confirmed entirely.

  phone  10-digit string
    Updated phone number if member declines the one on file.
    Normalize spoken digits. Return ambiguous if not exactly 10 digits.

  email  valid email string
    Updated email if member declines the one on file.
    Must contain "@" and a domain. Return ambiguous if format unclear.

  timeline_response  "yes" | "no" | "question"
    The member's response to the timeline offer.
    Extract only when awaiting_slot is "timeline_question".
    "how long", "when will", "how many days", "how much longer",
    "what is the timeline", "when will it be done",
    any question about duration or timing → question
    "yes", "yep","sure", "go ahead", "okay", "please do", "sure go ahead" → yes
    "no", "no thanks", "skip", "that's fine", "move on","I'm good", "no need" → no
    Only extract when the intent is clear, not when the
    member is confused or gave an unclear answer → event_type "ambiguous"

  notification_opted_out  "yes"
    Extract only when awaiting_slot is "n2_notification_method" and the
    member clearly declines to set up N2 notifications.
    "no", "no thanks", "no that's all", "that's all", "I'm done",
    "not right now", "no need", "skip", "I don't need that",
    "nothing", "neither", "I'm all set" → yes
    Only extract when the intent is clearly opt-out, not when the
    member is confused or gave an unclear answer.
    Do not extract alongside notification_method — if you extracted
    notification_method, leave this empty.

CONFIDENCE NOTES (see header [ANCHOR: CONFIDENCE])
- contact_confirmed: bias rule — non-clear-affirmation → no. Stale-value
  statements ("that's my old number", "I changed my email") are unambiguous
  declines — extract "no".
- phone: not exactly 10 digits → ambiguous.
- email: missing "@" or valid domain → ambiguous.
