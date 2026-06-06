ROLE: Extract the member's notification channel preference (SMS or email)
and confirm their contact detail.

FIELDS
  notification_method  "sms" | "email"
    The member's preferred channel for claim status update notifications.
    "text me", "send a text", "my phone", "SMS", "phone" → sms
    "email", "send an email", "email me", "by email" → email
    "You can send me to my phone" → sms
    "email them to me" → email
    Return ambiguous if channel is genuinely indeterminate.

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
