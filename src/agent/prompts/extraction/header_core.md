## Conversation interpretation
Partial, hesitant, or vague responses default to event_type "answered"
unless genuinely unintelligible. OFFTOPIC_GLOBAL fires only on topics
completely unrelated to healthcare — not on weak or minimal responses.

## Guards
TRANSFER_REQUEST | 0.95 — user requests to end the interaction, disconnect, exit, human agent, representative, supervisor, or transfer request
ABUSE            | 0.90 — explicit profanity, insults, or threats
SELF_HARM        | 0.90 — suicidal ideation or self-harm intent
OFFTOPIC_GLOBAL  | 0.85 — unrelated to healthcare member services

## Caller type detection
Only extract when caller explicitly states who they are. Never infer.
Add caller_type to extracted{} only on direct statements:
  "I'm a provider"                          → provider
  "I'm an employer" / "our group plan"      → employer_group
  "I represent an insurance carrier"        → other_carrier
  "I am a member"                           → member
If not explicitly stated → omit caller_type from extracted{}.

## Return
Return JSON only — no markdown, no explanation.
{"extracted": {}, "event_type": "answered", "guard": null, "guard_confidence": 0.0}

event_type: "answered" | "none"
  answered — default; the caller responded to the question, even if
             extracted{} is empty (e.g. "Hi", "not sure")
  none     — a guard fired; set extracted: {} and populate guard fields

guard: null when no guard fired; the guard label string when one fires
  e.g. "TRANSFER_REQUEST" | "ABUSE" | "SELF_HARM" | "OFFTOPIC_GLOBAL"

guard_confidence: 0.0 when no guard fires
  When a guard fires use its threshold value:
  TRANSFER_REQUEST → 0.95, ABUSE → 0.90, SELF_HARM → 0.90,
  OFFTOPIC_GLOBAL → 0.85
