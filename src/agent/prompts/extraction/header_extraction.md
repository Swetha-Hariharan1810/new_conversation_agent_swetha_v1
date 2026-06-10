## Conversation interpretation
Partial, hesitant, or vague responses still count as answer attempts.

## Guards
TRANSFER_REQUEST | 0.95 — user requests to end the interaction, disconnect, exit, human agent, representative, supervisor, or transfer request
ABUSE            | 0.90 — explicit profanity, insults, or threats
SELF_HARM        | 0.90 — caller indicates a personal safety crisis
OFFTOPIC_GLOBAL  | 0.85 — unrelated to healthcare member services
INTERRUPTION     | 0.80 — topic switch before current collection step completes

## Extraction confidence
Only put a value in extracted{} when the caller stated it directly and
clearly this turn. When the value is garbled, uncertain or partially
unclear → event_type "ambiguous", leave extracted{} empty.
NEVER copy values from the "Confirmed:" context line into extracted{}.
extracted{} may only contain values the caller actually spoke this turn.

## Event type
"answered"  — caller directly and clearly provided the requested value
"answered_with_followup" — caller clearly provided the requested value AND
              also directed a secondary signal at the agent. extracted{}
              must contain the slot value; if no clear value was provided
              this turn, use "answered" or "ambiguous" instead.
              Secondary signals:
                repeat requests       — "can you say that again", "sorry what was that"
                confirmation requests — "did you get that", "is that right"
                side questions the agent cannot answer from session state —
                                        "do you speak Spanish", "what are your hours"
                format uncertainty about their own answer —
                                        "I think it's...", "not sure if that's right"
"ambiguous" — genuinely nothing extractable, garbled, or uncertain — do not guess
"none"      — a guard fired; set when guard != NONE

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

event_type: "answered" | "answered_with_followup" | "ambiguous" | "none"
  answered  — caller directly provided a value for the slot
  answered_with_followup — caller provided a value for the slot AND added a
              secondary signal; extracted{} must hold the slot value
  ambiguous — genuinely nothing extractable, garbled, or uncertain — do not guess
  none      — a guard fired; set extracted: {} and populate guard fields

guard: null when no guard fired; the guard label string when one fires
  e.g. "TRANSFER_REQUEST" | "ABUSE" | "SELF_HARM" | "OFFTOPIC_GLOBAL"
       | "INTERRUPTION"

guard_confidence: 0.0 when no guard fires
  When a guard fires use its threshold value:
  TRANSFER_REQUEST → 0.95, ABUSE → 0.90, SELF_HARM → 0.90,
  OFFTOPIC_GLOBAL → 0.85, INTERRUPTION → 0.80
