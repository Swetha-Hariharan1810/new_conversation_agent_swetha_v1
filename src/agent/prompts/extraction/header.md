## Conversation interpretation — evaluate this first
Before classifying any guard or extracting any field:
- Partial, hesitant, malformed, uncertain responses still count as attempts to answer.
- Weak, vague, minimal, or ambiguous responses should default to NONE

## Spelling & NATO
Accept spelled letters and NATO phonetics ("H as in Hotel").

## Spelling-confirmation rule [ANCHOR: SPELL_CONFIRM]
When the caller provides a name then spells it letter-by-letter
(e.g., "Ried, R-e-e-d" or "Thompson, T H O M P S O N"), the spelled
letters are the authoritative source. Extract the name reconstructed
from the spelled letters, NOT the spoken pronunciation.
  "Ried, R-e-e-d"            → last_name=Reed   (spelled letters win)
  "Its Olivia, O L I V I A"  → first_name=Olivia (spelled confirms spoken)
  "Thompson, T H O M P S O N" → last_name=Thompson
  "Jhon, J-o-h-n"            → first_name=John  (spelled letters win)

When the spoken name and the spelling agree, extract the spoken name as-is.
When they disagree, the spelled version is correct — always reconstruct
the name from the letters provided and use that as the extracted value.
Do NOT store raw letters (e.g. "R-e-e-d") in the extracted field —
reconstruct the word ("Reed") and store that.

## GUARDS
TRANSFER_REQUEST | 0.95 — user requests to end the interaction, disconnect, exit, human agent, representative, supervisor, or transfer request
ABUSE | 0.90 — explicit profanity, insults, threats
SELF_HARM | 0.90 — caller indicates a personal safety crisis
OFFTOPIC_GLOBAL | 0.85 — unrelated to healthcare member services
INTERRUPTION | 0.80 — topic switch before current step is completed
NONE | default

## Extraction confidence rule [ANCHOR: CONFIDENCE]
Only put a value in extracted{} or corrections{} when the caller
stated it directly and clearly this turn.

NEVER infer, pad, complete, or add characters the caller did not say.

Extract ALL identity fields mentioned — not just the field being asked for.

Set extracted:{}, corrections:{}, event_type:"ambiguous" when any of:
- Speech sounds garbled / value implausible for the field
- Phrasing is indirect ("I think", "it should be")
- You are inferring from context rather than what was just said
- Value partially matches but one or more characters are uncertain

When in doubt → event_type:"ambiguous".

## EVENT_TYPE
"answered"  — caller directly and clearly answered the awaiting slot.
"corrected" — caller is explicitly changing a value in Confirmed[].
              corrections{} must be non-empty; otherwise use "ambiguous".
              If Confirmed[] is empty, use "answered" instead.
              Exception: a value-less update request sets update_target with
              empty corrections{} — see UPDATE REQUESTS below.
"answered_with_followup" — caller clearly answered the awaiting slot AND also
              asked for a repeat, a read-back, a confirmation, or a side
              question. Extract the value into extracted{} as normal.
"wait"      — caller is asking for time, not answering — see WAIT below.
"ambiguous" — genuinely nothing extractable, garbled, or uncertain — do not guess (see CONFIDENCE anchor above).

## WAIT
"wait" — the caller is asking for time to find or think about the value,
NOT answering and NOT refusing. Examples: "give me a minute",
"hold on, let me grab my card", "one second", "let me check",
"wait", "just a sec", "let me find it".
Set extracted:{}, corrections:{}, event_type:"wait".
Do NOT classify as ambiguous. Do NOT classify as answered.
If the utterance ALSO contains a valid value ("hold on... okay it's M451982"),
extract the value and use event_type:"answered" — the value wins.
"I don't have it / I lost it / never received it" is NOT wait — that is a
cannot-provide statement; leave existing behavior (event_type stays as-is,
Python-side detect_cannot_provide handles it).

## UPDATE REQUESTS
Caller wants to change a previously accepted value. Three shapes:

1. Update WITH new value, no answer to awaiting slot
   ("actually my last name is Smith")
   → corrections:{last_name:"Smith"}, event_type:"corrected"
2. Update WITH new value, PLUS a valid answer to the awaiting slot
   ("it's 90210 — and actually my email is a@b.com")
   → extracted:{zip_code:"90210"}, corrections:{email:"a@b.com"},
     event_type:"answered_with_followup", followup_disposition:"answer_now"
3. Update WITHOUT a value (with or without an answer)
   ("and I need to change my email" / "it's 90210, oh and I need to change my email")
   → update_target:"email"; if awaiting slot answered, extract it and use
     event_type:"answered_with_followup" + disposition "answer_now";
     if not answered, event_type:"corrected" with empty corrections{} and
     update_target set.

update_target / corrections keys MUST be a slot listed in Confirmed:.
Never a LOCKED FIELD. If the caller asks to change something not in
Confirmed: and not a known slot → treat as a follow-up question and
classify disposition per the table below (usually "decline").

## FOLLOWUP DISPOSITION
Applies only when event_type = answered_with_followup.
Set followup_query to the caller's side question (short paraphrase).
Set followup_disposition:
  answer_now — the question is answerable purely from values in Confirmed:
               (or is a request to repeat/read back something already said,
               or is an update request per UPDATE REQUESTS above)
  park       — the question maps to a slot in Pending: or a later stage of
               this same call (e.g. asks about notifications while identity
               is still being verified)
  decline    — anything else: unrelated to this call, requires data the
               system will never collect, or general knowledge
When event_type != answered_with_followup, omit or set "none".

### ANSWERED vs AMBIGUOUS quick examples
| Caller says                        | event_type | Reason                              |
|------------------------------------|------------|-------------------------------------|
| "uh, July twenty-third"            | answered   | Partial but clearly a date attempt  |
| "I think maybe M… something"       | ambiguous  | No extractable value, indirect      |
| "no wait that's wrong"             | ambiguous  | Correction intent, no new value     |
| "actually it's M451982"            | corrected  | Explicit replacement with new value |
| "November 5 1992"                  | answered   | Clear complete value                |
| "It's Jhonny — could you repeat the question?" | answered_with_followup | Valid value + repeat request |
| "give me a minute"                 | wait       | Asking for time, no value           |
| "hold on, let me grab my card"     | wait       | Asking for time, no value           |
| "hold on... okay it's M451982"     | answered   | Value present — the value wins over wait |
| "it's 90210 — what was my member ID again?" (member_id in Confirmed:) | answered_with_followup | disposition "answer_now" — answerable from Confirmed: |
| "it's 90210 — will I get a text about this?" (notifications in Pending:) | answered_with_followup | disposition "park" — maps to a pending slot / later stage |
| "it's 90210 — do you sell car insurance?" | answered_with_followup | disposition "decline" — unrelated to this call |
| "it's 90210 — sorry, say that again?" | answered_with_followup | disposition "answer_now" — repeat request |
| "actually my last name is Smith"   | corrected  | Update shape 1: new value, no answer to awaiting slot |
| "it's 90210 — and actually my email is a@b.com" | answered_with_followup | Update shape 2: answer + corrections{email}, disposition "answer_now" |
| "and I need to change my email"    | corrected  | Update shape 3: no value — corrections{} empty, update_target "email" |

## LOCKED FIELDS
Never put these in corrections{}: member_status_verify, call_intent.
If the caller disputes one of these, return extracted: {}, corrections: {},
event_type: "answered".

## CALLER TYPE DETECTION [ANCHOR: CALLER_TYPE]
Only extract when caller EXPLICITLY states who they are. Never infer.
Add caller_type to extracted{} only on direct statements.
  "I'm a provider"  → provider
  "I'm an employer" / "calling about our group plan"   → employer_group
  "I represent an insurance carrier"                   → other_carrier
  "I am a member"                                      → member
If not explicitly stated → omit caller_type from extracted{}.

## RETURN
Return JSON only — no markdown, no explanation.
{ "extracted": {}, "corrections": {}, "event_type": "answered", "guard": null, "guard_confidence": 0.0, "followup_disposition": "none", "followup_query": null, "update_target": null }
event_type: "answered" | "answered_with_followup" | "corrected" | "ambiguous" | "wait" | "none" — default "answered"
`extracted` — newly provided slot values; `corrections` — replaces a previously accepted slot
`guard` — triggered guard label or null; `guard_confidence` — 0.0 when no guard fires
`followup_disposition` — "answer_now" | "park" | "decline" | "none"; "none" unless event_type is "answered_with_followup"
`followup_query` — the caller's side question, condensed, verbatim-ish; null when no follow-up
`update_target` — slot the caller wants to change when NO new value was given; null otherwise
