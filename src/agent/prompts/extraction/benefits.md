ROLE: Extract the member's yes/no response to the Care Coach program offer.

OFFTOPIC_AGENT | 0.85 — anything unrelated to the Care Coach offer response.

FIELDS
  care_coach_response  "yes" | "no"
    Only extract when the agent just offered Care Coach details.
    "yes please" / "sure" / "that sounds interesting" → yes
    "no thanks" / "not right now" → no
    Ambiguous responses ("maybe later") → event_type "ambiguous"

CONFIDENCE NOTES (see header [ANCHOR: CONFIDENCE])
- Only extract when it is unambiguous whether the member is accepting
  or declining the Care Coach offer specifically.
