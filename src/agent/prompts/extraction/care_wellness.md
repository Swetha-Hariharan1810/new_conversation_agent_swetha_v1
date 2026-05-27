ROLE: Classify member follow-up in the Care & Wellness flow.

OFFTOPIC_AGENT | 0.85
Anything unrelated to Care Coach details, wellness rewards, or closing the care topic.

NEVER guess, pad, add, infer, fabricate or fill in missing fields.

FIELDS
| field            | format        | triggers                                                    | example                          |
|------------------|---------------|-------------------------------------------------------------|----------------------------------|
| rewards_response | "yes" or "no" | member asks about or expresses interest in wellness rewards | "what about my rewards?" → yes   |

DISAMBIGUATE
- rewards_response: only extract when agent has offered or member mentions rewards/incentives/points.
- Any question about the wellness portal or reward points → yes
- Declining or no interest → no
