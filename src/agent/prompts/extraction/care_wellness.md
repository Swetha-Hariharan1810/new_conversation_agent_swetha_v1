ROLE: Classify member follow-up in the Care & Wellness flow.

OFFTOPIC_AGENT | 0.85 — anything unrelated to Care Coach details,
    wellness rewards, or closing the care topic.

FIELDS
  rewards_response  "yes" | "no"
    Only extract when agent has offered or member mentions
    rewards / incentives / points.
    Any question about the wellness portal or reward points → yes
    Declining or no interest → no
