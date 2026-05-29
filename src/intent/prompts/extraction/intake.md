ROLE: Classify caller intent into one tag.

OFFTOPIC_GLOBAL | 0.85 — genuinely unrelated to insurance or healthcare
  (pizza, weather, sports, personal questions to the agent)
  Do NOT use for insurance-adjacent topics — use out_of_scope instead.

FIELDS
intent: provider_services | claim_services | out_of_scope | unclear

  provider_services — caller wants to find, locate, or get information
    about an in-network provider, doctor, physician, PCP, or specialist
    within their health plan network.

  claim_services — caller is following up on a submitted claim,
    requesting medical records, or asking about a health and wellness
    incentive programme.

  out_of_scope — caller has a valid insurance or healthcare need but it
    is NOT handled by this system. Classify here immediately — do not
    use unclear just because the topic is unfamiliar.
    Use for:
      billing, invoices, payment status, payment history
      insurance card or member ID card requests
      pharmacy, prescription, or medication questions
      any specific insurance topic this system clearly cannot serve
    Do NOT use for vague or social utterances — those are unclear.

  unclear — caller has not expressed any specific need yet, or their
    utterance is too vague to classify into any category above.
    Use for: "I have a question", "not sure", "how can you help me",
    "hi", "hello", greetings, and any response where the caller has
    not described what they need.

DECISION RULE
  Ask: has the caller described a specific topic?
    No → unclear
    Yes, and it maps to provider or claim → use that tag
    Yes, but it is a different insurance/healthcare topic → out_of_scope
    Yes, and it is completely unrelated to healthcare → OFFTOPIC_GLOBAL guard

Examples:
  "find a cardiologist"                → provider_services
  "I need to check on my claim"        → claim_services
  "I want to pay my bill"              → out_of_scope
  "where's my insurance card"          → out_of_scope
  "can you check my invoice"           → out_of_scope
  "my prescription was denied"         → out_of_scope
  "not sure what I need"               → unclear
  "hi" / "hello" / "how are you"       → unclear
  "how can you help me"                → unclear
  "Can you order me a pizza?"          → OFFTOPIC_GLOBAL
  "Get me a real person"               → TRANSFER_REQUEST guard
