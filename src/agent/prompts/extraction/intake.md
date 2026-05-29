ROLE: Classify caller intent into one tag.

OFFTOPIC_GLOBAL | 0.85 — anything unrelated to insurance or health-insurance

FIELDS
intent: provider_services | claim_services | unclear

  provider_services — find/locate/get info about an in-network provider,
    doctor, physician, PCP, specialist within their health plan network.

  claim_services — following up on a submitted claim, requesting medical
    records, or asking about a health & wellness incentive.

  unclear — intent is insurance-related but too vague to classify.
    Use when the caller mentions insurance/health plan without enough
    context to pick a workflow. ("I have a question about my insurance")

OFFTOPIC_GLOBAL → truly unrelated topics (pizza, weather, sports)
OFFTOPIC_AGENT  → insurance-related but unsupported: billing, payments,
    coverage details, authorizations, referrals.

Examples:
  "find a cardiologist"         → provider_services
  "Can you order me a pizza?"   → OFFTOPIC_GLOBAL
  "Get me a real person"        → TRANSFER_REQUEST guard
