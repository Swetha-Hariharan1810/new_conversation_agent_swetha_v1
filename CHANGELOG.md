# Changelog

## Deterministic request-detection layer (extraction-stability root cause)

The routing built in Phases 3–6 hinges on the extraction LLM populating
`update_target` / `request_kind` and not mislabeling correction turns; in
production it does so intermittently. This phase adds a pure regex
fallback + veto layer so those detections are deterministic — the LLM stays
primary and regex never overrides a concrete LLM detection with a different
target.

- New `src/agent/core/request_detection.py`: `DetectedRequest`,
  `detect_request()`. Per-slot update patterns are DERIVED from
  `SLOT_OWNERSHIP` keys plus a `SLOT_LABEL_ALIASES` map (dob → "date of
  birth"/"birthday", zip_code → "zip"/"postal code", …), so future registry
  entries get baseline coverage automatically; hand-written patterns only
  for phrasings that don't name the slot ("I moved" → zip_code, "instead of
  fax" → redo delivery). Redo/replay tables map to canonical capability
  topics (`delivery`, `benefits`, `provider_list`). Update beats redo beats
  replay; cannot-provide statements and "when will you update…"
  meta-questions return None. Dependency-light: stdlib + slot_ownership only.
- `reconcile_worker_result(result, last_user)` wired in after every
  `WorkerResult` extraction call (all agent `llm.py` modules): fills a
  missed `update_target`/`request_kind` (logged `source=regex_fallback`)
  and vetoes `event_type=WAIT` on correction turns — downgrades to
  CORRECTED (bare request → C2) or ANSWERED_WITH_FOLLOWUP (value captured
  in the same turn).
- `agent.utils.detect_wait_request` returns False when `detect_request`
  fires — "hold on, new zip" is a correction, not a hold request.
- `slot_manager._handle_answered_followup`: backfills an empty
  `update_target` from `followup_query`/last user message (skipping
  meta-questions about already-promised items) BEFORE disposition routing,
  and documents + enforces the invariant that allow/route detours always
  win over the LLM's park/decline; declining a registry-updatable slot now
  logs a resolution/registry-mismatch warning.
- `_collect_slot` C2 path: a bare CORRECTED turn with empty corrections and
  no `update_target` recovers the target via `detect_request` instead of
  downgrading the caller's request to ANSWERED.
- `header_core.md` WAIT section: explicit carve-out — a wait word followed
  by a correction/change statement is NOT wait; classify the update instead.
- New `src/agent/tests/test_request_detection.py` (112 tests): exhaustive
  positive/negative tables, registry-derivation coverage, reconcile
  semantics, and slot_manager-level variants where a missing
  `update_target` previously broke routing.

## LLM-2 payload hygiene + dialogue routing (Bugs A–D, production transcripts)

A five-phase fix series driven by three production transcripts: the Emily
Carter correction double-ask (Bug A), a notification question declined instead
of parked and update requests blanket-escalated (Bug B), a ZIP change ignored
during fax confirmation and the list dispatched from the disputed ZIP (Bug C),
and raw attempt/pending/confirmed-value internals leaking into the LLM-2
payload (Bug D).

### Phase 1 — LLM-2 payload hygiene (Bug D)

- New `src/agent/llm/redaction.py`: `MASKED_SLOTS` single source of truth,
  `mask_confirmed()` (slot-name masking + value-shape regex second pass),
  `_is_reportable_slot()` (filters counter/flag pseudo-slots). Applied
  centrally inside the payload renderer so no call site can leak.
- `guards._generate_guard_response` sends values only (`last_value` or
  `"confirmed"`) for confirmed, reportable slots — never the attempt dicts.
- LLM-2 payload: `Pending:` line removed entirely (the `pending_slots` kwarg
  is gone from `generate_recovery_message`; the list still flows through
  Python for next-ask selection). `Attempt: N` replaced by a coarse
  `Tone: first ask / gentle retry / patient retry` hint;
  `recovery_base.md` keys off the labels.
- Post-capture FOLLOWUP guards render
  `Collecting: (nothing — this turn's value was captured; …)` instead of the
  slot label; `followup_*.md` updated to match.

### Phase 2 — Single-ask invariant (Bug A)

- `sanitize_generated()` in `response_generator.py`: strips sentences that
  re-ask a confirmed slot (fuzzy label + `SLOT_ASK_SYNONYMS` match); when
  Python appends the next static ask, also strips next-slot mentions and
  trailing questions so the combined utterance has exactly one ask; falls
  back to the guard's `_FALLBACKS` entry when emptied; every strip logged.
- Wired at both append points (`_generate_slot_retry_response`,
  `_generate_correction_ack`) and on all plain FOLLOWUP/RETRY paths.
- New `CORRECTION_ACK` guard + `events/correction_ack.md`: pure corrections
  (applied, no side question) no longer fall into the FOLLOWUP_DECLINE
  default that told callers "I can't help with that" about their own
  correction.
- `followup_answer/park/decline.md` and `correction.md` hardened with a
  no-trailing-question rule and an Emily-Carter negative example each.

### Phase 3 — Follow-up disposition correctness + structured parking (Bug B)

- `header.md`: park is the explicit winner for delivery/notification/timeline
  questions ("choose park — never decline") with a disposition example table;
  answer_now guidance for questions the current stage already answers.
- `parked_followups` entries are structured
  `{query, kind: question|action, target}`;
  `state.normalize_parked_followups()` coerces legacy plain strings at every
  read site.
- `follow_up_agent` honors parks: `kind="question"` items keep the LLM answer
  path; `kind="action"` items route via the slot ownership registry —
  `MSG_UPDATE_REQUEST_ESCALATE` only fires for human-only slots.

### Phase 4 — Slot ownership + ZIP update routing (Bug C)

- `src/agent/core/slot_ownership.py`: `SlotOwnership(agent, updatable:
  in_flow|route_to_owner|human_only, invalidates)` registry. zip_code →
  provider_search (route_to_owner, invalidates the provider list); fax/email →
  delivery_management (in_flow); identity slots → verification (in_flow);
  phone_number / member_status_verify / call_intent → human_only.
- `CALLER_LOCKED_SLOTS` scoped to the truly human-only fields; zip/fax/email
  corrections during verification now park as `kind="action"` (or route)
  instead of being silently dropped — and are never ghost-acknowledged.
- `resolve_update_target()` (allow / route / decline) replaces the old bool;
  "route" hands off NOW via `pending_slot_update` (never "later"), the
  orchestrator fast-path returns to `return_to_agent` at `return_awaiting`
  and arms a one-shot `slot_update_resume` acknowledgement.
- Dispatch precondition: `_proceed_to_dispatch` routes any pending/parked
  zip-invalidating update before sending — a list built from a disputed ZIP
  is never dispatched.
- Repeated-ignored-request guard (`ignored_request_<target>`, max 2) on the
  update-decline paths and the OFFTOPIC_AGENT guard: the second identical
  deflected request escalates honestly instead of repeating the same re-ask.
- Meta-questions about promised items ("when will you update my zip?") are
  answered concretely via FOLLOWUP_ANSWER with the promise in context.

### Phase 5 — End-to-end scenarios + eval sweep

- Three live E2E scenarios mirroring the production transcripts:
  `emily_carter_correction_single_ask` (O-1),
  `notification_followup_not_declined` (O-2),
  `zip_update_during_fax_confirmation` (O-3, mutating, run-df1e16a9).
- Regression sweep: zero remaining callers pass `pending_slots` into LLM-2,
  reference the removed `_update_target_allowed`, or write plain-string
  `parked_followups` entries.
- Unit coverage lives in `src/agent/tests/` (payload hygiene, sanitizer +
  CORRECTION_ACK, disposition/parking, ownership routing round-trip).

### Phase 6 — Capability registry: redo/replay cross-agent requests

- `CAPABILITY_REGISTRY` in `core/slot_ownership.py`, keyed by
  `(kind, topic)`: `("redo","delivery")` → delivery_management (re-dispatch
  the provider list with a new method/destination), `("replay","benefits")` →
  benefits (re-explain), `("replay","provider_list")` → delivery_management
  (re-state what was sent, where, and the window). `resolve_capability` +
  `capability_topic` canonicalize extraction-level targets
  (`delivery_method` → `delivery`); unknown topics resolve to None and park
  as questions — never a hard decline.
- `pending_slot_update` generalized to `pending_cross_agent_request`
  (`{"kind": update|redo|replay, "target", "return_to_agent",
  "return_awaiting"}`). `normalize_cross_agent_request()` in `state.py` is
  the single read path; legacy checkpoints fall back with `kind="update"`.
  fast_path/orchestrator return-hop mechanics unchanged, except the resume
  flag is only armed when a slot is actually restored.
- Extraction: `request_kind` on `WorkerResult` (+ `request_kind`/
  `request_target` on `FollowUpResult`); the UPDATE REQUESTS prompt sections
  became CROSS-CALL REQUESTS with redo/replay shapes and few-shot rows
  (header.md, header_extraction.md, benefits.md, follow_up*.md).
- Re-entry contracts: delivery_management serves a pending redo before its
  completed-flow early exit (re-collects the method, re-dispatches, announces
  the re-send, never repeats the benefits offer) and replays the
  provider-list summary from state; benefits_agent replays the benefits
  summary before its care_coach_offered exit without re-offering the Care
  Coach, and acknowledges completed redos on resume. Shared helpers:
  `BaseAgent.consume_cross_agent_request(state, kinds, targets)` +
  `SlotManagerMixin.route_capability_request(...)`.
- follow_up_agent routes live redo/replay requests and parked delivery
  actions through the capability registry; UPDATE_REQUEST escalation now
  applies only to human-only targets (routable slots reroute to their
  owning flow). Requests whose owner IS the active agent resolve in-flow —
  zero routing.
- Live E2E scenarios P-1…P-5 (redo from benefits round-trip, replay from
  the post-flow stage, both in-flow variants, unknown-topic park);
  `follow_up_update_request` retargeted to the phone number (the one
  human-only case that still escalates). Unit coverage in
  `src/agent/tests/test_cross_agent_requests.py`.

## Context-Retention rebuild — one grounded voice + multi-intent (Issues 1 & 2)

A five-phase, flag-gated rebuild that (1) unifies every turn onto one grounded
generator voice and (2) stops silently dropping a second request bundled into a
single utterance. Decision (resolver + validators, deterministic) and phrasing
(the generator) are kept separate, so the free-flowing surface can never move
faster than the deterministic safety layer allows.

### Feature flags (central: `src/agent/core/flags.py`)

| Flag | Values | OLD default | Owns |
|---|---|---|---|
| `UNIFIED_VOICE` | bool | `false` | one generated voice for asks/transitions/retries/clarifies/corrections |
| `TURNPLAN_DECODE` | `off`\|`shadow`\|`live` | `off` | the LLM TurnPlan understanding decode |
| `MULTI_INTENT_LIVE` | bool | `false` | acting on multi-intent turns, narrated by the generator |
| `STREAM_GENERATION` | bool | `false` | stream the generator to first token |
| `PARK_ANSWERABLE` | bool | `false` | park an answerable follow-up instead of answering it inline |

At the OLD defaults the system behaves exactly as before (all flags off), so
merging the code changes nothing until a flag is turned on.

### What each phase added

- **Phase 0** — `flags.py`, the baseline dashboard
  (`scripts/baseline_dashboard.py`), and characterization tests locking today's
  behavior. Zero output change.
- **Phase 1** (`UNIFIED_VOICE`) — one persona/voice
  (`prompts/system/global_generation.md`), a positive, structured recovery prompt
  (`prompts/generation/recovery.md`), and routing the happy-path ask/transition
  through the same generator. Templates demoted to fallback. Grounding + no-false-
  accept guardrails (`src/agent/responses/grounding.py`).
- **Phase 2** (`TURNPLAN_DECODE=shadow`) — the real LLM `TurnPlan` decoder
  (`src/agent/llm/turnplan_decoder.py`), run log-only alongside the unchanged live
  path. Fast-path skips the decode on single-intent turns. Snapshot generalized to
  `src/agent/llm/snapshot.py`. Fallback chain: try → LLM → heuristic → None.
- **Phase 3** (`MULTI_INTENT_LIVE`, needs `TURNPLAN_DECODE=live`) — the resolver
  outcome is narrated by the generator as ONE sentence (accept → answer inline →
  park → decline → ask next). In-scope answerable follow-ups answered inline from
  the snapshot (or parked when `PARK_ANSWERABLE=true`); out-of-scope asides
  declined in the same sentence; invalidating corrections ack + mark dirty +
  rewind. No fact introduced beyond `accepted ∪ answered_inline`.
- **Phase 4** (`STREAM_GENERATION` + guards) — stream to first token with a
  mid-flight fallback to the template; a belt-and-suspenders grounding guard on
  EVERY generated turn (any member-id/ZIP/phone/date/email-shaped token not in
  `confirmed_slots ∪ validated_answer` → fall back to the template for that act);
  one-call discipline; latency percentiles in the dashboard.

### Rollout order

1. `UNIFIED_VOICE=true`
2. `TURNPLAN_DECODE=shadow` → validate the shadow logs (LLM TurnPlan agrees with
   GPT on single-intent turns; recovers the bundled secondary)
3. `TURNPLAN_DECODE=live` + `MULTI_INTENT_LIVE=true`
4. `STREAM_GENERATION=true` + Phase-4 guards

### Rollback

Any regression: flip the one owning flag back to its OLD default (table above).
The flags are independent, so a rollback is surgical — e.g. drop streaming with
`STREAM_GENERATION=false` while keeping multi-intent live.

Keep the regex `heuristic_decoder` and the `turn_acts` template renderers in
place as fallbacks — do NOT delete them until Phase 4 is validated in production.
They are the deterministic floor the generator falls back to on decode failure,
stream error, or a grounding-guard violation.
