# Changelog

## Stability hardening: variance matrix + decision provenance

- `reconcile_worker_result` now logs every field it changes with
  `extra={"source": "regex_fallback"|"regex_veto", "field": …,
  "llm_value": …, "final_value": …, "matched": …}` so production variance
  (how often the deterministic layer intervenes, and on which fields) is
  directly measurable.
- New `src/agent/tests/test_extraction_variance_matrix.py` (27 tests) — the
  codified definition of "stable results per run": for each of the five
  production transcripts (BUG-1…BUG-5), every plausible extraction result
  (ideal, dropped fields, regex-only, WAIT and AMBIGUOUS mislabels,
  decline/park misreads) must produce an IDENTICAL final routing signature
  (next_node + awaiting_slot + pending request + message class).
- Prompt regression notes appended to `header_core.md`,
  `delivery_management.md`, `verification_provider.md`,
  `verification_claims.md`, and `follow_up.md`, each documenting the new
  rules and the transcript that motivated them.

## follow_up: parked questions route to the owning agent; closure ordering

Fixes BUG-1: a parked notification/delivery question surfaced in follow_up
was answered by the generation LLM, which hallucinated the channel/address
something was "sent" to; and parked items could leak past an explicit
closure.

- `follow_up/agent.py`: new `_route_parked_question`, consulted right after
  the parked-action routing and BEFORE any LLM answer attempt (including
  the first-entry turn — no opener first). Each parked question is matched
  via `detect_request(query)` plus keyword rules
  (notification/list/delivery/sent/fax/email → replay `provider_list`;
  benefits/deductible/coinsurance/OOP → replay `benefits`); when the
  capability resolves AND the data-exists flag is set
  (`provider_list_sent` / `benefits_explained`), the question converts to
  a routed hop via `route_capability_request(kind="replay", …)`, consuming
  that parked item — the owner answers from real state
  (`_replay_provider_list` / `_replay_benefits`), never from generation.
  Questions with no owning capability (or missing data) stay on the
  grounded LLM path.
- Closure ordering: a bare closure keyword skips parked routing entirely;
  when the classifier returns DONE, follow_up closes immediately
  (`closure_requested=True`) even with parked items — the list is cleared
  and dropped loudly (`warning`, `extra={"dropped_parked": [...]}`). A
  parked question is never answered in the same turn as, or after, closure.
- Grounding: the injected PARKED QUESTIONS block (`follow_up/llm.py`) and
  the Answering sections of `follow_up.md` / `follow_up_claims.md` now
  carry a hard rule — answers may ONLY restate facts present verbatim in
  the session snapshot; never state a destination address, channel, or
  timestamp not in the snapshot ("Do NOT invent which channel or address
  something was sent to"); missing fact → answer=null (existing
  cannot-answer machinery takes over).
- New `src/agent/tests/test_follow_up_parked_routing.py` (12 tests):
  BUG-1 routing across four question phrasings, the delivery second leg
  answering from real state, benefits replay hop, unowned/data-missing
  questions staying on the LLM path, first-entry routing before the
  opener, and closure ordering (bare-keyword skip + LLM DONE both close
  immediately with the dropped-parked warning; no trailing answer).

## benefits: Care-Coach offer honors a delivery redo immediately

Fixes BUG-2: "send that list to my email instead of fax", voiced while the
Care Coach offer is pending, must hand off to delivery_management NOW and
bring the member back to the offer exactly once.

- Registry (`core/slot_ownership.py`): re-sending the provider list IS a
  delivery redo — new `_REDO_TOPIC_EQUIVALENTS` (provider_list → delivery)
  applied inside `resolve_capability` via the new
  `canonical_capability_topic(kind, target)`, so ("redo", "provider_list")
  resolves to the delivery capability. Replay is NOT equivalent (replaying
  provider_list recaps state). `route_capability_request` now records the
  CANONICAL topic in `pending_cross_agent_request`, so delivery's
  `redo_active` re-entry gate fires for list-phrased redos too.
- `delivery_management`: the live-redo pre-branch and `_maybe_switch_method`
  use `canonical_capability_topic("redo", …)` and explicitly exclude
  replay-kind requests (replays recap, never re-send).
- `benefits._handle_care_coach_response`: deterministic Phase-1 reconcile
  right after conversation guards, before the redo/replay hook (which stays
  ahead of the yes/no extraction) — a delivery-phrased redo always yields
  kind redo / target delivery, making the "unknown topic → park" branch
  unreachable for those turns. That branch now logs at WARNING with the
  raw kind/target/utterance for observability.
- benefits' `slot_update_resume` acknowledgement branch already existed —
  covered by the round-trip test rather than re-added.
- Tests: `test_redo_provider_list_resolves_to_delivery` in
  test_cross_agent_requests.py; new `test_benefits_redo.py` — BUG-2 hand-off
  across five extraction variants (ideal, provider_list-phrased, regex-only,
  WAIT mislabel, care-coach-decline misread), never-park regex-only
  phrasings, unknown-topic warning path, yes/no control, and a regex-only
  round trip asserting the resume acks the re-send and re-asks the offer
  exactly once (single "?" in the resume message).

## verification: identity updates mid-collection always honored in-flow

Fixes BUG-4: "m nine zero seven five zero three — oh, also I need to update
my last name" must confirm the captured member_id AND open the last-name
detour now — never park ("in just a moment"), never decline ("a
representative"), regardless of how the extraction LLM labeled the turn.

- `verification/agent.py`: deterministic pre-branch
  `reconcile_worker_result` after conversation guards (same rationale as
  delivery_management — extraction fallbacks and faked results bypass the
  llm.py wiring).
- `_collect_slot` (core): a valid answer accompanied by a value-less update
  request now reaches `_handle_answered_followup` even when the LLM
  flattened the event to ANSWERED/CORRECTED — previously the clean-confirm
  path silently dropped the request. Corrections-with-values and
  redo/replay keep their existing paths.
- `reconcile_worker_result`: new veto — a bare request (update_target set,
  no extracted values, no corrections) labeled ANSWERED upgrades to
  CORRECTED, matching the extraction contract; only the CORRECTED path (C2)
  can honor a target with no value.
- Prompt hardening: `verification_provider.md` and `verification_claims.md`
  gain a "MID-VERIFICATION UPDATE REQUESTS" section with the exact
  transcript example (answer + update_target="last_name",
  request_kind="update", disposition left "none") and an explicit
  never-park / never-decline rule for first_name/last_name/member_id/dob/
  relationship — the system decides disposition.
- New `src/agent/tests/test_verification_identity_update.py` (13 tests) at
  the real-pipeline level (real _NORMALIZERS/_VALIDATORS, only the LLM
  calls faked): the transcript turn across seven extraction variants
  (ideal, dropped target, dropped query, park/decline overridden, ANSWERED
  flattening, CORRECTED mislabel), member_status_verify invalidation, the
  park/decline-never regression guard, bare-request C2 detours (ANSWERED
  and WAIT mislabels), cascade-table checks (last_name update keeps
  first_name; member_id update keeps the dob captured in the same turn),
  and a clean-answer control.

## delivery_management: method switch + update routing in confirmation branches

Fixes BUG-3 (a channel switch during fax/email confirmation was treated as a
failed yes/no and re-asked verbatim) and BUG-5 (a "my ZIP changed" turn
during a confirmation read-back burned retries instead of routing).

- Deterministic pre-branch reconcile: `run()` re-applies
  `reconcile_worker_result` right after conversation guards, so
  `update_target`/`request_kind` are populated even when extraction fell
  back to an empty result (or was faked in tests); the zip-corrections shim
  is unchanged.
- New `_maybe_switch_method`, consulted at the top of the `fax_confirmed`,
  `fax`, `email_confirmed`, and `email` branches. Triggers: an extracted
  `delivery_method` different from the current one; the other channel's
  valid value answering this channel's question (carried through as the new
  pending contact); or a redo/update aimed at the delivery topic — the
  other channel is implied unless the caller named ONLY the current channel
  (same-channel redirect, handled by the existing decline path).
  Pre-dispatch it switches the method, clears the abandoned channel's
  pending value and change-cycle/confirmation counters, and asks the new
  channel's confirmation (or the pending read-back when the value arrived
  in the same utterance), logging `LOG_METHOD_COLLECTED` with
  `switched_from`. Post-dispatch (list sent, no redo in flight) it
  delegates to `_begin_redispatch`.
- Never verbatim-repeat over an unhandled request: both confirmation
  branches' "no clear yes/no" fallbacks first run
  `_reroute_unhandled_request` — a routable foreign update ("my ZIP
  changed") hands off via `_route_slot_update`, a delivery switch goes
  through `_maybe_switch_method`, a same-channel update ("change my fax
  number") takes the decline-equivalent new-value ask. Only genuinely
  unclassifiable turns burn a `slot_fail` retry.
- `delivery_management.md`: new "Channel SWITCH vs same-channel redirect"
  section (switch examples extract `delivery_method` and omit the
  confirmation field; mirrored for fax) and an "Other-slot changes are
  never confirmation answers" rule (ZIP change → `update_target`
  "zip_code", `request_kind` "update", never wait/ambiguous).
- New `src/agent/tests/test_delivery_method_switch.py`: BUG-3 and BUG-5
  classes parametrized over extraction variants (explicit method, LLM redo
  flag, regex-only, WAIT mislabel, decline misread, corrections shim),
  plus counter-reset, post-dispatch redispatch, same-channel redirect, and
  unclassifiable-retry negatives.

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
