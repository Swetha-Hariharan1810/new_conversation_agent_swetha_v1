# Changelog

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
