# Changelog

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
