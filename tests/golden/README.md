# Golden baseline — context-retention (multi-intent) defect

Phase 0 of the Context Retention rebuild. This suite **locks today's broken
behavior under test** so every later phase is measurable. It is the deterministic
counterpart to `tests/live_e2e/` (which drives the real graph against live Azure
OpenAI + Salesforce).

## What it proves

The defect (pending_action_items, **UAT-007**): *when the member says two things
at once, the VA handles one and silently drops the other.* It is conversation-wide,
so the fixtures pin it at several points, not just at delivery:

| Fixture | Stage | Surface | Status today |
|---|---|---|---|
| `uat_007_multi_intent.json` | delivery confirm | "Fax, but update my ZIP" | **2 known failures** |
| `slot_interrupt_fresh_request.json` | provider_search slot | answer + a benefits question this agent doesn't own | **known failure** |
| `mid_verification_correction.json` | verification slot | answer DOB + correct Member ID in one breath | **known failure** |
| `safety_injected_midflow.json` | delivery confirm | self-harm phrase on a slot turn | **green** (handled — floor) |
| `unsupported_injected_midflow.json` | delivery confirm | out-of-scope question invisible to the schema | **known failure** |

UAT-007's two known failures are asserted explicitly:

- **F1** — the ZIP-update request is never acknowledged (silent drop). *Still open — Phase 3.*
- **F2** — the provider list is dispatched on the disputed ZIP (`94107`). **Closed in Phase 1.**

## Phase 1 — deterministic stale-delivery guard (zero model cost)

`src/agent/orchestration/invalidation.py` adds a pure-Python dependency registry
(`INVALIDATION_MAP`, `INTENT_OWNER_REGISTRY`, `artifacts_invalidated_by`, plus
`mark_dirty`/`clear_dirty`/`is_dirty`) and State gains `dirty_artifacts`.

- **provider_search** marks `provider_list` dirty when the ZIP is disputed
  (decline / invalid) and clears it once a valid ZIP is resolved (`_signal_done`).
- **delivery_management** gates `_proceed_to_dispatch`: if `provider_list` is
  dirty it **refuses to dispatch** and redirects to the ZIP owner. The gate reads
  ONLY `dirty_artifacts` — so delivery on a disputed ZIP is impossible regardless
  of how the turn is classified. No new LLM call (latency unchanged).

The UAT-007 golden assertion #2 is flipped accordingly (no dispatch while
disputed → redirect to `provider_search_agent`). Focused unit tests live in
`test_phase1_stale_delivery.py`. F1 (the silent drop) is intentionally still
open — that is Phase 3.

## Phase 2 — dropped-request metric (observability only)

`src/agent/orchestration/observability.py` adds a deterministic, PII-safe
secondary-request detector (`detect_secondary_request` — a conjunction/clause +
redirect/imperative heuristic, zero model cost) and an `observe_dropped_requests`
decorator on the `delivery_management` and `provider_search` nodes. Per
multi-intent turn it emits one structured `dropped_request` metric event
(`logger.info("multi_intent_turn", extra={"metric": "dropped_request",
"outcome": "actioned"|"parked"|"dropped", ...})`) recording the PII-safe
utterance *shape*, and increments `State.dropped_request_count` only when the
secondary was dropped.

No behavior change: the decorator runs the node unchanged and only adds the
counter field + a log line. The golden harness surfaces the count via
`RunRecord.dropped_request_count`, so `test_uat_007_dropped_request_metric_fires`
shows a **non-zero** count today; Phase 3 will drive it to zero. Unit tests live
in `test_phase2_dropped_metric.py`.

## Phase 3A — TurnPlan schema + resolver, shadow mode (no behavior change)

The core of the rebuild, landed in shadow first:

- **Schema** (`src/agent/llm/schema.py`): `TurnPlan` (+ `SecondaryIntent`,
  `Correction`, `SecondaryIntentType`) — the multi-intent understanding decode,
  generalizing the `FollowUpResult` single-decode pattern. No free-text field.
- **Resolver** (`src/agent/orchestration/resolver.py`): pure Python, no LLM. Given
  a `TurnPlan` + `State` it validates `slot_answer` against the existing
  normalizer+validator, drops secondaries whose `verbatim_span` isn't in the
  utterance or whose `owner` doesn't resolve, rejects unresolved corrections,
  applies precedence (`safety > invalidating_correction > current-step completion
  > parked independents > closure`), enqueues independents, flips
  `dirty_artifacts` via `invalidation.py`, sets a rewind target, and selects one
  speech act from a **closed set** (`re_ask | clarify | correction_ack |
  unsupported_decline | multi_intent_ack | open_redirect`). Low-confidence /
  absent-span / unknown → `clarify`/`open_redirect` (ask, never act). Returns
  `ResolverOutcome(speech_act, state_updates, rewind_target, parked, dirty)`.
- **Shadow** (`src/agent/orchestration/shadow.py`): installed at the shared
  `_collect_slot` chokepoint so the single resolver runs on every slot turn in
  every agent and **only logs** (`turnplan_shadow`). The decoder is pluggable;
  the production default is **off** (no-op, zero cost) until the LLM decode lands
  in 3B. Tests use `heuristic_decoder`, which recovers the dropped multi-intent
  shape deterministically from the raw utterance + WorkerResult.

Tests: `test_phase3a_resolver.py` (exhaustive — precedence, span-drop,
owner-rejection, dirty-flag, speech-act selection) and `test_phase3a_shadow.py`
(the single resolver catches the UAT-007 ZIP request at the delivery chokepoint
and the independent at the provider_search chokepoint, the redirect requests
resolve to an actionable plan, and the live path is byte-for-byte unchanged).

## How it stays deterministic (no secrets, no network)

`driver.py` replaces the two external seams every agent touches:

1. **LLM seam** — `get_extraction_llm()` / `get_follow_up_llm()` /
   `get_generation_llm()` are patched to a `FakeLLM` that *replays the fixture's
   `extraction` block*, one `WorkerResult` per member turn. That replayed
   `extracted` dict is the deterministic stand-in for LLM-1 — and documents
   exactly what the single-intent schema can/can't represent (the defect).
2. **Storage seam** — `dispatch_provider_list` / `update_member_contact` /
   `update_zip_code` are patched to `FakeTool`s that record call args (so we can
   assert "dispatched on the disputed ZIP") and return success.

Agents are invoked as plain callables; the driver merges each returned update
into state with LangGraph's reducer semantics (`messages` appends, everything
else last-write-wins). No compiled graph, checkpointer, or env var is needed.

## Latency probe

Every turn is wrapped in a `time.perf_counter()` probe that prints
`[golden-latency] <id> turn <n> (<agent>) wall_clock=<ms>ms ...`. This is the
seed of the wall-clock bench Phase 3/4 will assert a budget against. Run with
`-s` to see the lines; `test_latency_probe_emits_per_turn_wall_clock` asserts a
number is recorded for every turn.

## Run

```bash
uv run pytest tests/golden -s
```

(`tests/golden` is outside the default `testpaths`, exactly like `tests/live_e2e`,
so it is run explicitly.)

## Phase-flip contract

Assertions describe **current** behavior and are marked inline with
`F1 regressed (good!)` / `F2 regressed (good!)` style messages. When a later
phase fixes a drop, the corresponding assertion is the one to flip — change it
from "side intent dropped" to "side intent acknowledged/actioned", and the
fixture's `known_failures[].baseline_assertion` becomes the new target. The
`expected_behavior_target` field on each turn already records what the fixed
behavior should be.
