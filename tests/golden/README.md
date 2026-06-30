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

## Phase 3B — promote the invalidating-correction path live + closed templates

- **Closed-set templates** (`src/agent/responses/turn_acts.py`): several phrasings
  per speech-act (`re_ask`, `clarify`, `correction_ack`, `unsupported_decline`),
  rotated deterministically by attempt count, filled only with resolver-validated
  values. Zero generative surface.
- **Live promotion** (`_collect_slot`): the understanding decode is now installed
  by default (`shadow.heuristic_decoder`; clearable as a kill-switch). On a slot
  turn where the member answers AND fires an *invalidating correction* (UAT-007:
  "Fax, but I need to update my ZIP code"), the resolver now ACTS: it accepts the
  validated slot answer, marks `provider_list` dirty (Phase 1's gate then forbids
  delivery), sets the rewind target, and emits a templated `correction_ack` that
  acknowledges **both** the fax and the ZIP-update — then routes to
  `provider_search` (awaiting `zip_code`) to re-resolve before delivery. Every
  other resolver outcome stays shadow-only, so single-intent and other flows are
  unchanged (the `ANSWERED_WITH_FOLLOWUP` path is preserved).

**UAT-007 assertion #1 is flipped:** the ZIP request is acknowledged in the same
turn and `dropped_request_count` for that turn is **0** (the Phase 2 metric now
records it as `parked`, not `dropped`). The fixture's `resolved_failures` records
F1 (Phase 3B) and F2 (Phase 1) both closed. Tests: `test_phase3b_live.py`
(templates + live UAT-007 + single-intent regression) plus the flipped
assertions in `test_golden_baseline.py` and `test_phase2_dropped_metric.py`.

## Phase 3C — multi-intent acknowledgement (template-first) + open redirect

- **Templates** (`turn_acts.py`): `render_multi_intent_ack` (keyed on the
  resolver's parked owners, with an optional "rebuilding" phrase) and
  `render_open_redirect`, plus an `OWNER_LABELS` map. Deterministic rotation,
  zero generative surface.
- **Evaluation → keep templates, NO model call.** `test_phase3c.py` proves the
  templates reliably cover every UAT-007 combinatoric at the resolver→render
  level (the fax-redirect phrasings, "send another fax + benefits later", the
  invalidating-ZIP ack, and the out-of-scope decline/redirect). Because coverage
  is reliable, the plan-constrained generation fallback is **not** added; it
  remains the documented escape hatch (fed the validated `TurnPlan` only, never
  the raw utterance) if reliability ever degrades.
- **Live wiring**: `_apply_resolver_outcome` (generalized from 3B) now also acts
  on `multi_intent_ack` (acknowledge + enqueue the parked intent for draining —
  no per-parked-intent fan-out this turn) and on `unsupported_decline` /
  `open_redirect` (a spoken outcome for an unanswerable side-question; never
  acts). The Phase 2 metric counts these as `parked`, not `dropped`.
- **End-to-end** (`run_conversation`, a new multi-agent driver that follows
  `next_node`): the UAT-007 ZIP detour runs delivery → provider_search →
  delivery. Every member turn gets a spoken outcome, the ZIP is re-resolved
  (`update_zip_code(94110)`), and the list is dispatched exactly once on the
  **re-resolved** ZIP (`94110`), never the disputed `94107`; `dropped_request_count`
  is 0 and per-turn latency stays within a deterministic budget.

Note: the UAT-007 fax-redirect turns that arrive on delivery's *inline* yes/no
branches (benefits_response, fax_confirmed) are covered at the resolver+template
level now; wiring those non-`_collect_slot` branches onto the resolver is part of
Phase 3D (roll the live path across every agent).

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
