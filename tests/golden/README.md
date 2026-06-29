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

- **F1** — the ZIP-update request is never acknowledged (silent drop).
- **F2** — the provider list is dispatched on the disputed ZIP (`94107`).

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
