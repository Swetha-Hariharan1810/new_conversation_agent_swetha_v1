# Partial re-verification (targeted identity re-ask)

When a caller's identity lookup fails, the Verification Agent no longer wipes
every collected field and starts over. If the **Member ID is found** but one
identity field doesn't match, the agent re-asks **only the mismatched field**,
keeps everything else, and re-runs the lookup. This shortens the unhappy path
("the date of birth didn't match — could you confirm it again?") instead of
re-collecting name, Member ID, and date of birth from scratch.

This document records the design decisions, the data shapes, the runtime flow,
and the security trade-off that was chosen.

---

## Phase 0 decisions

| Decision | Choice | Rationale |
|---|---|---|
| **Re-ask message style** | **Field-specific (disclosing)** | Naming the wrong field ("the date of birth didn't match") is the clearest UX — the caller knows exactly what to restate. |
| **Member ID not found** | **Full restart** (`MSG_RESTART`) | If no record exists for the ID, there's nothing to compare against; re-ask everything from the top. |
| **Attempt cap** | **Global** (`guard_loop_limit("lookup_fail", MAX_LOOKUP_ATTEMPTS)`) | Keep the existing single counter across all fields; `MAX_LOOKUP_ATTEMPTS = 2`, then escalate to a human. |
| **Name mismatch → name confirmation** | Reset `name_confirmed` **only** when a name field (first/last) is in the mismatch set | A wrong name means the earlier spelled read-back confirmed the wrong name, so the corrected name must be re-read-back. A DOB-only mismatch leaves `name_confirmed` untouched. |

Multi-field mismatches fall back to a **non-disclosing** generic message
(`MSG_REASK_GENERIC`) so the agent never enumerates every wrong detail at once.

---

## Building blocks (storage layer)

`src/agent/storage/queries/members.py`

- **`get_member_contact(member_id)`** — existing query, reused to fetch a record
  by Member ID alone (no new duplicate query was added).
- **`compare_identity_fields(record, *, first_name, last_name, dob) -> dict[str, bool]`**
  — pure, I/O-free helper. Returns a per-field match map. Both sides are
  normalized with the **same** normalizers the identity pipeline uses
  (`normalize_name` for names; `normalize_dob` + `dob_to_db_format` for dob), so
  equivalent values never read as mismatches:
  - `"JAMES"` vs `"James"` → match
  - `"7/13/1977"` vs `"1977-07-13"` → match
  - An **empty** caller-provided field is treated as "not yet provided"
    (reports `True`), not a mismatch.

---

## Lookup return shape

`src/agent/storage/tools.py :: lookup_member` (additive — `"verified"` is always
present, so callers that only read `verified` are unaffected):

```jsonc
// Full identity match (unchanged contract)
{ "verified": true, "member_id": "...", "phone_number": "...", "zip_code": "...",
  "fax": "...", "email": "...", "relationship": "...", "record": { ... } }

// Failed full match, but the Member ID exists
{ "verified": false, "member_id_found": true,
  "field_matches": { "first_name": true, "last_name": true, "dob": false },
  "record": { ... } }

// Failed full match, no record for that Member ID
{ "verified": false, "member_id_found": false }

// Exception path (defensive)
{ "verified": false }
```

The Member-ID-only fetch (`get_member_contact`) only runs on the **failure**
path, so the happy path makes no extra Salesforce call.

---

## Re-ask flow

`src/agent/agents/verification/handlers.py :: lookup_and_verify`

```
lookup fails (verified == False)
│
├─ guard_loop_limit("lookup_fail", MAX_LOOKUP_ATTEMPTS)   # global cap
│     └─ exhausted → escalate with MSG_ESCALATE
│
├─ member_id_found == False  (or no usable field_matches)
│     └─ _full_restart()  → wipe all four identity fields, MSG_RESTART
│
└─ member_id_found == True  → _partial_reask(mismatched)
      • clear ONLY the mismatched slot values; keep Member ID + matched fields
      • reset attempt counters ONLY for the mismatched slots
      • name_confirmed / name_confirm_attempts reset ONLY if a name field
        mismatched; caller_first_name cleared ONLY if first_name mismatched
      • awaiting_slot ← first mismatched field (identity order)
      • verification_restart_index ← len(messages)   # extractor re-reads turns
      • message: MSG_REASK_DOB | MSG_REASK_LAST_NAME | MSG_REASK_FIRST_NAME
                 (single field) or MSG_REASK_GENERIC (multi-field)
      • LOG_PARTIAL_REASK logs the mismatch set (field NAMES only — no PII)
```

### Run-loop wiring

`src/agent/agents/verification/agent.py`

- `run()` recomputes `awaiting_slot` as `state.get("awaiting_slot") or <first
  empty slot>`. `_partial_reask` sets `awaiting_slot` to the first mismatched
  field so the extractor gets the correct slot context on the re-ask turn (a
  stale pointer would otherwise mislabel it).
- **DOB-only mismatch:** `name_confirmed` is preserved, so the name gate and
  both spelled-name read-back intercepts stay dormant — no read-back, no
  name/Member-ID re-ask. The pipeline skips the still-populated slots and
  collects only DOB; `member_status_verify` is never set on the failure path, so
  the next turn re-runs the lookup → verified.
- **Name mismatch:** `name_confirmed` is reset, so the corrected name is read
  back once more. On confirmation, `_name_confirmed_proceed` finds **no empty
  identity slot** (Member ID + DOB were retained) and proceeds straight to the
  lookup via the shared `_finish_after_identity` helper — it does **not** re-ask
  the already-known Member ID. (This branch only triggers when every identity
  slot is already populated, which never happens in the normal first-time flow,
  so existing name-confirmation behavior is unchanged.)

---

## Security trade-off

**Chosen: field-specific disclosing re-ask messages.**

- **Benefit:** best caller experience — the member is told exactly which single
  detail to correct, avoiding a frustrating full restart.
- **Cost:** a disclosing message reveals *which* field was wrong, which
  marginally helps an attacker who already holds a valid Member ID and is
  probing the remaining fields.

**Mitigations that bound the exposure:**

1. **Global attempt cap** — after `MAX_LOOKUP_ATTEMPTS` (2) failed lookups the
   call escalates to a live representative; an attacker gets very few probes.
2. **Member-ID-not-found does not disclose** — it triggers a full restart, never
   a field-specific hint.
3. **Multi-field mismatches use the non-disclosing** `MSG_REASK_GENERIC`.
4. **Locked Salesforce fields** (`phone_number`, `zip_code`, `fax`, `email`,
   `relationship`, …) are never caller-re-askable; only `first_name`,
   `last_name`, and `dob` participate in the partial re-ask.
5. **Logs carry no PII** — `LOG_PARTIAL_REASK` records only the mismatched field
   *names*; lookup logging stays at `member_tail` (last 4 of the Member ID).

---

## Test coverage

- **Unit** (`src/agent/tests/`):
  - `test_compare_identity_fields.py` — match/mismatch matrix, case/format
    insensitivity, empty-field and `None`-record handling.
  - `test_lookup_member_tool.py` — all three lookup branches + verified-only
    caller invariant (queries layer mocked).
  - `test_partial_reask.py` — `_reask_message` selection and `_partial_reask` /
    `_full_restart` state (matched fields + Member ID preserved, only mismatches
    cleared, `awaiting_slot` points at the first mismatched field).
- **Live E2E** (`tests/live_e2e/scenarios.py`, group B2):
  - `verification_dob_only_mismatch` — DOB wrong; read-back appears once.
  - `verification_last_name_only_mismatch` — last name wrong.
  - `verification_first_name_only_mismatch` — first name wrong (clears cached
    `caller_first_name`).
  - `verification_name_mismatch_bare_no_at_readback` — name re-ask plus a bare
    "no" at the confirmation read-back (name-correction sub-loop).
  - `verification_multi_field_mismatch_generic` — first + last wrong → the
    non-disclosing `MSG_REASK_GENERIC`.
  - `verification_member_id_not_found_restart` — full restart path.
  - `verification_repeated_dob_mismatch_escalates` — global cap → escalation.

  The harness gained `Expected.transcript_count` to assert the spelled-name
  read-back appears exactly the expected number of times.

  Run the whole group (optionally as a stress test with `--repeat`):

  ```bash
  python -m tests.live_e2e.run_live_tests --repeat 10 --only \
  verification_dob_only_mismatch,verification_last_name_only_mismatch,\
  verification_first_name_only_mismatch,verification_name_mismatch_bare_no_at_readback,\
  verification_multi_field_mismatch_generic,verification_member_id_not_found_restart,\
  verification_repeated_dob_mismatch_escalates
  ```
