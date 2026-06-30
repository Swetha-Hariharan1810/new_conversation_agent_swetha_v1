"""
test_multi_intent_scenarios.py — Section-11 scenarios, END-TO-END (live LLM).

These drive the REAL graph (live Azure OpenAI + Salesforce) and are the live
counterparts of the deterministic tests in tests/golden/test_scenarios_s1_s8.py.
They prove the whole loop, not just the Python guarantees.

Cast: the Section-11 list uses the UAT-007 cast (Daniel Reed / M714598). The live
Salesforce org provisions Emily Carter / M907503 (see preflight), and the
multi-intent BEHAVIOUR under test is cast-independent, so these are authored
against the verified Emily fixture. To run literally as UAT-007, provision a
Daniel Reed / M714598 member and swap the verify prefix.

Run:  pytest -m live tests/live_e2e/test_multi_intent_scenarios.py
Requires the same env + fixtures as the rest of the live suite (Azure + SF).
Tagged ``live`` so the default and golden (CI) runs never collect them.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import tests.live_e2e  # noqa: F401 — ensures src/ is on sys.path
from tests.live_e2e.harness import Expected, Scenario, run_scenario

pytestmark = pytest.mark.live

RESULTS_DIR = Path(__file__).parent / "results"

# Emily Carter (M907503), DOB 1988-04-12 — the preflight-verified fixture.
_PCP_VERIFY = [
    "I need to find a primary care physician in my area.",
    "emily carter",
    "yes correct",  # name_confirmed
    "m nine zero seven five zero three",
    "April twelfth nineteen eighty eight",
    "I'm calling for myself",
]


# S1 — slot answer + invalidating correction (Phase 1 + 3B) ───────────────────
S1 = Scenario(
    name="s1_slot_answer_plus_invalidating_correction",
    flow="pcp",
    retries=1,
    user_turns=_PCP_VERIFY
    + [
        "Primary Care Physician",
        "yes that's correct",  # zip on file confirmed
        "Fax, but I need to update my ZIP code.",  # ── slot answer + invalidating correction
        "one two one three nine",  # provide a (re-resolved) ZIP
        "yes that's correct",  # confirm fax on file
        "no thanks",
        "no thank you",
        "no that's all",
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        # The ZIP-update request is acknowledged (not dropped) in the same turn.
        transcript_contains=[r"zip"],
        final_state={"provider_list_sent": True},
    ),
    notes="S1 canonical UAT-007: ack both; never dispatch on the disputed ZIP.",
)

# S3 — slot answer + fresh in-scope independent (Phase 3C) ─────────────────────
S3 = Scenario(
    name="s3_slot_answer_plus_independent",
    flow="pcp",
    retries=1,
    user_turns=_PCP_VERIFY
    + [
        "A pediatrician — and can you also go over my benefits afterward?",
        "yes that's correct",
        "send it to my fax",
        "yes that's correct",
        "yes please",  # benefits — drained later
        "no thank you",
        "no that's all",
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        transcript_contains=[r"benefit"],  # benefits acknowledged this turn / actioned later
    ),
    notes="S3: benefits parked + acknowledged, drained on a later turn; one decode/turn.",
)

# S4 — transfer injected mid-slot — precedence (Phase 3B) ──────────────────────
S4 = Scenario(
    name="s4_transfer_precedence",
    flow="pcp",
    user_turns=_PCP_VERIFY
    + [
        "Primary Care Physician",
        "yes that's correct",
        "send it to my fax",
        "Yes that's right — actually just transfer me to a human, this is urgent.",
    ],
    expect=Expected(
        completed=True,
        escalated=True,
        transfer_event=True,
        transfer_initiator="Caller",
    ),
    notes="S4: safety/transfer outranks the co-present slot answer; escalate immediately.",
)

# S5 — multi-intent in one breath — precedence + decline (Phase 3C/3D) ─────────
S5 = Scenario(
    name="s5_multi_intent_one_breath",
    flow="pcp",
    retries=1,
    user_turns=_PCP_VERIFY
    + [
        "Primary Care Physician",
        "yes that's correct",
        "Email — but my ZIP is wrong, and send it to a different fax, "
        "and what's my pharmacy copay?",
        "one two one three nine",  # re-resolved ZIP
        "yes that's correct",
        "no thanks",
        "no thank you",
        "no that's all",
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        # Every request gets a spoken outcome; the unsupported one is declined.
        transcript_contains=[r"zip", r"(can'?t|not able|unable|outside)"],
    ),
    notes="S5: precedence orders the intents; pharmacy copay declined, never fabricated.",
)

# S8 — single-intent happy path regression (all phases) ───────────────────────
S8 = Scenario(
    name="s8_single_intent_happy_path",
    flow="pcp",
    user_turns=_PCP_VERIFY
    + [
        "Primary Care Physician",
        "yes that's correct",
        "send it to my fax",
        "yes that's correct",
        "no thanks",
        "no thank you",
        "no that's all",
    ],
    expect=Expected(
        completed=True,
        escalated=False,
        final_state={"provider_list_sent": True},
    ),
    notes="S8: no multi-intent turns — behaviour unchanged, no spurious parking.",
)

# S2 — mid-verification identity correction (Phase 3D) ─────────────────────────
S2 = Scenario(
    name="s2_mid_verification_identity_correction",
    flow="pcp",
    retries=1,
    user_turns=[
        "I need to find a primary care physician in my area.",
        "emily carter",
        "yes correct",
        "m nine zero seven five zero three",
        # DOB answer + identity correction in one breath:
        "April twelfth nineteen eighty eight — wait, my member ID was wrong, "
        "it's m nine zero seven five zero four.",
        "m nine zero seven five zero three",  # re-confirm the correct ID
        "I'm calling for myself",
        "Primary Care Physician",
        "yes that's correct",
        "send it to my fax",
        "yes that's correct",
        "no thanks",
        "no thank you",
        "no that's all",
    ],
    expect=Expected(
        completed=True,
        # The ID correction is acknowledged and re-validated, not silently advanced.
        transcript_contains=[r"member id"],
    ),
    notes="S2: correction acknowledged + rewound within verification (conversation-wide).",
)

_SCENARIOS = [S1, S2, S3, S4, S5, S8]


@pytest.mark.parametrize("scenario", _SCENARIOS, ids=lambda s: s.name)
def test_multi_intent_scenario(scenario):
    result = asyncio.run(run_scenario(scenario, RESULTS_DIR))
    assert result.passed, "\n".join(result.failures)
