"""
test_followup_rescreen.py — Live E2E: follow-up re-screen through intake.

Covers the behavior added on top of the mid-call intent switch: a fresh intake
intent raised during follow-up that must pass back through the INTAKE node (not
straight to verification) so intake re-applies its front-door screening.

Two paths exercised:

  1. INTAKE_RESCREEN_INTENTS (provider_services) → _reroute_through_intake().
     reset_for_new_intent() runs, but call_intent / pending_intent are cleared
     and reverify_bridge_pending is forced False so intake re-classifies the
     triggering utterance instead of routing straight to verification. The value
     of re-screening shows up when the new provider request names an UNSUPPORTED
     specialty: intake's provider_type_unsupported gate now escalates BEFORE any
     identity is re-collected (test_unsupported_provider_*), whereas a SUPPORTED
     specialty re-classifies cleanly and bridges on to verification
     (test_supported_provider_*).

  2. Appeal / grievance keyword gate. Appeals and grievances are out_of_scope
     topics, but the follow-up classifier has no tag for them and its new_intent
     branch only fires on cross-intent switches — so they surface mid-call as a
     plain `question`. follow_up._is_appeal_or_grievance() catches them by keyword
     and reroutes through intake, whose out_of_scope screening routes the caller
     to the appeals/grievance team and hard-ENDs the call. The gate is keyword
     based, so it fires regardless of the follow-up LLM's classification.

The decisive, LLM-independent fact in the escalate / out_of_scope cases is that
the member is NEVER re-verified: ``member_status_verify`` is cleared and
``first_name`` is None at END. That proves intake's screening caught the request
at the front door, before identity collection — the whole point of routing the
re-screen through the intake node rather than straight to verification.

LIVE: nothing is mocked — every LLM / Salesforce call is real. Requires the
AZURE_OPENAI_* + SF_* env vars and the Salesforce fixtures from preflight.py
(Emily Carter M907503). Marked `live`, so default `pytest` runs skip it. These
tests inspect state right after the follow-up pivot, so they drive the compiled
graph directly with the same tiny local driver test_intent_change.py uses.

NAMING NOTE: staged intent values are intent TAGs ("provider_services" /
"claim_services"), the call_intent / detected_intent vocabulary — not node names.
"""

from __future__ import annotations

import re
import uuid

import pytest

import tests.live_e2e  # noqa: F401 — ensures src/ is on sys.path
from tests.live_e2e.harness import extract_last_ai_message
from tests.live_e2e.preflight import PreflightError, restore_contacts, run_preflight

pytestmark = pytest.mark.live


# ──────────────────────────────────────────────────────────────────────────────
# Scripted turns — reuse the same Emily fixture flow test_intent_change.py uses
# ──────────────────────────────────────────────────────────────────────────────

# Emily Carter M907503 — provider flow all the way through to the follow-up
# "anything else?" prompt. provider_list_sent becomes True, so a subsequent
# provider request qualifies as a fresh intake (prior flow complete).
_EMILY_PROVIDER_TO_FOLLOW_UP = [
    "I need to find a primary care physician in my area.",  # intent
    "emily",
    "carter",
    "yes correct",  # name readback confirmed
    "m nine zero seven five zero three",  # member id
    "April twelfth nineteen eighty eight",  # dob
    "I'm calling for myself",  # relationship
    "Primary Care Physician",  # provider type
    "yes that's correct",  # zip on file confirmed
    "send it to my fax",  # delivery method
    "yes that's correct",  # fax on file confirmed
    "no thanks",  # decline benefits
    "no thank you",  # decline Care Coach → follow-up "anything else?"
]


# ──────────────────────────────────────────────────────────────────────────────
# Minimal live driver — mirrors test_intent_change.py / harness._drive
# ──────────────────────────────────────────────────────────────────────────────


def _new_graph():
    from langgraph.checkpoint.memory import MemorySaver

    from agent.app_graph import build_graph

    graph = build_graph(checkpointer=MemorySaver())
    config = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "tags": ["live_e2e", "followup_rescreen"],
    }
    return graph, config


async def _start(graph, config) -> dict:
    return await graph.ainvoke({}, config=config)


async def _send(graph, config, user: str) -> dict:
    """Resume with one user utterance, then auto-advance through any internal
    (non-interrupt, non-terminal) super-steps until the graph pauses for input
    or reaches END. Returns the state at that pause/END."""
    from langgraph.graph import END
    from langgraph.types import Command

    state = await graph.ainvoke(Command(resume=user), config=config)
    guard = 0
    while not state.get("is_interrupt") and state.get("next_node") not in (END, "END") and guard < 12:
        state = await graph.ainvoke(Command(resume=""), config=config)
        guard += 1
    return state


async def _send_all(graph, config, turns: list[str]) -> dict:
    state: dict = {}
    for turn in turns:
        state = await _send(graph, config, turn)
    return state


def _is_terminal(state: dict) -> bool:
    from langgraph.graph import END

    return state.get("next_node") in (END, "END")


async def _drive_emily_to_follow_up(graph, config) -> dict:
    """Run Emily fully through the provider flow into the follow-up phase and
    assert the preconditions a re-screen pivot relies on."""
    await _start(graph, config)
    state = await _send_all(graph, config, _EMILY_PROVIDER_TO_FOLLOW_UP)
    assert state.get("member_status_verify") is True, "Emily should be verified after the provider flow"
    assert state.get("call_intent") == "provider_services"
    assert state.get("provider_list_sent") is True, "provider flow should have sent the provider list"
    assert (state.get("first_name") or "").lower() == "emily"
    return state


# ──────────────────────────────────────────────────────────────────────────────
# Fixture
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
async def snapshot():
    """Preflight once per module; yield the Salesforce contact snapshot."""
    try:
        snap = await run_preflight(warm=True)
    except PreflightError as exc:
        pytest.fail(f"Preflight failed:\n{exc}", pytrace=False)
    yield snap
    await restore_contacts(snap)


# ──────────────────────────────────────────────────────────────────────────────
# Test A — SUPPORTED provider request in follow-up re-screens THROUGH INTAKE
# ──────────────────────────────────────────────────────────────────────────────


async def test_supported_provider_rescreens_through_intake(snapshot):
    """A new (supported) provider request raised in follow-up is rerouted through
    the INTAKE node — not straight to verification. Intake re-classifies the
    request, re-sets call_intent itself, and emits its own first-name bridge.

    The distinguishing facts vs. the direct-to-verification path:
      * active_agent is intake_agent (intake ran), and
      * pending_intent is empty (the verification-direct reroute would STAGE it).
    Identity is reset (member must re-verify) but no re-verification has happened
    yet — the call is paused on intake's bridge heading into verification.
    """
    graph, config = _new_graph()
    await _drive_emily_to_follow_up(graph, config)

    # Pivot to a brand-new SUPPORTED provider request (dermatologist is one of the
    # five supported specialties) — a fresh provider_services intake intent.
    state = await _send(graph, config, "Actually, I also need to find a dermatologist in my network.")

    # ── Routed through INTAKE, not the verification-direct path ──────────────
    assert state.get("active_agent") == "intake_agent", (
        f"a provider re-screen must run through the intake node, got "
        f"active_agent={state.get('active_agent')!r}"
    )
    assert not state.get("pending_intent"), (
        "the intake re-screen path clears pending_intent; only the "
        "direct-to-verification reroute stages it"
    )
    # ── Identity was reset on the pivot; re-verification has NOT happened yet ─
    assert not state.get("member_status_verify"), "verification flag must be cleared on the pivot"
    assert state.get("first_name") is None, "identity must be cleared on the pivot"
    # ── Intake re-classified and is bridging on to verification ──────────────
    assert state.get("call_intent") == "provider_services", "intake should re-classify provider_services"
    assert state.get("is_interrupt") is True
    assert state.get("next_node") == "verification_agent", (
        f"intake's bridge should route to verification, got {state.get('next_node')!r}"
    )
    ai = extract_last_ai_message(state.get("messages", []))
    assert "first name" in ai.lower(), f"intake should re-ask for the first name, got: {ai!r}"


# ──────────────────────────────────────────────────────────────────────────────
# Test B — UNSUPPORTED provider request escalates at intake BEFORE re-verifying
# ──────────────────────────────────────────────────────────────────────────────


async def test_unsupported_provider_rescreen_escalates_before_reverify(snapshot):
    """The payoff of routing the re-screen through intake: a new provider request
    for an UNSUPPORTED specialty (oncologist) hits intake's provider_type
    unsupported gate and escalates immediately — before any identity is
    re-collected. Previously the request would have gone straight to verification
    and only been rejected after re-collecting identity.
    """
    from agent.agents.intake.constants import PROVIDER_TYPE_UNSUPPORTED_REASON

    graph, config = _new_graph()
    await _drive_emily_to_follow_up(graph, config)

    # Pivot to a brand-new UNSUPPORTED provider request.
    state = await _send(graph, config, "Actually, I need to find an oncologist instead.")

    # ── Escalated to a human via the intake unsupported-type gate ────────────
    assert _is_terminal(state), f"unsupported-type re-screen should end the call, got {state.get('next_node')!r}"
    assert state.get("active_agent") == "escalation_agent", (
        f"unsupported provider type must route through escalation, got "
        f"active_agent={state.get('active_agent')!r}"
    )
    # signal_escalate stashes the unsupported-type message in escalation_pre_message;
    # it survives to END (escalation_agent only appends the reference number).
    pre = state.get("escalation_pre_message") or ""
    assert re.search(r"Pediatricians,\s*Cardiologists,\s*Dermatologists", pre, re.IGNORECASE), (
        f"escalation should carry the unsupported-provider-type message, got: {pre!r}"
    )
    # The reason is staged on the escalating signal before escalation_agent
    # overwrites last_agent_signal — assert via the intake reason constant.
    assert PROVIDER_TYPE_UNSUPPORTED_REASON == "provider_type_unsupported_at_intake"

    # ── DECISIVE: the member was NEVER re-verified — the gate caught the request
    #    at the front door, before identity collection. ───────────────────────
    assert not state.get("member_status_verify"), (
        "member must NOT be re-verified — the unsupported type is rejected before identity collection"
    )
    assert state.get("first_name") is None, "identity must never have been re-collected"

    ai = extract_last_ai_message(state.get("messages", []))
    assert re.search(r"Pediatricians,\s*Cardiologists,\s*Dermatologists", ai, re.IGNORECASE), (
        f"the member should hear the supported-specialty message, got: {ai!r}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test C — appeal keyword in follow-up reroutes through intake → out_of_scope
# ──────────────────────────────────────────────────────────────────────────────


async def test_appeal_in_followup_routes_out_of_scope(snapshot):
    """An appeal raised in follow-up is caught by the keyword gate and rerouted
    through intake, which classifies it out_of_scope and hands the caller to the
    appeals team with a hard END (no transfer event, no re-verification)."""
    from agent.agents.intake.constants import OUT_OF_SCOPE_REASON

    graph, config = _new_graph()
    await _drive_emily_to_follow_up(graph, config)

    # Pivot to an appeal — the keyword gate fires regardless of the follow-up tag.
    state = await _send(graph, config, "Actually, I'd like to appeal a denial on my claim.")

    # ── Hard END via intake out_of_scope routing ─────────────────────────────
    assert _is_terminal(state), f"out_of_scope appeal should hard-END, got {state.get('next_node')!r}"
    assert state.get("active_agent") == "intake_agent", (
        f"the appeal must be handled by intake's out_of_scope path, got "
        f"active_agent={state.get('active_agent')!r}"
    )
    assert state.get("escalation_reason") == OUT_OF_SCOPE_REASON, (
        f"out_of_scope reason should be set, got {state.get('escalation_reason')!r}"
    )
    # "appeal" routes to the appeals team (OUT_OF_SCOPE_KEYWORD_ROUTING).
    ai = extract_last_ai_message(state.get("messages", []))
    assert re.search(r"appeals?\s+team|1-800-555-0105", ai, re.IGNORECASE), (
        f"the member should be routed to the appeals team, got: {ai!r}"
    )

    # ── No re-verification: out_of_scope is decided at the front door ─────────
    assert not state.get("member_status_verify"), "out_of_scope must not re-verify the member"
    assert state.get("first_name") is None, "identity must never have been re-collected"


# ──────────────────────────────────────────────────────────────────────────────
# Test D — grievance keyword also routes out_of_scope (documents routing gap)
# ──────────────────────────────────────────────────────────────────────────────


async def test_grievance_in_followup_routes_out_of_scope(snapshot):
    """The other half of APPEAL_GRIEVANCE_KEYWORDS: a grievance raised in
    follow-up is likewise caught and routed out_of_scope with a hard END.

    NOTE: "grievance" is NOT yet in OUT_OF_SCOPE_KEYWORD_ROUTING, so the message
    falls back to the default support team rather than a dedicated grievance team.
    This test asserts the out_of_scope OUTCOME (the keyword gate + screening),
    not a specific team, so it documents — rather than masks — that routing gap.
    """
    from agent.agents.intake.constants import OUT_OF_SCOPE_REASON

    graph, config = _new_graph()
    await _drive_emily_to_follow_up(graph, config)

    state = await _send(graph, config, "Actually, I want to file a grievance about how my claim was handled.")

    assert _is_terminal(state), f"out_of_scope grievance should hard-END, got {state.get('next_node')!r}"
    assert state.get("active_agent") == "intake_agent", (
        f"the grievance must be handled by intake's out_of_scope path, got "
        f"active_agent={state.get('active_agent')!r}"
    )
    assert state.get("escalation_reason") == OUT_OF_SCOPE_REASON, (
        f"out_of_scope reason should be set, got {state.get('escalation_reason')!r}"
    )
    # No re-verification — caught at the front door.
    assert not state.get("member_status_verify"), "out_of_scope must not re-verify the member"
    assert state.get("first_name") is None, "identity must never have been re-collected"
