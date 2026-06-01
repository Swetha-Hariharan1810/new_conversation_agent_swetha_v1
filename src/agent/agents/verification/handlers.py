"""
handlers.py — Verification workflow handlers. Updated to use pick().
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent.agents.verification.constants import IDENTITY_SLOT_ORDER, MAX_LOOKUP_ATTEMPTS

if TYPE_CHECKING:
    from agent.llm.schema import WorkerResult
from agent.responses.builder import build_initial_prompt
from agent.responses.message_builders import (
    build_offtopic_redirect,
    build_phone_confirmation_prompt,
    build_relationship_confirmation_prompt,
)
from agent.slots.normalizers import (
    normalize_dob,
    normalize_member_id,
    normalize_name,
)
from agent.slots.validators import (
    validate_dob,
    validate_member_id,
    validate_name,
)
from agent.utils import pick

# Escalation messages — delivered at the moment of verification failure handoff
MSG_ESCALATE = [
    (
        "I'm sorry, I wasn't able to verify your account with the details provided. "
        "Let me connect you with a representative who can help verify your identity directly."
    ),
    (
        "Unfortunately I wasn't able to confirm your account after a couple of attempts. "
        "A representative will be able to look into this with you and get things sorted out."
    ),
    (
        "I wasn't able to match the details you provided to an account in our system. "
        "Let me get a representative on the line who can assist you further."
    ),
]
MSG_RESTART = [
    (
        "I wasn't able to find an account with those details — "
        "let's try once more. "
        "Could I start with your first name?"
    ),
    (
        "I wasn't able to match those details to an account. "
        "Let's give it one more try — could I get your first name again?"
    ),
    (
        "Those details didn't quite match what we have on file. "
        "Let's try again — could I start with your first name please?"
    ),
]
MSG_OFFTOPIC_PREFIX = [
    "I'm currently verifying your identity. ",
    "Let me finish verifying your account first. ",
]

# Slots set by the system (Salesforce lookup) or business rules.
# Callers cannot change these values — they must be referred to a human agent.
# Any value the LLM puts in corrections{} for a locked slot is silently dropped
# AND does not trigger a correction acknowledgement.
CALLER_LOCKED_SLOTS: frozenset[str] = frozenset(
    {
        "phone_number",  # from SF record — disputes go to a human
        "zip_code",  # from SF record
        "fax",  # from SF record
        "email",  # from SF record
        "member_status_verify",  # system flag, never caller-stated
        "call_intent",  # classified by intake agent
        # Add domain-agent locked slots here as they are built:
        # "coverage_tier", "plan_type", "benefit_level", etc.
    }
)

_NORMALIZERS = {
    "first_name": normalize_name,
    "last_name": normalize_name,
    "member_id": normalize_member_id,
    "dob": normalize_dob,
}
_VALIDATORS = {
    "first_name": validate_name,
    "last_name": validate_name,
    "member_id": validate_member_id,
    "dob": validate_dob,
}


async def lookup_and_verify(agent, state, collected):
    import asyncio as _asyncio

    # Lazy import — avoids circular import risk at module level.
    from agent.storage.queries.benefits import get_member_benefits

    # Lazy import: agents/verification → storage.tools → storage.db → storage.client;
    # importing at module level would create a agent → storage → agent cycle via
    # lookup_member usage in handlers.py being imported by agent.py at startup.
    from agent.storage.tools import lookup_member

    member_id = collected.get("member_id", "")

    # Run member lookup and benefits fetch concurrently.
    # Benefits failure must never block or fail the verification step —
    # benefits_agent has its own fallback fetch path.
    try:
        result, benefits_record = await _asyncio.gather(
            lookup_member.ainvoke(collected),
            get_member_benefits(member_id),
            return_exceptions=False,
        )
    except Exception:
        # If gather raises (e.g. benefits call throws), fall back to
        # lookup-only so verification is never blocked by a benefits error.
        import logging

        logging.getLogger(__name__).warning(
            "lookup_and_verify: benefits prefetch raised — retrying lookup alone"
        )
        result = await lookup_member.ainvoke(collected)
        benefits_record = None

    verified = result is not None and result.get("verified") is True

    if not verified:
        if escalation := agent.guard_loop_limit(
            state,
            "lookup_fail",
            MAX_LOOKUP_ATTEMPTS,
            escalate_message=pick(MSG_ESCALATE),
            escalate_reason="Verification failed after max lookup attempts",
        ):
            return None, escalation
        # Reset attempt counters so the second round starts with a full budget
        for _slot in ("first_name", "last_name", "member_id", "dob"):
            agent.get_slot(_slot).reset()
        restart = agent.ask_member(state, pick(MSG_RESTART))
        restart.update({"first_name": "", "last_name": "", "member_id": "", "dob": ""})
        restart["verification_restart_index"] = len(state.get("messages") or [])
        return None, restart

    # Merge prefetched benefits into the result dict so _signal_verified()
    # can pass them through context_updates into state, making them available
    # to benefits_agent without a second Salesforce call.
    if benefits_record:
        result["individual_deductible"] = str(benefits_record.get("individual_deductible") or "")
        result["family_deductible"] = str(benefits_record.get("family_deductible") or "")
        result["coinsurance_percent"] = str(benefits_record.get("coinsurance_percent") or "")
        result["individual_oop_max"] = str(benefits_record.get("individual_oop_max") or "")
        result["family_oop_max"] = str(benefits_record.get("family_oop_max") or "")
        import logging

        logging.getLogger(__name__).info(
            "lookup_and_verify: benefits prefetched and merged into verification result"
        )
    else:
        import logging

        logging.getLogger(__name__).warning(
            "lookup_and_verify: benefits prefetch returned None — benefits_agent will fetch on demand"
        )

    return result, None


async def collect_post_lookup(
    agent,
    state,
    messages,
    collected,
    call_intent,
    member_record,
    decision: "WorkerResult | None",
    claims_pipeline,
    provider_pipeline,
):
    post_collected: dict = {}

    if call_intent == "claim_services":
        phone = (member_record or {}).get("phone_number") or state.get("phone_number") or ""
        prompt = (
            build_phone_confirmation_prompt(phone)
            if phone
            else "Thank you. Could you confirm your phone number on file?"
        )
        claims_pipeline.configs["phone_confirmed"].prompt = prompt
        pipeline = claims_pipeline
    else:
        relationship_str = (member_record or {}).get("relationship") or ""
        prompt = build_relationship_confirmation_prompt(relationship_str)
        provider_pipeline.configs["relationship"].prompt = prompt
        pipeline = provider_pipeline

    interrupt = await pipeline.collect(state, messages, post_collected, decision=decision)
    if interrupt:
        interrupt.update(collected)
        # Persist verification status and SF contact fields so subsequent retry turns
        # find member_status_verify=True and skip the Salesforce lookup entirely.
        # Without this, every retry enters the `if not state.get("member_status_verify")`
        # branch and fires another SF HTTP call.
        interrupt["member_status_verify"] = True
        if member_record:
            for field in ("phone_number", "zip_code", "fax", "email", "relationship"):
                if val := member_record.get(field):
                    interrupt[field] = val
        return interrupt

    collected.update(post_collected)
    if "phone_confirmed" in post_collected:
        collected["phone_confirmed"] = True
        collected["phone_update_requested"] = post_collected["phone_confirmed"] == "no"

    return None


def redirect_off_topic(agent, state, collected, identity_pipeline):
    next_slot = next((s for s in IDENTITY_SLOT_ORDER if not collected.get(s)), None)
    if not next_slot:
        # All identity slots are collected — we are in the post-lookup phase
        # awaiting relationship or phone_confirmed. Re-ask that slot directly.
        # For any other awaiting_slot, use the prefix alone — never concatenate
        # a fallback string with the prefix, as they may contain the same text
        # and produce a duplicate message.
        awaiting = state.get("awaiting_slot", "")
        if awaiting == "relationship":
            relationship_str = state.get("relationship", "")
            next_prompt = build_relationship_confirmation_prompt(relationship_str)
        elif awaiting in ("phone_confirmed", "phone_confirmation"):
            phone = state.get("phone_number", "")
            next_prompt = (
                build_phone_confirmation_prompt(phone)
                if phone
                else "Could you confirm your phone number on file?"
            )
        else:
            # No specific re-ask available — prefix alone is a complete sentence
            message = pick(MSG_OFFTOPIC_PREFIX).strip()
            result = agent.ask_member(state, message)
            result.update({k: v for k, v in collected.items() if v})
            return result
    else:
        cfg = identity_pipeline.configs[next_slot]
        # Identity pipeline slots carry slot_type but leave prompt="" because
        # _collect_slot normally drives generation through the response builder.
        # Reading cfg.prompt directly here would return an empty string, producing
        # a prefix-only message. Use build_initial_prompt when slot_type is set.
        if cfg.slot_type:
            next_prompt = build_initial_prompt(cfg.slot_type)
        else:
            next_prompt = cfg.prompt(collected) if callable(cfg.prompt) else cfg.prompt

    prefix = pick(MSG_OFFTOPIC_PREFIX)

    result = agent.ask_member(state, build_offtopic_redirect(next_prompt, prefix=prefix))
    result.update({k: v for k, v in collected.items() if v})
    return result


def apply_corrections(agent, collected, state, decision: "WorkerResult | None"):
    """
    Apply slot corrections from the LLM extraction result.

    Returns the list of slot names that were successfully corrected.
    Locked slots (CALLER_LOCKED_SLOTS) are silently dropped — they do NOT
    appear in the returned list, so no correction acknowledgement fires.
    """
    corrections = (decision.corrections or {}) if decision else {}
    corrected: list[str] = []

    for slot_name, raw in corrections.items():
        # Silently drop any correction targeting a locked slot.
        # This prevents ghost acknowledgements ("Got it — updated zip code")
        # when the system never actually changed the value.
        if slot_name in CALLER_LOCKED_SLOTS:
            continue
        if not raw:
            continue
        norm = _NORMALIZERS.get(slot_name)
        val = _VALIDATORS.get(slot_name)
        if norm and val:
            normalized = norm(str(raw))
            if normalized and val(normalized).valid:
                collected[slot_name] = normalized
                agent.slot_ok(slot_name, normalized)
                corrected.append(slot_name)

    # only cascade-clear if value not provided in same utterance
    extracted_this_turn = (decision.extracted or {}) if decision else {}
    if corrections.get("first_name") and not extracted_this_turn.get("last_name"):
        collected["last_name"] = ""
    if corrections.get("member_id") and not extracted_this_turn.get("dob"):
        collected["dob"] = ""

    return corrected
