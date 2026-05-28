"""
llm.py — LLM calls for FollowUpAgent.

extract_follow_up_decision — guard classification and intent routing via
    structured extraction (WorkerResult). No slots are collected; awaiting_slot=""
    signals to the extraction LLM that nothing is being collected. Guard fields
    and follow_up_intent in WorkerResult.extracted still populate normally.

generate_follow_up_answer — answer generation from the session snapshot.
    The entire session state is serialised into a plain-text context block and
    passed to the generation LLM together with the member's question.
"""

from __future__ import annotations

from agent.llm.extractor import build_worker_input
from agent.llm.schema import WorkerResult
from agent.logger import get_logger
from agent.state import State

logger = get_logger(__name__)

# Sentinel returned by the LLM when it cannot answer from the given context.
_CANNOT_ANSWER = "[CANNOT_ANSWER]"


def _build_session_snapshot(state: State) -> str:
    """
    Serialise all relevant session state into a compact, readable text block.

    Rules:
      - Only include fields that have a non-empty value.
      - Group related fields under a plain-text heading.
      - Use concrete dollar amounts, not variable names.
      - Keep it short — this is LLM context, not a report.
    """
    lines: list[str] = []

    # ── Member identity ───────────────────────────────────────────────────────
    first = (state.get("first_name") or "").strip()
    last = (state.get("last_name") or "").strip()
    name = f"{first} {last}".strip()
    if name:
        lines.append(f"Member name: {name}")

    # ── Benefits ──────────────────────────────────────────────────────────────
    indv_ded = state.get("individual_deductible") or ""
    fam_ded = state.get("family_deductible") or ""
    coins = state.get("coinsurance_percent") or ""
    indv_oop = state.get("individual_oop_max") or ""
    fam_oop = state.get("family_oop_max") or ""

    if any([indv_ded, fam_ded, coins, indv_oop, fam_oop]):
        lines.append("\nBenefits:")
        if indv_ded:
            lines.append(f"  Individual deductible: ${indv_ded} per calendar year")
        if fam_ded:
            lines.append(f"  Family deductible: ${fam_ded}")
        if coins:
            lines.append(f"  Coinsurance: {coins}% after deductible is met")
        if indv_oop:
            lines.append(f"  Individual out-of-pocket maximum: ${indv_oop} per year")
        if fam_oop:
            lines.append(f"  Family out-of-pocket maximum: ${fam_oop}")
        lines.append(
            "  Once the out-of-pocket maximum is reached, the plan pays "
            "100% of covered in-network services for the rest of the year."
        )
        lines.append("  Member is eligible for a free health and wellness coach.")

    # ── Provider search ───────────────────────────────────────────────────────
    provider_type = state.get("provider_type") or ""
    zip_used = state.get("zip_code_used") or state.get("zip_code") or ""
    if provider_type or zip_used:
        lines.append("\nProvider search:")
        if provider_type:
            lines.append(f"  Provider type requested: {provider_type}")
        if zip_used:
            lines.append(f"  ZIP code used: {zip_used}")
        if state.get("provider_list_sent"):
            lines.append("  In-network provider list was sent to the member this call.")

    # ── Delivery ──────────────────────────────────────────────────────────────
    method = (state.get("delivery_method") or "").strip()
    fax = (state.get("fax") or "").strip()
    email = (state.get("email") or "").strip()
    if method == "fax" and fax:
        lines.append(f"\nDelivery: sent by fax to {fax}")
    elif method == "email" and email:
        lines.append(f"\nDelivery: sent by email to {email}")

    # ── Care Coach ────────────────────────────────────────────────────────────
    if state.get("care_coach_details_sent"):
        contact = fax if method == "fax" else email
        lines.append(
            "\nCare Coach details were sent to the member"
            + (f" ({method}: {contact})" if contact else "")
            + " this call."
        )
    elif state.get("care_coach_offered"):
        lines.append("\nCare Coach offer was presented to the member this call.")

    return "\n".join(lines)


async def generate_follow_up_answer(state: State, question: str) -> str:
    """
    Ask the generation LLM to answer `question` using only the session snapshot.

    Returns:
      - A non-empty answer string on success.
      - '' when the LLM says it cannot answer or when an exception occurs.
        The caller (agent.py) handles the empty case.
    """
    session_snapshot = _build_session_snapshot(state)

    if not session_snapshot:
        # Nothing was collected this session — nothing to answer from.
        logger.warning("generate_follow_up_answer: empty session snapshot")
        return ""

    system_prompt = (
        "You are a warm, efficient Sagility Health member services agent.\n"
        "\n"
        "You are answering a member's follow-up question at the end of a call.\n"
        "You have access to a SESSION SNAPSHOT containing everything collected\n"
        "during this call. Answer the question using ONLY information present\n"
        "in the snapshot.\n"
        "\n"
        "Rules:\n"
        "1. Answer directly and concisely — 1 to 4 sentences maximum.\n"
        "2. Use only values from the SESSION SNAPSHOT. Never invent figures.\n"
        "3. Do not add a closing question — the caller appends that separately.\n"
        "4. Do not use bullet points, headers, or markdown.\n"
        "5. If the snapshot does not contain enough information to answer,\n"
        f"   respond with exactly: {_CANNOT_ANSWER}\n"
        "6. Match the tone: warm, reassuring, professional."
    )

    user_content = f"SESSION SNAPSHOT:\n{session_snapshot}\n\nMEMBER QUESTION: {question}"

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        from agent.llm.config import get_generation_llm

        llm = get_generation_llm()
        response = await llm.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content),
            ]
        )
        raw = (response.content or "").strip()

        if not raw or raw == _CANNOT_ANSWER:
            return ""

        return raw

    except Exception:
        logger.exception("generate_follow_up_answer: LLM call failed")
        return ""


async def extract_follow_up_decision(
    llm,
    system_prompt: str,
    *,
    last_agent_message: str,
    last_user_message: str,
    recent_messages: list | None = None,
) -> WorkerResult:
    """
    Run one LLM call for guard classification and intent routing.

    Falls back to an empty WorkerResult on any exception.
    The caller (agent.py) handles the empty case via keyword fallback.
    """
    messages = build_worker_input(
        system_prompt,
        awaiting_slot="",
        last_agent_message=last_agent_message,
        last_user_message=last_user_message,
        recent_messages=recent_messages,
    )
    try:
        result: WorkerResult = await llm.with_structured_output(WorkerResult).ainvoke(messages)
        return result
    except Exception:
        logger.exception("extract_follow_up_decision: LLM extraction failed")
        return WorkerResult()
