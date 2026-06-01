"""
test_followup_live.py — Live integration tests for FollowUpAgent + ClosureAgent.

These tests call the REAL LLM — no mocks.
They verify that the Phase 2 prompt fix (intent classification + grounding
rules) actually works with the model.

Run with:
    RUN_LIVE_TESTS=1 pytest src/agent/tests/test_followup_live.py -v --tb=short -s

Skip guard: if RUN_LIVE_TESTS is not set, the entire module is skipped.
"""

from __future__ import annotations

import logging
import os
import re

import pytest
from langgraph.graph import END

from agent.agents.closure.agent import ClosureAgent
from agent.agents.follow_up.agent import FollowUpAgent

if not os.getenv("RUN_LIVE_TESTS"):
    pytest.skip(
        "Live tests skipped — set RUN_LIVE_TESTS=1 to run",
        allow_module_level=True,
    )

pytestmark = [pytest.mark.live, pytest.mark.followup]

logger = logging.getLogger(__name__)

# ── Snapshot values present in the fixture (used in grounding assertions) ────
_SNAPSHOT_DOLLAR_VALUES = {"750", "2500", "20", "3000", "7000"}


# ── Helpers ───────────────────────────────────────────────────────────────────


async def run_follow_up_turn(state: dict, user_message: str) -> dict:
    """Append user_message to state and run one FollowUpAgent turn."""
    # NOTE: this test calls real LLM — may take up to 30s
    messages = list(state.get("messages", []))
    messages.append({"role": "user", "content": user_message})
    test_state = {**state, "messages": messages, "is_interrupt": False}

    agent = FollowUpAgent.from_state(test_state)
    result = await agent.execute(test_state)
    return result


async def run_closure_turn(state: dict, follow_up_result: dict) -> dict:
    """Run one ClosureAgent turn given the result from follow_up_agent."""
    # NOTE: this test calls real LLM — may take up to 30s
    merged = {**state, **follow_up_result, "is_interrupt": False}
    agent = ClosureAgent.from_state(merged)
    result = await agent.execute(merged)
    return result


def _extract_last_assistant_text(result: dict) -> str:
    """Return the last assistant/AI message content from result, lowercased."""
    messages = result.get("messages", [])
    if isinstance(messages, dict):
        return messages.get("content", "").lower()
    if isinstance(messages, list):
        for m in reversed(messages):
            if isinstance(m, dict):
                role = m.get("role", "")
                if role in ("assistant", "ai"):
                    return m.get("content", "").lower()
            else:
                role = getattr(m, "type", getattr(m, "role", ""))
                if role in ("assistant", "ai"):
                    return getattr(m, "content", "").lower()
    return ""


# ── Fixture ───────────────────────────────────────────────────────────────────


@pytest.fixture
def completed_call_state() -> dict:
    """Realistic post-call session state for a completed provider services call."""
    return {
        # Identity
        "first_name": "Emily",
        "last_name": "Carter",
        "member_id": "M907503",
        "dob": "1988-04-12",
        "relationship": "plan_holder",
        "member_status_verify": True,
        # Benefits — populated by benefits_agent
        "individual_deductible": "750",
        "family_deductible": "2500",
        "coinsurance_percent": "20",
        "individual_oop_max": "3000",
        "family_oop_max": "7000",
        "benefits_explained": True,
        # Provider search
        "provider_type": "Primary Care Physician",
        "zip_code": "12134",
        "zip_code_used": "12134",
        "provider_list_sent": True,
        "delivery_method": "email",
        "email": "emily.carter@gmail.com",
        # Care coach
        "care_coach_offered": True,
        "care_coach_details_sent": True,
        # Flow
        "call_intent": "provider_services",
        "is_interrupt": False,
        "awaiting_slot": "",
        "slot_attempts": {},
        "ambiguous_counts": {},
        "correction_return_to": "",
        "follow_up_turn_count": 0,
        "follow_up_last_question": "",
        "proactive_offer_available": True,
        "next_node": "",
        "metadata_events": [],
        "app_run_id": "test-live-001",
        "last_agent_signal": {
            "status": "complete",
            "resolved_intents": ["care_wellness"],
            "closure_requested": False,
            "context_updates": {},
            "proactive_offer_available": True,
            "reasoning": "care wellness complete",
        },
        "conversation_context": {
            "caller_first_name": "Emily",
            "confirmed_slots": ["first_name", "last_name", "member_id", "dob"],
            "session_turn_count": 12,
            "agent_turn_count": 2,
            "total_slots_in_pipeline": 4,
            "active_agent_name": "follow_up_agent",
            "llm_recovery_message": "",
        },
        "messages": [
            {
                "role": "assistant",
                "content": "Is there anything else I can help you with today?",
            },
        ],
    }


# ═════════════════════════════════════════════════════════════════════════════
# CLASS 1: TestFollowUpClosure
# ═════════════════════════════════════════════════════════════════════════════


class TestFollowUpClosure:
    """
    Verify the LLM classifies closure phrases as follow_up_intent=DONE
    and that the full closure path produces a goodbye message.

    All phrases are variants NOT in the prompt examples, to test generalisation.
    """

    @pytest.mark.parametrize(
        "user_message",
        [
            # Casual negatives
            pytest.param("nope that's it", id="casual-nope-thats-it"),
            pytest.param("nah i'm good", id="casual-nah-im-good"),
            pytest.param("no i think we're done", id="casual-no-we-are-done"),
            pytest.param("not right now", id="casual-not-right-now"),
            # Completion signals
            pytest.param("i think that covers it", id="completion-covers-it"),
            pytest.param("that's everything i needed", id="completion-everything-needed"),
            pytest.param("we covered everything", id="completion-covered-everything"),
            pytest.param("i got what i needed", id="completion-got-what-needed"),
            pytest.param("that answered it", id="completion-answered-it"),
            # Gratitude variants
            pytest.param("appreciate it", id="gratitude-appreciate-it"),
            pytest.param("much appreciated", id="gratitude-much-appreciated"),
            pytest.param("you've been very helpful", id="gratitude-very-helpful"),
            pytest.param("that was exactly what i needed", id="gratitude-exactly-needed"),
            # Short dismissals
            pytest.param("i'm fine", id="dismissal-im-fine"),
            pytest.param("no worries", id="dismissal-no-worries"),
            pytest.param("that'll do it", id="dismissal-thatll-do-it"),
        ],
    )
    async def test_closure_end_of_call_variants(self, completed_call_state: dict, user_message: str) -> None:
        # ── Step 1: follow_up_agent turn ──────────────────────────────────────
        result = await run_follow_up_turn(completed_call_state, user_message)

        last_signal = result.get("last_agent_signal", {})
        logger.info(
            "Live test turn: input=%r signal=%r",
            user_message,
            last_signal,
        )

        assert result.get("is_interrupt") is False, (
            f"FAILED for input: '{user_message}'\n"
            f"Expected: is_interrupt=False (signal_complete, not ask_member)\n"
            f"Got: is_interrupt={result.get('is_interrupt')!r}\n"
            f"last_agent_signal={last_signal}"
        )

        assert last_signal.get("closure_requested") is True, (
            f"FAILED for input: '{user_message}'\n"
            f"Expected: closure_requested=True\n"
            f"Got: last_agent_signal={last_signal}"
        )

        # ── Step 2: closure_agent turn ────────────────────────────────────────
        closure_result = await run_closure_turn(completed_call_state, result)

        goodbye_text = _extract_last_assistant_text(closure_result)
        logger.info(
            "Live test closure: input=%r goodbye=%r",
            user_message,
            goodbye_text[:200],
        )

        next_node = closure_result.get("next_node")
        assert next_node in (END, "__end__"), (
            f"FAILED for input: '{user_message}'\n"
            f"Expected: next_node=END or '__end__'\n"
            f"Got: next_node={next_node!r}"
        )

        assert "messages" in closure_result, (
            f"FAILED for input: '{user_message}'\n"
            f"Expected: 'messages' key in closure_result\n"
            f"Got keys: {list(closure_result.keys())}"
        )

        goodbye_phrases = [
            "thank you",
            "have a",
            "great day",
            "wonderful day",
            "take care",
            "goodbye",
            "pleasure",
        ]
        assert any(phrase in goodbye_text for phrase in goodbye_phrases), (
            f"FAILED for input: '{user_message}'\n"
            f"Expected goodbye message containing one of {goodbye_phrases}\n"
            f"Got: '{goodbye_text}'"
        )

        assert closure_result.get("is_interrupt") is False, (
            f"FAILED for input: '{user_message}'\n"
            f"Expected: closure_result is_interrupt=False\n"
            f"Got: {closure_result.get('is_interrupt')!r}"
        )

    @pytest.mark.parametrize(
        "user_message",
        [
            pytest.param("remind me of the deductible amount", id="q-deductible-remind"),
            pytest.param("where exactly is the list being sent", id="q-where-sent"),
            pytest.param("how long until i receive it", id="q-how-long"),
            pytest.param("what was the out of pocket again", id="q-oop-again"),
        ],
    )
    async def test_closure_does_not_fire_on_genuine_question(
        self, completed_call_state: dict, user_message: str
    ) -> None:
        """Genuine questions must NOT trigger closure."""
        result = await run_follow_up_turn(completed_call_state, user_message)

        last_signal = result.get("last_agent_signal", {})
        answer_text = _extract_last_assistant_text(result)
        logger.info(
            "Live test (no-closure): input=%r signal=%r answer=%r",
            user_message,
            last_signal,
            answer_text[:200],
        )

        closure_requested = last_signal.get("closure_requested")
        is_interrupt = result.get("is_interrupt")

        # Agent must either answer (is_interrupt=True) or ask for clarification
        # — it must NOT signal closure for a genuine question.
        assert closure_requested is not True or is_interrupt is True, (
            f"FAILED for input: '{user_message}'\n"
            f"Expected: agent answered or nudged (not closure)\n"
            f"Got: closure_requested={closure_requested!r}, "
            f"is_interrupt={is_interrupt!r}\n"
            f"last_agent_signal={last_signal}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# CLASS 2: TestFollowUpAnswering
# ═════════════════════════════════════════════════════════════════════════════


class TestFollowUpAnswering:
    """
    Verify the LLM answers questions from the session snapshot using exact
    snapshot values. No hallucination of values not in the snapshot.
    """

    @pytest.mark.parametrize(
        "user_message,expected_values",
        [
            # Deductible questions (snapshot: individual_deductible=750)
            pytest.param(
                "can you tell me the deductible one more time",
                ["750", "individual"],
                id="deductible-one-more-time",
            ),
            pytest.param(
                "how much do i have to pay before insurance kicks in",
                ["750"],
                id="deductible-before-kicks-in",
            ),
            pytest.param(
                "remind me what the family deductible was",
                ["2500", "family"],
                id="family-deductible-remind",
            ),
            # Out-of-pocket maximum (snapshot: individual_oop_max=3000)
            pytest.param(
                "what's the most i would ever have to pay",
                ["3000"],
                id="oop-most-ever-pay",
            ),
            pytest.param(
                "after what amount does insurance cover everything",
                ["3000"],
                id="oop-cover-everything",
            ),
            # Coinsurance (snapshot: coinsurance_percent=20)
            pytest.param(
                "what percentage do i pay after the deductible",
                ["20"],
                id="coinsurance-percentage",
            ),
            pytest.param(
                "what's my share once the deductible is met",
                ["20"],
                id="coinsurance-share",
            ),
            # Delivery (snapshot: email=emily.carter@gmail.com)
            pytest.param(
                "which email address did you send it to",
                ["emily.carter@gmail.com"],
                id="email-which-address",
            ),
            pytest.param(
                "where should i look for the provider list",
                ["email", "emily.carter"],
                id="provider-list-where-look",
            ),
        ],
    )
    async def test_answerable_questions_from_session(
        self,
        completed_call_state: dict,
        user_message: str,
        expected_values: list[str],
    ) -> None:
        # NOTE: this test calls real LLM — may take up to 30s
        result = await run_follow_up_turn(completed_call_state, user_message)

        answer_text = _extract_last_assistant_text(result)
        logger.info(
            "Live test (answering): input=%r answer=%r",
            user_message,
            answer_text[:200],
        )

        assert result.get("is_interrupt") is True, (
            f"FAILED for input: '{user_message}'\n"
            f"Expected: is_interrupt=True (agent answered and offered more)\n"
            f"Got: is_interrupt={result.get('is_interrupt')!r}\n"
            f"last_agent_signal={result.get('last_agent_signal', {})}"
        )

        for value in expected_values:
            assert value.lower() in answer_text, (
                f"FAILED for input: '{user_message}'\nExpected '{value}' in answer\nGot: '{answer_text}'"
            )


# ═════════════════════════════════════════════════════════════════════════════
# CLASS 3: TestFollowUpGrounding
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestFollowUpGrounding:
    """
    Verify the LLM does NOT hallucinate answers for topics not in the session
    snapshot. This is the most safety-critical class.

    The snapshot has: deductible, coinsurance, OOP max, email, provider type,
    zip code, care coach.
    The snapshot does NOT have: specialist/urgent care copays, pharmacy
    benefits, prior auth, premium, plan tier.
    """

    @pytest.mark.parametrize(
        "user_message",
        [
            # Copay questions (no copay data in snapshot)
            pytest.param("how much is the copay if i see a specialist", id="copay-specialist"),
            pytest.param("what do i pay per visit to my doctor", id="copay-per-visit"),
            pytest.param("is there a copay for urgent care", id="copay-urgent-care"),
            pytest.param("how much will the visit itself cost me", id="copay-visit-cost"),
            # Pharmacy questions
            pytest.param("does my plan cover prescriptions", id="pharmacy-prescriptions"),
            pytest.param("what's the cost for generic medications", id="pharmacy-generic-cost"),
            # Prior auth / referral
            pytest.param("do i need a referral to see a specialist", id="referral-specialist"),
            pytest.param("does my plan require prior approval for procedures", id="prior-auth-procedures"),
            # Premium / plan details
            pytest.param("how much is my monthly premium", id="premium-monthly"),
            pytest.param("what plan tier am i on", id="plan-tier"),
        ],
    )
    async def test_out_of_scope_questions_not_hallucinated(
        self, completed_call_state: dict, user_message: str
    ) -> None:
        # NOTE: this test calls real LLM — may take up to 30s
        result = await run_follow_up_turn(completed_call_state, user_message)

        answer_text = _extract_last_assistant_text(result)
        logger.info(
            "Live test (grounding): input=%r answer=%r",
            user_message,
            answer_text[:200],
        )

        # Assert A — no hallucinated dollar amounts
        dollar_amounts = re.findall(r"\$[\d,]+|\b\d+\s*dollars?\b", answer_text)
        for amount in dollar_amounts:
            digits = re.sub(r"[^\d]", "", amount)
            assert digits in _SNAPSHOT_DOLLAR_VALUES, (
                f"FAILED for input: '{user_message}'\n"
                f"Expected: no hallucinated dollar amounts (snapshot has {_SNAPSHOT_DOLLAR_VALUES})\n"
                f"Got hallucinated amount '{amount}' in: '{answer_text}'"
            )

        # Assert B — agent acknowledged it cannot answer
        cannot_answer_phrases = [
            "don't have",
            "do not have",
            "not available",
            "not covered in",
            "wasn't discussed",
            "was not discussed",
            "contact",
            "representative",
            "unable to",
            "can't provide",
            "cannot provide",
            "i don't have that",
            "not in",
        ]
        assert any(phrase in answer_text for phrase in cannot_answer_phrases), (
            f"FAILED for input: '{user_message}'\n"
            f"Expected: agent acknowledged it cannot answer "
            f"(one of {cannot_answer_phrases})\n"
            f"Got: '{answer_text}'"
        )

        # Assert C — agent still offered to help with something else
        continuation_phrases = [
            "anything else",
            "else i can",
            "help you with",
            "other questions",
            "is there",
        ]
        assert any(phrase in answer_text for phrase in continuation_phrases), (
            f"FAILED for input: '{user_message}'\n"
            f"Expected: agent offered continuation (one of {continuation_phrases})\n"
            f"Got: '{answer_text}'"
        )

    async def test_no_hallucination_after_closure_phrase(self, completed_call_state: dict) -> None:
        """Goodbye message from closure_agent must not contain dollar amounts or medical info."""
        # NOTE: this test calls real LLM — may take up to 30s
        user_message = "i think that covers everything i needed"

        result = await run_follow_up_turn(completed_call_state, user_message)

        # follow_up_agent should signal closure
        assert result.get("last_agent_signal", {}).get("closure_requested") is True, (
            f"FAILED for input: '{user_message}'\n"
            f"Expected: closure_requested=True from follow_up_agent\n"
            f"Got: last_agent_signal={result.get('last_agent_signal', {})}"
        )

        # Run closure_agent — it produces the actual goodbye
        closure_result = await run_closure_turn(completed_call_state, result)

        # Get full message content (not lowercased — we check raw for dollar signs)
        messages = closure_result.get("messages", [])
        if isinstance(messages, dict):
            content = messages.get("content", "")
        else:
            content = ""
            for m in reversed(messages):
                if isinstance(m, dict):
                    role = m.get("role", "")
                    if role in ("assistant", "ai"):
                        content = m.get("content", "")
                        break
                else:
                    role = getattr(m, "type", getattr(m, "role", ""))
                    if role in ("assistant", "ai"):
                        content = getattr(m, "content", "")
                        break

        logger.info(
            "Live test (no-hallucination-closure): input=%r goodbye=%r",
            user_message,
            content[:200],
        )

        dollar_amounts = re.findall(r"\$[\d,]+", content)
        assert len(dollar_amounts) == 0, (
            f"FAILED for input: '{user_message}'\n"
            f"Expected: goodbye message contains no dollar amounts\n"
            f"Got dollar amounts {dollar_amounts} in: '{content}'"
        )

        assert closure_result.get("last_agent_signal", {}).get("closure_requested") is True, (
            f"FAILED for input: '{user_message}'\n"
            f"Expected: closure_result has closure_requested=True\n"
            f"Got: last_agent_signal={closure_result.get('last_agent_signal', {})}"
        )
