# ruff: noqa: E501  — CALLER_TURNS must stay verbatim single-line strings
"""
replay_transcript.py — Replay the caller side of a recorded transcript
through the real compiled LangGraph and print what the agent says back.

This is an OBSERVATION tool, not a test suite: fixed caller input, live
agent output (real LLM calls, real prompts), human eyeballs the result.
No assertions, no pass/fail.

Usage:
    python -m tests.replay.replay_transcript
    python -m tests.replay.replay_transcript --zip-on-file 30308
    python -m tests.replay.replay_transcript --real-sf

Salesforce is mocked by default (--mock-sf); --real-sf hits the real org.
With --zip-on-file 30309 (default) the caller's mid-call correction to
30308 is a GENUINE correction; with --zip-on-file 30308 the same turn is
a same-value restatement — run both with a one-flag change.

A per-turn state log is written to tests/replay/artifacts/ as JSON, even
when the run dies partway (try/finally flush).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

# ZIP on the mocked Salesforce member record. Overridable via --zip-on-file.
# The mock closures read this module global at call time.
ZIP_ON_FILE = "30309"

CALLER_TURNS = [
    "Oh, um… hello there. Could you—could you help me find a list of, uh, the in-network providers? I think it's for my area, yes. Sorry, could you say that again? I want to make sure I'm asking for the right thing… in-network providers, right?",
    "Oh, um, I think I just need a list… a list of doctors or maybe clinics, you know, close to me. My zip code is 30309. I—I want to make sure they're, uh, in-network with my plan. Could you say that one more time, dear? I just want to be sure I'm asking for the right thing.",
    "Oh, my first name? It's, uh… Sophia. Did you need me to spell that for you, dear? Sorry, could you repeat the question just to be sure? Yes, it's Sophia.",
    "Oh, uh… let me see… My last name is Nguyen—that's N-G-U-Y-E-N, just like it sounds, I suppose. And, um, my first name is Sophia. Now, my birthday is January 18th, 1990. I think that's what you need, right? I'm the one on the plan—the plan holder, yes. The member ID, is that the number you want? It's M330784. And, uh, my zip code is 30309. If you need to call me, it's 404-555-7101—did you get that? I also have an email, it's sophia.nguyen@gmail.com. Is there… uh, anything else",
    "Oh, yes, well… I'm sorry, could you say that question again, dear? I just want to make sure I answer it right. I, um… I just moved here to the 30309 area not too long ago—maybe a few months back—and I, uh, haven't really picked out a primary care doctor yet. I'd really like to find someone who's, you know, in my network and, if possible, close to where I live… my apartment. Also, are there any general clinics around here that my plan covers? Maybe some that stay open a bit later, or, um, take walk-ins? Sorry, I'm not sure if I'm explaining that properly. Could you maybe help me find",
    "Oh dear, I'm sorry—I think I might have mixed that up. My zip code is actually 30308, not 30309. Could you repeat that back to me, just to make sure you've got the right one?",
    "Oh, um… email, yes, that would be best for me, I think. My email is sophia.nguyen@gmail.com—did you catch that? Sometimes I can't hear too well, so if you could repeat it back, I'd appreciate it. Now, I just want to double-check… the list you're sending, does it include both doctors and clinics around the 30308 area? Or am I misunderstanding something? I hope I'm asking the right thing.",
    "Yes, yes, that's—oh, that's right, it's sophia.nguyen@gmail.com. Thank you for making sure, dear. And, um… just so I'm clear, the list you're sending—does it have both doctors and clinics close to my zip code? I just want to make sure I'm not missing anything. Could you maybe say that part again for me?",
    "Oh, um, yes, I think that would be very helpful… Could you, uh, maybe explain what I would pay for a regular visit to my primary doctor with my plan? I'm sorry, I always get a little mixed up with things like deductibles and, uh, coinsurance—those words just confuse me sometimes. If you could speak up a bit and go over that slowly, I'd appreciate it.",
    "Oh, um, a health and wellness coach, you said? I'm not quite sure I know what that means… Could you, maybe, explain a bit more about what they do? If you could speak up and just tell me what it involves before you send anything, I'd appreciate it. I want to make sure I understand, you know?",
    "Oh, um, I'm sorry, could you repeat that last part? I'm just trying to understand—so, this coach, is it only for things like eating better, or do they help with, uh, exercise and stress and all that too? I get a bit muddled sometimes. And, uh, is there any extra cost if I talk to this coach with my plan? I just want to make sure I'm not missing anything.",
    "Oh, uh, yes, that would be good, thank you. If you could, um, send me that information about the health and wellness coach, I'd appreciate it. But can you, uh, just say again—there's no extra cost for that, right? I just want to be sure I'm understanding. Sometimes I get things mixed up, so if you could confirm that for me, I'd feel better.",
    "Oh, no, dear, I think that's all I needed right now. Thank you, thank you for being so patient and explaining everything—sometimes it takes me a little while to catch on. I'll, um, I'll keep an eye on my email for that information. Could you just remind me again, it should be coming soon, right? Thank you so much.",
]

SNAPSHOT_FIELDS = [
    "active_agent",
    "awaiting_slot",
    "next_node",
    "call_intent",
    "zip_code",
    "zip_code_used",
    "pending_zip_code",
    "pending_email",
    "pending_fax",
    "pending_phone",
    "provider_type",
    "delivery_method",
    "member_status_verify",
    "provider_list_sent",
    "care_coach_details_sent",
]

SEPARATOR = "-" * 78


def _ensure_src_on_path() -> None:
    """Make `import agent` work without an editable install."""
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))


def _msg_role(msg: Any) -> str:
    return (msg.get("role") if isinstance(msg, dict) else getattr(msg, "type", "")) or ""


def _msg_content(msg: Any) -> str:
    content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
    if not isinstance(content, str):
        content = str(content or "")
    return content.strip()


def _new_assistant_messages(messages: list, start_index: int) -> list[str]:
    """All assistant/ai message contents appended at or after start_index."""
    out: list[str] = []
    for msg in messages[start_index:]:
        if _msg_role(msg) in ("assistant", "ai"):
            content = _msg_content(msg)
            if content:
                out.append(content)
    return out


def _snapshot_fields(values: dict) -> dict:
    # .get() throughout — pending_* fields may not exist on older branches
    return {field: values.get(field) for field in SNAPSHOT_FIELDS}


def mock_salesforce() -> None:
    """
    Patch the @tool objects in agent.storage.tools (NOT the agent-layer
    wrappers — those stay under test) plus the benefits query, by
    reassigning .ainvoke / the module attribute.
    """
    _ensure_src_on_path()
    import agent.storage.queries.benefits as benefits_queries
    from agent.storage import tools

    async def _lookup_member(payload: dict, *_args: Any, **_kwargs: Any) -> dict:
        return {
            "verified": True,
            "member_id": "M330784",
            "phone_number": "4045557101",
            "zip_code": ZIP_ON_FILE,
            "fax": "4045557199",
            "email": "sophia.nguyen@gmail.com",
            "relationship": "plan holder",
        }

    tools.lookup_member.ainvoke = _lookup_member

    for name in (
        "update_zip_code",
        "update_member_contact",
        "dispatch_provider_list",
        "dispatch_care_coach_details",
    ):

        async def _write(payload: dict, *_args: Any, _n: str = name, **_kwargs: Any) -> bool:
            print(f"    [SF WRITE] {_n} {payload}")
            return True

        getattr(tools, name).ainvoke = _write

    async def _get_member_benefits(member_id: str, *_args: Any, **_kwargs: Any) -> dict:
        return {
            "individual_deductible": "1000",
            "family_deductible": "3000",
            "coinsurance_percent": "30",
            "individual_oop_max": "4000",
            "family_oop_max": "9000",
        }

    benefits_queries.get_member_benefits = _get_member_benefits


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay the Sophia transcript through the live graph for manual inspection.",
    )
    parser.add_argument(
        "--zip-on-file",
        default="30309",
        help="ZIP on the mocked member record (default 30309; use 30308 to make "
        "the caller's mid-call ZIP correction a same-value restatement)",
    )
    sf_group = parser.add_mutually_exclusive_group()
    sf_group.add_argument(
        "--mock-sf",
        dest="mock_sf",
        action="store_true",
        default=True,
        help="mock all Salesforce tools and the benefits query (default)",
    )
    sf_group.add_argument(
        "--real-sf",
        dest="mock_sf",
        action="store_false",
        help="disable the Salesforce mock and hit the real org",
    )
    args = parser.parse_args()

    global ZIP_ON_FILE
    ZIP_ON_FILE = args.zip_on_file

    if args.mock_sf:
        mock_salesforce()
        print(f"[setup] Salesforce mocked — ZIP on file: {ZIP_ON_FILE}")
    else:
        print("[setup] Salesforce mock DISABLED — using real org")

    _ensure_src_on_path()
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    from agent.app_graph import build_graph

    graph = build_graph(MemorySaver())
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    print(f"[setup] thread_id: {thread_id}")
    print(SEPARATOR)

    turn_log: list[dict] = []
    artifact_path = ARTIFACTS_DIR / f"sophia_replay_{datetime.now():%Y%m%d_%H%M%S}.json"
    msg_count = 0

    try:
        # The first invoke produces the greeting before any caller input
        await graph.ainvoke({"messages": []}, config)
        values = (await graph.aget_state(config)).values
        messages = list(values.get("messages") or [])
        greeting = "\n".join(_new_assistant_messages(messages, msg_count))
        msg_count = len(messages)
        print(f"AGENT: {greeting}")
        print(SEPARATOR)
        turn_log.append({"turn": 0, "caller": None, "agent": greeting, "state": _snapshot_fields(values)})

        for turn_no, caller_msg in enumerate(CALLER_TURNS, start=1):
            values = (await graph.aget_state(config)).values
            if not values.get("is_interrupt"):
                skipped = list(range(turn_no, len(CALLER_TURNS) + 1))
                print(f"[notice] graph reached END before turn {turn_no} — stopping.")
                print(f"[notice] skipped caller turns: {skipped}")
                break

            print(f"CALLER (turn {turn_no}): {caller_msg}")
            try:
                await graph.ainvoke(Command(resume=caller_msg), config)
            except Exception:
                print(f"[error] graph raised while processing caller turn {turn_no} — aborting replay")
                raise

            values = (await graph.aget_state(config)).values
            messages = list(values.get("messages") or [])
            agent_reply = "\n".join(_new_assistant_messages(messages, msg_count))
            msg_count = len(messages)
            print(f"AGENT: {agent_reply}")
            print(SEPARATOR)
            turn_log.append(
                {
                    "turn": turn_no,
                    "caller": caller_msg,
                    "agent": agent_reply,
                    "state": _snapshot_fields(values),
                }
            )
    finally:
        # Flush whatever we have, even on a mid-run crash
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(turn_log, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[artifact] {artifact_path}")


if __name__ == "__main__":
    asyncio.run(main())
