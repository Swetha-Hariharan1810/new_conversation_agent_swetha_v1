import importlib.resources
import random
import re
from typing import Any

from agent.logger import get_logger

logger = get_logger(__name__)


def read_prompt(file_path: str) -> str:
    try:
        return (
            importlib.resources.files("agent")
            .joinpath("prompts", file_path)
            .read_text(encoding="utf-8")
            .strip()
        )
    except Exception as e:
        logger.warning("Prompt not found: %s — %s", file_path, e)
        return ""


def clean_asr_input(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\b(um+|uh+|er+|hmm+)\b", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def human_join(items: list, *, final: str = "or") -> str:
    items = [i.strip() for i in items if i and i.strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} {final} {items[1]}"
    return f"{', '.join(items[:-1])}, {final} {items[-1]}"


def _last_assistant_msg(messages: list) -> str:
    for m in reversed(messages):
        role = m.get("role") if isinstance(m, dict) else getattr(m, "type", "")
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
        if role in ("assistant", "ai"):
            return (content or "").strip()
    return ""


def _last_user_msg(messages: list) -> str:
    for m in reversed(messages):
        role = m.get("role") if isinstance(m, dict) else getattr(m, "type", "")
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
        if role in ("user", "human"):
            return (content or "").strip()
    return ""


def build_history(messages: list, n: int = None) -> list[str]:
    """Build a compact turn-by-turn history for LLM context."""
    if n is None:
        from agent.core.constants import HISTORY_WINDOW_SIZE

        n = HISTORY_WINDOW_SIZE
    result = []

    for m in messages[-n:]:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "type", "")
        content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")

        normalized_role = str(role).lower()

        if normalized_role in {"user", "human"}:
            speaker = "Caller"
        elif normalized_role in {"assistant", "ai"}:
            speaker = "AI"
        else:
            speaker = normalized_role or "Unknown"

        result.append(f"{speaker}: {content}")

    return result


# Fallback only — primary transfer detection is handled by the LLM guard in guards.py.
_TRANSFER_PATTERNS: dict[str, list[str]] = {
    "explicit_transfer": [
        "transfer me",
        "transfer to",
        "connect me to",
        "put me through",
        "patch me through",
    ],
    "human_agent_request": [
        "human agent",
        "live agent",
        "real agent",
        "actual agent",
        "live person",
        "real person",
        "actual person",
        "human being",
    ],
    "speak_to_someone": [
        "speak to someone",
        "speak to a person",
        "speak to an agent",
        "speak to a human",
        "speak to a representative",
        "speak to a rep",
        "speak to a supervisor",
        "speak to a manager",
        "speak to an operator",
    ],
    "talk_to_someone": [
        "talk to someone",
        "talk to a person",
        "talk to an agent",
        "talk to a human",
        "talk to a representative",
        "talk to a rep",
        "talk to a supervisor",
        "talk to a manager",
        "talk to an operator",
        "get a person",
        "get an agent",
        "get a human",
        "get a supervisor",
        "talk to representative",
        "talk to rep",
    ],
    "end_or_exit": [
        "end call",
        "end the call",
        "hang up",
        "just cancel",
        "cancel this",
    ],
    "frustration_signals": [
        "you're not helping",
        "youre not helping",
        "you are not helping",
        "this isn't working",
        "this is not working",
        "this isnt working",
        "no one is helping",
        "nobody is helping",
    ],
}


def detect_transfer_request(state: Any) -> bool:
    msgs = state.get("messages", []) if isinstance(state, dict) else []
    if not msgs:
        return False
    last = msgs[-1]
    content = (last.get("content", "") if isinstance(last, dict) else getattr(last, "content", "")).lower()
    all_patterns = [p for group in _TRANSFER_PATTERNS.values() for p in group]
    return any(p in content for p in all_patterns)


def build_extraction_prompt(agent_prompt_file: str) -> str:
    """
    System prompt for LLM 1 (get_extraction_llm).
    Combines extraction header + agent-specific rules only.
    No global behavioural rules — extraction LLM does not need them.
    """
    global_prompt = read_prompt("system/global_extraction.md")
    header = read_prompt("extraction/header.md")
    agent = read_prompt(agent_prompt_file)
    return f"{global_prompt}\n\n---\n\n{header}\n\n---\n\n{agent}\n\n"


def build_generation_prompt() -> str:
    """
    System prompt for LLM 2 (response generator) and orchestrator.
    Used by response_generator.py and orchestration.py.
    """
    recovery_prompt = read_prompt("generation/recovery.md")
    global_prompt = read_prompt("system/global_generation.md")
    return f"{global_prompt}\n\n---\n\n{recovery_prompt}"


def build_system_prompt(agent_prompt_file: str) -> str:
    """
    Backward-compatible shim. Routes to the correct builder by path prefix.

    BUG FIX: previously called build_generation_prompt(agent_prompt_file)
    but that function takes no arguments — would raise TypeError at runtime.
    """
    if agent_prompt_file.startswith("generation/"):
        return build_generation_prompt()  # no argument
    return build_extraction_prompt(agent_prompt_file)


# ---------------------------------------------------------------------------
# Humanized message variation helpers (merged from utils_humanize.py)
# ---------------------------------------------------------------------------


def pick(pool) -> str:
    """Randomly select from a pool, or return the string as-is."""
    if isinstance(pool, list):
        return random.choice(pool) if pool else ""
    return pool or ""


def _quick_yes_no(text: str) -> str:
    """
    Fast keyword check for unambiguous yes/no responses.
    Returns 'yes', 'no', or '' (ambiguous — needs LLM).

    Used as fast-path before LLM extraction for any yes/no decision.
    Avoids LLM call for clear responses, improving latency.

    Rules:
    - Word boundary matching to avoid false positives
    - Checks negation before YES patterns
      ("not sure" must not match YES via "sure")
    - Returns '' for anything ambiguous — caller decides via LLM
    """
    t = text.lower().strip()
    if not t:
        return ""

    # Check negation first — blocks false YES matches
    _NEGATION = re.compile(r"\b(not|don't|doesn't|can't|won't|never)\b")
    has_negation = bool(_NEGATION.search(t))

    # Clear YES — only when no leading negation
    if not has_negation:
        _YES = re.compile(
            r"\b(yes|yeah|yep|yup|sure|please|transfer|"
            r"connect|go ahead|ok|okay|absolutely)\b"
        )
        if _YES.search(t):
            return "yes"
        if t.startswith(("yes ", "yeah ", "sure ", "please ")):
            return "yes"

    # Clear NO
    _NO = re.compile(
        r"\b(no|nope|nah|never mind|nevermind|"
        r"continue|stay|keep going|help me)\b"
    )
    if _NO.search(t):
        return "no"
    if t.startswith(("no ", "nope ", "nah ")):
        return "no"

    return ""  # ambiguous — needs LLM


def name_part(source) -> str:
    """
    Return ', FirstName' if a first name is known, else ''.
    source: State dict, ConversationContext dict, or plain name string.
    """
    if isinstance(source, dict):
        name = source.get("caller_first_name") or source.get("first_name") or ""
    elif isinstance(source, str):
        name = source
    else:
        name = ""
    name = (name or "").strip().title()
    return f", {name}" if name else ""
