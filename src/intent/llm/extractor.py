import logging

from agent.utils import build_history

logger = logging.getLogger(__name__)


def build_worker_input(
    system_prompt: str,
    awaiting_slot: str,
    last_agent_message: str,
    last_user_message: str,
    *,
    confirmed_slots: dict | None = None,
    attempt: int = 0,
    recent_messages: list | None = None,
) -> list[dict]:
    """
    Build the message list for LLM 1 (extraction model).

    Parameters
    ----------
    system_prompt:
        The system prompt for the extraction model.
    awaiting_slot:
        The single slot currently being collected.
    last_agent_message:
        The question the agent just asked. Used to build history when
        recent_messages is absent.
    last_user_message:
        The caller's response. Used to build history when recent_messages
        is absent.
    confirmed_slots:
        Dict of slot name → value for all slots already confirmed. Only
        entries with non-empty string values are included in the prompt.
        Omitted entirely when None or empty.
    attempt:
        How many collection attempts have been made for awaiting_slot so far.
    recent_messages:
        Recent conversation turns. Each entry is a dict with "role" and
        "content" keys. Up to the last 6 messages are used.
    """
    # Build conversation history — priority: recent_messages > individual params
    history_block = ""
    if recent_messages:
        history_block = "\n".join(build_history(recent_messages, n=6))
        history_block += "\n"
    elif last_agent_message or last_user_message:
        history_block = f"Agent: {last_agent_message}\nCaller: {last_user_message}\n\n"

    # Build context lines
    context_lines = [
        f"Currently asking for: {awaiting_slot}",
        f"Attempt: {attempt}",
    ]
    if confirmed_slots:
        filled = {k: v for k, v in confirmed_slots.items() if isinstance(v, str) and v}
        if filled:
            confirmed_str = ", ".join(f"{k}={v}" for k, v in filled.items())
            context_lines.append(f"Confirmed: {confirmed_str}")

    user_content = history_block + "\n".join(context_lines)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
