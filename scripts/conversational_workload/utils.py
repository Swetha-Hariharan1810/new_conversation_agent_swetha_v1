import uuid


def new_conversation_id() -> str:
    return str(uuid.uuid4())[:8]


def extract_last_ai_message(messages: list) -> str:
    """Return content of the most recent assistant message."""
    for m in reversed(messages):
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", "")
        content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
        if role == "assistant":
            return content.strip() if isinstance(content, str) else str(content).strip()
    return ""
