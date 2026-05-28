import uuid


def new_conversation_id() -> str:
    return str(uuid.uuid4())


def extract_last_ai_message(messages) -> str:
    for m in reversed(messages):
        if hasattr(m, "type") and m.type == "ai":
            return m.content
        if isinstance(m, dict) and m.get("role") == "assistant":
            return m.get("content", "")
    return ""
