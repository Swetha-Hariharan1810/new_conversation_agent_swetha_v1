# src/agent/llm/redaction.py

Purpose
- Redaction utilities to remove or mask sensitive information before sending to an LLM or before logging.

Key functions
- redact_text(text): returns redacted string and metadata about redacted spans.
- Policy-driven masking based on slot types and sensitivity labels.

Where used
- Used before LLM calls and when storing or emitting logs that may contain PII.

Source
- https://github.com/Swetha-Hariharan1810/new_conversation_agent_swetha_v1/blob/main/src/agent/llm/redaction.py
