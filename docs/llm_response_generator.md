# src/agent/llm/response_generator.py

Purpose
- The LLM response orchestration layer: prepares prompts, sends requests to LLM clients, and consolidates replies.

Key responsibilities
- Build prompt templates using conversation context and schema rules.
- Manage request batching, retries, and parse streaming/non-streaming outputs.
- Apply post-processing like redaction and extraction.

Where used
- Invoked by graph nodes that require natural-language generation or completion from an LLM.

Source
- https://github.com/Swetha-Hariharan1810/new_conversation_agent_swetha_v1/blob/main/src/agent/llm/response_generator.py
