# src/agent/llm/extractor.py

Purpose
- Post-processing helpers that extract structured data (entities, slots, fields) from LLM responses.

Key functions
- extract_entities(response_text): extracts and normalizes entities for slot filling.
- Validation and fallback heuristics if extraction confidence is low.

Where used
- Called after LLM responses are received to populate conversation slots or structured outputs.

Source
- https://github.com/Swetha-Hariharan1810/new_conversation_agent_swetha_v1/blob/main/src/agent/llm/extractor.py
