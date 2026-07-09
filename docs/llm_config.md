# src/agent/llm/config.py

Purpose
- Configuration and factory logic for LLM adapters/clients.

Key contents
- Definition of available models, provider-specific settings, timeout and retry defaults.
- Initialization code used to create LLM client instances (used during warm-up).

Where used
- app_graph warm-up and any module that constructs or calls LLM clients.

Source
- https://github.com/Swetha-Hariharan1810/new_conversation_agent_swetha_v1/blob/main/src/agent/llm/config.py
