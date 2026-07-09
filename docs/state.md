# src/agent/state.py

Purpose
- Global process-level state for the conversation agent (LLM clients, caches, connection pools, runtime flags).

Key contents
- State objects/containers that hold references to initialized LLM clients and shared resources.
- Lifecycle helpers to create, inspect, and teardown global resources.

Where used
- The app startup lifecycle (app_graph.warm_llm_connections) will populate state.
- Modules needing shared clients import from state to access LLM adapters or caches.

Source
- https://github.com/Swetha-Hariharan1810/new_conversation_agent_swetha_v1/blob/main/src/agent/state.py
