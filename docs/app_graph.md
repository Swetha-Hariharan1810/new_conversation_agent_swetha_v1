# src/agent/app_graph.py

Purpose
- Responsible for wiring up and warming LLM connections and other graph-level resources required before serving requests.

Responsibilities / public functions (high-level)
- warm_llm_connections(): performs blocking async warm-up of configured LLM adapters/connections.
- Graph initialization helpers that ensure LLM clients, caches, and other node-level resources are created.

Where used
- Called from main.lifespan to ensure system readiness before accepting traffic.
- Refer to main.py for lifecycle usage.

Source
- https://github.com/Swetha-Hariharan1810/new_conversation_agent_swetha_v1/blob/main/src/agent/app_graph.py
