# src/agent/core/agent.py

Purpose
- Core agent runtime abstractions for defining an agent, handling requests, and executing the conversation graph.

Key classes / functions
- Agent class: describes behavior, available actions, and entry points for a conversation.
- Request handling entrypoints: convert external request -> internal graph execution.

Where used
- Agents under src/agent/agents implement domain behaviors and are wired into the runtime via this module.

Source
- https://github.com/Swetha-Hariharan1810/new_conversation_agent_swetha_v1/blob/main/src/agent/core/agent.py
