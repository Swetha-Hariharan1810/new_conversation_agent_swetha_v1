# src/agent/core/guards.py

Purpose
- Guard predicates and validation logic used to decide whether a given graph node/action should run.

Key responsibilities
- Functions that inspect conversation context and return booleans.
- Access control, precondition checks, and gating logic used by orchestration.

Where used
- Orchestration and node-execution code consult guards to determine valid execution paths.

Source
- https://github.com/Swetha-Hariharan1810/new_conversation_agent_swetha_v1/blob/main/src/agent/core/guards.py
