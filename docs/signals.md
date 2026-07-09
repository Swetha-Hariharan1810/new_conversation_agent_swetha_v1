# src/agent/core/signals.py and src/agent/core/signal.py

Purpose
- signal.py: low-level signal/notification primitives (events, typed signal objects).
- signals.py: higher-level signal definitions and helpers for the runtime.

Key responsibilities
- Define and emit signals used to coordinate slot ownership, state changes, or async events.
- Utilities to subscribe/observe signals and to dispatch them synchronously or asynchronously.

Where used
- Orchestration/slot manager and cross-node coordination code use signals to react to state changes.

Sources
- signal.py: https://github.com/Swetha-Hariharan1810/new_conversation_agent_swetha_v1/blob/main/src/agent/core/signal.py
- signals.py: https://github.com/Swetha-Hariharan1810/new_conversation_agent_swetha_v1/blob/main/src/agent/core/signals.py
