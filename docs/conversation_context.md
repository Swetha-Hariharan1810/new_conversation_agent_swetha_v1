# src/agent/conversation/context.py

Purpose
- Encapsulates conversation-level context and helper methods for building/reading conversation state.

Key classes / responsibilities
- ConversationContext (or similarly named): holds conversation variables, slots, turn history, and serialization helpers.
- Functions to add/extract messages, references, or metadata used by the graph.

Where used
- Invoked by graph nodes and the slot manager to read/write per-conversation data.

Source
- https://github.com/Swetha-Hariharan1810/new_conversation_agent_swetha_v1/blob/main/src/agent/conversation/context.py
