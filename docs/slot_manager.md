# src/agent/core/slot_manager.py

Purpose
- Central manager for conversation "slots" — claimed pieces of context (customer ID, dates, intent details) and their lifecycle.

Key features
- Claim / release slot ownership
- Persistence or ephemeral storage of slot values for conversation lifetime
- Conflict resolution and ownership transfer logic

Important classes / functions
- SlotManager: primary API (methods to get/set/claim slots)
- Serialization helpers to persist or checkpoint conversation slots

Where used
- Used by agent nodes and orchestration to read/write and coordinate shared conversation data.

Source
- https://github.com/Swetha-Hariharan1810/new_conversation_agent_swetha_v1/blob/main/src/agent/core/slot_manager.py
