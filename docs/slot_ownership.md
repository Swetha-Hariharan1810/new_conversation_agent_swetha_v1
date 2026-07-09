# src/agent/core/slot_ownership.py

Purpose
- Implements fine-grained logic for which agent or node owns a particular slot and how ownership transfers happen.

Key responsibilities
- Ownership metadata structures
- Policies for preemption, soft/hard claims, and timeouts

Where used
- Works with slot_manager to ensure consistent slot updates across concurrent nodes.

Source
- https://github.com/Swetha-Hariharan1810/new_conversation_agent_swetha_v1/blob/main/src/agent/core/slot_ownership.py
