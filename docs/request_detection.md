# src/agent/core/request_detection.py

Purpose
- Logic to detect and normalize user intents/requests from text input.

Key functions
- Request classification: map raw user text to an internal intent or request object.
- Normalization and confidence scoring for downstream routing.

Where used
- Invoked early in the request pipeline to decide which agent or flow should handle the conversation.

Source
- https://github.com/Swetha-Hariharan1810/new_conversation_agent_swetha_v1/blob/main/src/agent/core/request_detection.py
