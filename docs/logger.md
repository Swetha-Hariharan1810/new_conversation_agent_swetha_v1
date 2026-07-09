# src/agent/logger.py

Purpose
- Centralized logger factory used across the agent codebase to obtain consistent loggers and log formatting.

Key functions
- get_logger(name): returns a configured logger instance (uses logging-config.yaml where relevant).
- Helpers for logging context or structured logging if present.

Where used
- Imported by scripts (e.g., scripts.latency_workload.main) and by many src modules to log runtime events.

Source
- https://github.com/Swetha-Hariharan1810/new_conversation_agent_swetha_v1/blob/main/src/agent/logger.py

Related
- logging-config.yaml provides the config used by the logger factory.
