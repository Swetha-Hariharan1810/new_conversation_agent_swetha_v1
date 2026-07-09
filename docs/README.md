# Repository documentation (auto-generated)

This docs/ collection summarizes the repository's main runtime modules and services, focusing on the conversation agent implementation under src/agent and the latency benchmark in scripts/.

Included files:
- main.md — top-level entry (lifespan / warm up)
- app_graph.md — LLM/graph warm-up and app wiring
- logger.md, logging-config.md — logging helpers and config
- state.md, utils.md — global state and utilities
- core_* — core agent runtime, request detection, slot manager, guards
- llm_* — LLM configuration, extractors, redaction, response generation
- conversation_context.md — conversation context helpers
- latency_workload.md — benchmark CLI and runner

Source pointers: each module doc links to the module source for quick lookup.
