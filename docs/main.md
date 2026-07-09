# main.py

Purpose
- Application entry helpers used by the ASGI/FastAPI lifecycle.
- Warming LLM connections on application startup so the server only accepts traffic after LLM warm-up completes.

Key items
- lifespan(app): asynccontextmanager that calls warm_llm_connections() before yielding.
- main(): simple local-run entrypoint that prints a greeting when run as a script.

Where to find the source
- main.py: https://github.com/Swetha-Hariharan1810/new_conversation_agent_swetha_v1/blob/08f9ac4e873a1501f04fa095e34f2f33617ca417/main.py
