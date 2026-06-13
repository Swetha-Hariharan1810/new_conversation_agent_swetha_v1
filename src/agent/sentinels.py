"""
sentinels.py — Routing sentinels shared across agents and the graph.

Our agents store the literal string ``END`` (``END_SENTINEL``) in
``State.next_node`` to request termination. LangGraph's real end object
(``_LANGGRAPH_END == "__end__"``) is only ever returned to the graph runtime
at the routing boundary — the conditional-edge functions in ``app_graph.py``
and ``human_node``'s ``Command.goto``. Keeping this constant in a dependency-free
leaf module avoids circular imports with ``app_graph.py``.
"""

# Value stored in State.next_node by our agents to request termination.
END_SENTINEL = "END"
