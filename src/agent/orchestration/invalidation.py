"""
invalidation.py — Deterministic dependency-invalidation registry (zero model cost).

Phase 1 of the context-retention rebuild ("refuse to deliver when a depended-on
value is disputed"). When an upstream value the member depends on is disputed or
changed (e.g. their ZIP code), any downstream artifact derived from it (e.g. the
in-network provider list) is *stale* and must not be acted on until the upstream
value is re-resolved and the artifact is rebuilt.

This module is pure Python: no LLM, no I/O, no state mutation. Agents consume it
to mark/read a small ``dirty_artifacts`` registry on State. The gate that
enforces it (delivery_management) reads ONLY that registry, so the dangerous
outcome is impossible regardless of how any turn is classified.

Extending it:
  * add an upstream→downstream edge to INVALIDATION_MAP, and
  * register the owner agent for the new field/artifact in INTENT_OWNER_REGISTRY.
"""

from __future__ import annotations

# Upstream owner-field → downstream artifact(s) it invalidates when disputed.
INVALIDATION_MAP: dict[str, list[str]] = {
    "zip_code": ["provider_list"],
}

# Each owner field / derived artifact → the agent responsible for (re)resolving it.
# Used to redirect the caller to the right owner when an artifact is stale.
INTENT_OWNER_REGISTRY: dict[str, str] = {
    "zip_code": "provider_search_agent",
    "provider_list": "delivery_management_agent",
}


def artifacts_invalidated_by(field: str) -> list[str]:
    """Return the downstream artifacts invalidated when ``field`` is disputed/changed.

    Pure lookup — returns a fresh list (never the stored one) so callers cannot
    mutate the registry by accident. Unknown fields invalidate nothing.
    """
    return list(INVALIDATION_MAP.get(field, []))


def owner_of(name: str) -> str | None:
    """Return the agent responsible for the given owner field or artifact, if known."""
    return INTENT_OWNER_REGISTRY.get(name)


# ── Helpers over the State.dirty_artifacts registry (dict[str, bool]) ──────────
# Each returns a NEW dict so agents can drop the result straight into an update
# without mutating the live state object.


def mark_dirty(dirty: dict | None, field: str) -> dict:
    """Mark every artifact invalidated by ``field`` as dirty. Returns a new dict."""
    updated = dict(dirty or {})
    for artifact in artifacts_invalidated_by(field):
        updated[artifact] = True
    return updated


def clear_dirty(dirty: dict | None, artifact: str) -> dict:
    """Clear the dirty flag for ``artifact`` (resolved/rebuilt). Returns a new dict."""
    updated = dict(dirty or {})
    updated[artifact] = False
    return updated


def is_dirty(dirty: dict | None, artifact: str) -> bool:
    """True when ``artifact`` is currently marked stale."""
    return bool((dirty or {}).get(artifact))
