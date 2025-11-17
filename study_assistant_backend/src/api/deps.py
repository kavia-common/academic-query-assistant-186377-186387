from __future__ import annotations

from functools import lru_cache

from .services.session_store import InMemorySessionStore


# PUBLIC_INTERFACE
def get_session_store() -> InMemorySessionStore:
    """FastAPI dependency returning a process-wide singleton session store.

    Returns:
        A singleton instance of InMemorySessionStore for managing session histories.
    """
    return _get_store_singleton()


@lru_cache(maxsize=1)
def _get_store_singleton() -> InMemorySessionStore:
    """Create or return the singleton store instance."""
    return InMemorySessionStore()
