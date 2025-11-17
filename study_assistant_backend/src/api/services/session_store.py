from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional, TypedDict


class StoredMessage(TypedDict):
    """Internal representation for a stored message in history."""
    role: str  # "user" or "assistant" (or "system" if extended later)
    content: str
    timestamp: float


class SessionData:
    """Holds per-session chat history and metadata."""
    def __init__(self) -> None:
        self.history: List[StoredMessage] = []
        self.created_at: float = time.time()
        self.updated_at: float = self.created_at


class InMemorySessionStore:
    """Thread-safe in-memory store for session chat histories."""

    def __init__(self) -> None:
        # Mapping of session_id to SessionData
        self._sessions: Dict[str, SessionData] = {}
        self._lock = threading.RLock()

    # PUBLIC_INTERFACE
    def append_message(self, session_id: str, role: str, content: str) -> StoredMessage:
        """Append a message to the given session and return it.

        Args:
            session_id: Unique identifier for a chat session.
            role: The role of the message author ("user" or "assistant").
            content: The message text.

        Returns:
            The stored message with timestamp.

        Raises:
            ValueError: If role or content is invalid.
        """
        role = role.strip().lower()
        if role not in {"user", "assistant"}:
            raise ValueError("role must be 'user' or 'assistant'")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a non-empty string")

        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = SessionData()
                self._sessions[session_id] = session

            message: StoredMessage = {
                "role": role,
                "content": content.strip(),
                "timestamp": time.time(),
            }
            session.history.append(message)
            session.updated_at = message["timestamp"]
            return message

    # PUBLIC_INTERFACE
    def get_history(self, session_id: str, limit: Optional[int] = None) -> List[StoredMessage]:
        """Get the chat history for a session.

        Args:
            session_id: Unique identifier for the chat session.
            limit: Optional limit on the number of most recent messages to return.

        Returns:
            A list of stored messages in chronological order.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return []
            if limit is not None and limit > 0:
                return session.history[-limit:]
            return list(session.history)

    # PUBLIC_INTERFACE
    def clear_session(self, session_id: str) -> None:
        """Clear and remove a session and its history.

        Args:
            session_id: Unique identifier for the chat session.
        """
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]

    # PUBLIC_INTERFACE
    def list_sessions(self) -> List[str]:
        """List all known session IDs."""
        with self._lock:
            return list(self._sessions.keys())

    # PUBLIC_INTERFACE
    def session_stats(self, session_id: str) -> Dict[str, float | int]:
        """Return basic stats for a session.

        Args:
            session_id: Unique identifier for the chat session.

        Returns:
            Dictionary containing created_at, updated_at, and message_count. If the
            session does not exist, returns zeros and count 0.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return {"created_at": 0.0, "updated_at": 0.0, "message_count": 0}
            return {
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "message_count": len(session.history),
            }
