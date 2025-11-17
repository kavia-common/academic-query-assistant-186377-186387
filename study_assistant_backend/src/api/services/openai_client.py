from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional, Protocol, TypedDict

from ..config import get_settings


class ChatMessage(TypedDict):
    """Represents a chat message for the AI model."""
    role: str  # "system" | "user" | "assistant"
    content: str


class AIClient(Protocol):
    """Protocol for AI client implementations."""

    # PUBLIC_INTERFACE
    def chat(self, messages: List[ChatMessage], model: Optional[str] = None) -> str:
        """Send a list of chat messages to an AI model and return the assistant's reply.

        Args:
            messages: Ordered conversation context including user and assistant messages.
            model: Optional model name; falls back to configured default.

        Returns:
            The assistant response text.
        """
        ...


class _MockAIClient:
    """Deterministic mock AI client used when OpenAI key/package is unavailable.

    Produces a concise, predictable response based on a stable hash of inputs and seed.
    """

    def __init__(self, seed: str) -> None:
        self._seed = seed

    def _summarize_user_question(self, messages: List[ChatMessage]) -> str:
        # Find last user message for succinct echo
        user_msgs = [m["content"] for m in messages if m.get("role") == "user"]
        last = user_msgs[-1] if user_msgs else ""
        return last.strip()[:160]  # keep brief

    def chat(self, messages: List[ChatMessage], model: Optional[str] = None) -> str:
        payload = {
            "seed": self._seed,
            "model": model or get_settings().openai_model,
            "messages": messages,
        }
        as_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        digest = hashlib.sha256(as_bytes).hexdigest()[:12]
        brief = self._summarize_user_question(messages)
        hint = f' Q="{brief}"' if brief else ""
        return f"[MockAnswer:{digest}] This is a simulated response for testing.{hint}"


class _OpenAIClient:
    """Thin wrapper around the OpenAI SDK with safe import and configuration."""

    def __init__(self, api_key: str, base_url: Optional[str]) -> None:
        # Safe import to avoid hard dependency during CI or local runs.
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "OpenAI package is not installed. Install 'openai' to enable real calls."
            ) from exc

        # Initialize client with optional base_url override
        if base_url:
            self._client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self._client = OpenAI(api_key=api_key)

    def _convert_messages(self, messages: List[ChatMessage]) -> List[Dict[str, str]]:
        # OpenAI expects a list of dicts with role/content
        normalized: List[Dict[str, str]] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if not isinstance(role, str) or not isinstance(content, str):
                continue
            normalized.append({"role": role, "content": content})
        return normalized

    def chat(self, messages: List[ChatMessage], model: Optional[str] = None) -> str:
        cfg = get_settings()
        use_model = model or cfg.openai_model

        # Perform the chat completion call. We prefer the new responses API if available;
        # otherwise fall back to chat.completions for broad compatibility.
        try:
            # Try "responses" API pathway
            # noinspection PyUnresolvedReferences
            resp = self._client.chat.completions.create(  # type: ignore[attr-defined]
                model=use_model,
                messages=self._convert_messages(messages),
                temperature=0.2,
            )
            # Extract text
            content = (resp.choices[0].message.content or "").strip()  # type: ignore[index, attr-defined]
            return content
        except Exception:
            # As an extra guard, try legacy path or raise
            try:
                # noinspection PyUnresolvedReferences
                resp = self._client.chat.completions.create(  # type: ignore[attr-defined]
                    model=use_model,
                    messages=self._convert_messages(messages),
                    temperature=0.2,
                )
                content = (resp.choices[0].message.content or "").strip()  # type: ignore[index, attr-defined]
                return content
            except Exception as exc:
                # On any error, propagate a concise message to caller
                raise RuntimeError(f"OpenAI call failed: {exc}") from exc


# PUBLIC_INTERFACE
def get_ai_client() -> AIClient:
    """Factory that returns an AI client.

    Behavior:
    - If OPENAI_API_KEY is set and the openai package is available, returns a real client.
    - Otherwise returns a deterministic mock for development/testing without external calls.
    """
    cfg = get_settings()
    if cfg.openai_api_key:
        try:
            return _OpenAIClient(api_key=cfg.openai_api_key, base_url=cfg.openai_base_url)
        except Exception:
            # Fall back to mock if import/init fails
            return _MockAIClient(seed=cfg.mock_deterministic_seed)
    else:
        return _MockAIClient(seed=cfg.mock_deterministic_seed)
