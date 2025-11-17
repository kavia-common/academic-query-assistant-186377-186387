from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional


class Settings:
    """Application settings sourced from environment variables.

    Attributes:
        openai_api_key: API key for OpenAI. If missing, the system will use a mock client.
        openai_model: The model name to use when calling OpenAI (default 'gpt-4o-mini').
        openai_base_url: Optional override for OpenAI API base URL.
        app_env: Optional application environment (e.g., development, production).
        mock_deterministic_seed: Seed string to make mock responses deterministic per input.
    """

    def __init__(self) -> None:
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY") or None
        self.openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.openai_base_url: Optional[str] = os.getenv("OPENAI_BASE_URL") or None
        self.app_env: str = os.getenv("APP_ENV", "development")
        # If provided, controls mock output deterministically
        self.mock_deterministic_seed: str = os.getenv("MOCK_DETERMINISTIC_SEED", "academic-query-assistant")


# PUBLIC_INTERFACE
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return process-wide application settings loaded from environment."""
    return Settings()
