# Academic Query Assistant — Backend (FastAPI)

## Overview

This backend powers the Academic Query Assistant MVP. It exposes REST endpoints for session management, chat, and history, and integrates with a mock or real AI provider depending on environment configuration.

- Thread-safe in-memory session store (no database required in MVP).
- Endpoints: `/` (health), `/session`, `/chat`, `/history`.
- Uses a deterministic mock AI unless `OPENAI_API_KEY` is set and the `openai` package is available.

## MVP storage model and session management

- In-memory session store:
  - Implemented in `src/api/services/session_store.py` as `InMemorySessionStore`.
  - Per-session histories are stored in process memory and are not persisted across restarts.
  - A `threading.RLock` is used internally to ensure thread-safety for concurrent access.
  - A singleton instance is provided via FastAPI dependency in `src/api/deps.py` using `@lru_cache(maxsize=1)`.

- No database setup required:
  - The MVP does not depend on any external database service.
  - This design is acceptable for the current scope focused on functionality and tests.
  - Future iterations can introduce a persistent store without changing the public API shape.

## Environment variables

Backend relevant:
- `OPENAI_API_KEY` (optional in MVP): when set and the `openai` package is installed, real OpenAI calls are used; otherwise, a deterministic mock client is used.
- `OPENAI_MODEL` (optional): defaults to `gpt-4o-mini`.
- `OPENAI_BASE_URL` (optional): override OpenAI API base URL.
- `APP_ENV` (optional): defaults to `development`.

Note: The frontend container may define environment variables such as:
`REACT_APP_API_BASE, REACT_APP_BACKEND_URL, REACT_APP_FRONTEND_URL, REACT_APP_WS_URL, REACT_APP_NODE_ENV, REACT_APP_NEXT_TELEMETRY_DISABLED, REACT_APP_ENABLE_SOURCE_MAPS, REACT_APP_PORT, REACT_APP_TRUST_PROXY, REACT_APP_LOG_LEVEL, REACT_APP_HEALTHCHECK_PATH, REACT_APP_FEATURE_FLAGS, REACT_APP_EXPERIMENTS_ENABLED, REACT_APP_OPENAI_API_KEY`
These are not required by the backend for the MVP.

## Key files

- `src/api/services/session_store.py`: Thread-safe in-memory session store (`InMemorySessionStore`).
- `src/api/deps.py`: Provides a singleton session store via FastAPI dependency injection.
- `src/api/main.py`: FastAPI application and endpoints.
- `src/api/services/openai_client.py`: Mock and real AI client factory.

## Run locally

- Install dependencies:
  - `pip install -r requirements.txt`
- Start dev server:
  - `uvicorn src.api.main:app --host 0.0.0.0 --port 3001 --reload`
- OpenAPI docs: `http://localhost:3001/docs`

CORS is configured to allow `http://localhost:3000` (the React frontend) by default.

## Tests

- Run tests: `pytest`

The test suite covers:
- Health check
- Session creation
- Input validation and error responses
- Chat flow with auto-created session and mock AI responses
- History retrieval
- Upstream AI error translation to `502 Bad Gateway`
