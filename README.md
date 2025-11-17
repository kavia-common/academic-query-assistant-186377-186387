# Academic Query Assistant — Backend (FastAPI)

This backend powers the Academic Query Assistant MVP, exposing REST endpoints for session management, chat, and history.

## MVP data storage and session management

- In-memory session store (no external database):
  - The MVP uses a thread-safe in-memory session store implemented in `src/api/services/session_store.py` (`InMemorySessionStore`).
  - It relies on a `threading.RLock` to ensure thread-safety for concurrent access.
  - Session chat histories are kept in process memory and are not persisted between restarts.
  - A singleton instance is provided via FastAPI dependency in `src/api/deps.py` using `@lru_cache(maxsize=1)`.

- No database setup required:
  - There is no DB dependency for the MVP. You do not need to configure or run any database service.
  - This is acceptable for the current scope focused on functionality and tests. Future iterations can introduce a persistent store without changing the public API.

## Key files

- `src/api/services/session_store.py`: Thread-safe in-memory session store.
- `src/api/deps.py`: Provides a singleton session store for the app.
- `src/api/main.py`: FastAPI app and endpoints (`/`, `/session`, `/chat`, `/history`).

## Running tests

- Install dependencies: `pip install -r requirements.txt`
- Run tests: `pytest`

## Notes

- The service will use a deterministic mock AI client unless `OPENAI_API_KEY` is set and the `openai` package is available.
- CORS is configured to allow `http://localhost:3000` for the React frontend.