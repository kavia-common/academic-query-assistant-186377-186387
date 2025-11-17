from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure deps and schemas are importable for future endpoint wiring
# (No direct usage here yet, but keeps interfaces ready and discoverable)
try:
    from . import deps  # noqa: F401
    from . import schemas  # noqa: F401
except Exception:
    # Safe-guard in case of partial builds; endpoints do not depend on these yet.
    pass

app = FastAPI(
    title="Academic Query Assistant API",
    description="Backend API for handling academic questions, session histories, and AI answers.",
    version="0.1.0",
    openapi_tags=[
        {"name": "health", "description": "Operational endpoints."},
        {"name": "chat", "description": "Chat endpoints for asking questions and retrieving history."},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For MVP dev; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["health"], summary="Health Check")
def health_check():
    """Simple health check to verify the service is running.

    Returns:
        JSON with message 'Healthy' to indicate service availability.
    """
    return {"message": "Healthy"}
