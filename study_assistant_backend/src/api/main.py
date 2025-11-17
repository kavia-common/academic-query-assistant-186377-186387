from __future__ import annotations

import uuid
from typing import Any, List, Optional, Tuple

from fastapi import Depends, FastAPI, Header, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from .config import get_settings
from .deps import get_session_store
from .schemas import ChatRequest, ChatResponse, HistoryResponse, Message
from .services.openai_client import ChatMessage, get_ai_client
from .services.session_store import InMemorySessionStore

# Initialize FastAPI with metadata and tags for OpenAPI
app = FastAPI(
    title="Academic Query Assistant API",
    description="Backend API for handling academic questions, session histories, and AI answers.",
    version="0.1.0",
    openapi_tags=[
        {"name": "health", "description": "Operational endpoints."},
        {"name": "chat", "description": "Chat endpoints for asking questions and retrieving history."},
        {"name": "session", "description": "Session management endpoints."},
    ],
)

# Configure CORS for local development frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

def _ensure_session_id(header_session_id: Optional[str]) -> Tuple[str, bool]:
    """
    Ensure we have a session id; generate one if header missing/blank.

    Returns:
        (session_id, created_flag) where created_flag indicates whether a new id
        was generated this request.
    """
    if isinstance(header_session_id, str) and header_session_id.strip():
        return header_session_id.strip(), False
    return str(uuid.uuid4()), True

def _sanitize_val_error_detail(detail: Any) -> Any:
    """
    Sanitize ValidationError details to ensure JSON-serializable payloads.

    - If detail is a list[dict], ensure each dict only contains JSON-safe values.
    - Specifically, stringify any Exception instances under 'ctx' or any nested value.
    """
    def _safe(val: Any) -> Any:
        # Convert Exception instances to string
        if isinstance(val, Exception):
            return str(val)
        # Recursively sanitize dicts/lists
        if isinstance(val, dict):
            return {k: _safe(v) for k, v in val.items()}
        if isinstance(val, list):
            return [_safe(v) for v in val]
        return val

    return _safe(detail)

# PUBLIC_INTERFACE
@app.get("/", tags=["health"], summary="Health Check")
def health_check():
    """Simple health check to verify the service is running.

    Returns:
        JSON with message 'Healthy' to indicate service availability.
    """
    return {"message": "Healthy"}

# PUBLIC_INTERFACE
@app.get(
    "/session",
    tags=["session"],
    summary="Create a new chat session",
    description="Generates and returns a new session ID for tracking a user's chat history.",
    responses={
        200: {"description": "New session created"},
    },
)
def create_session() -> dict:
    """Create a new session ID.

    Returns:
        JSON containing 'session_id'.
    """
    sid = str(uuid.uuid4())
    return {"session_id": sid}

def _validate_question(text: str) -> Optional[str]:
    """
    Additional heuristic validation for input questions.

    Rules:
    - Trimmed length must be > 0 (redundant guard).
    - Max length 1000 characters.
    - Minimal clarity heuristic: must contain at least one alphanumeric character,
      and not be comprised solely of punctuation or whitespace.
    """
    if not isinstance(text, str):
        return "question must be a string"
    trimmed = text.strip()
    if not trimmed:
        return "question must not be empty"
    if len(trimmed) > 1000:
        return "question is too long; maximum 1000 characters"
    if not any(ch.isalnum() for ch in trimmed):
        return "question appears unclear; please include alphanumeric characters"
    return None

def _history_to_messages(history: List[dict]) -> List[ChatMessage]:
    """Convert stored messages to AI client messages."""
    msgs: List[ChatMessage] = []
    for m in history:
        role = m.get("role", "user")
        content = m.get("content", "")
        if not isinstance(content, str):
            continue
        msgs.append({"role": role, "content": content})
    return msgs

# PUBLIC_INTERFACE
@app.post(
    "/chat",
    tags=["chat"],
    summary="Submit a question and receive an AI-generated answer",
    description=(
        "Accepts a question, validates it, and returns an AI-generated answer. "
        "Maintains per-session history in memory. The session can be passed as the "
        "X-Session-Id header. If missing, a new one is created and returned in the response header."
    ),
    response_model=ChatResponse,
    responses={
        200: {"description": "Answer generated successfully"},
        201: {"description": "Answer generated successfully (new session id created)"},
        422: {"description": "Validation error"},
        502: {"description": "Upstream AI provider error"},
    },
)
def chat(
    payload: dict,
    response: Response,
    x_session_id: Optional[str] = Header(default=None, alias="X-Session-Id"),
    store: InMemorySessionStore = Depends(get_session_store),
) -> ChatResponse:
    """Chat endpoint that validates input and produces an AI answer.

    Parameters:
        payload: Request JSON body containing 'question' and optionally 'context' and 'max_history'.
                 The 'session_id' field is accepted but typically the session is provided via header.
        x_session_id: Optional session id provided via header 'X-Session-Id'. A new one is created if missing.
        store: Dependency-injected in-memory session store.

    Returns:
        ChatResponse with the answer and the effective session_id.

    Error handling:
        - Returns 422 for invalid inputs.
        - Returns 502 when OpenAI (or upstream AI) errors occur.
    """
    # Ensure / auto-create session id
    session_id, created = _ensure_session_id(x_session_id)
    if created:
        response.headers["X-Session-Id"] = session_id

    # Merge/normalize incoming body into ChatRequest using the effective session id
    data = dict(payload or {})
    data.setdefault("session_id", session_id)

    # Pydantic model validation for structural checks
    try:
        req = ChatRequest.model_validate(data)
    except ValidationError as ve:
        # Sanitize detail to ensure JSON-serializable, particularly ctx values
        safe_detail = _sanitize_val_error_detail(ve.errors())
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=safe_detail)

    # Additional heuristic validation for question
    err = _validate_question(req.question)
    if err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=[{"loc": ["body", "question"], "msg": err, "type": "value_error"}],
        )

    # Persist user message into history
    store.append_message(session_id=session_id, role="user", content=req.question)

    # Build AI prompt/messages using limited history for context
    history = store.get_history(session_id, limit=max(req.max_history or 0, 0) or None)
    messages = _history_to_messages(history)

    # Optionally include a brief system/context instruction if provided
    if req.context:
        messages.insert(0, {"role": "system", "content": f"Context: {req.context}".strip()})

    # Get AI client (real or mock)
    client = get_ai_client()
    cfg = get_settings()
    try:
        answer_text = client.chat(messages=messages, model=cfg.openai_model)
    except Exception as exc:
        # Translate upstream errors to 502 Bad Gateway and ensure no 201 is leaked.
        # We intentionally do not touch response.status_code here; FastAPI will use 502 from HTTPException.
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        )

    # Store assistant answer
    store.append_message(session_id=session_id, role="assistant", content=answer_text)

    # Choose status code: 201 if a new session was created during this request.
    # Set it immediately before returning on the success path only.
    if created:
        response.status_code = status.HTTP_201_CREATED

    return ChatResponse(session_id=session_id, answer=answer_text)

# PUBLIC_INTERFACE
@app.get(
    "/history",
    tags=["chat"],
    summary="Get session chat history",
    description=(
        "Returns the chat history for the provided session. "
        "Provide the session via 'X-Session-Id' header. If not provided, "
        "a new empty session is created and returned."
    ),
    response_model=HistoryResponse,
    responses={
        200: {"description": "History returned (or empty if new session)"},
        201: {"description": "New empty session created and returned"},
    },
)
def get_history(
    response: Response,
    x_session_id: Optional[str] = Header(default=None, alias="X-Session-Id"),
    store: InMemorySessionStore = Depends(get_session_store),
) -> HistoryResponse:
    """Return the message history for the requesting session.

    Parameters:
        x_session_id: Optional 'X-Session-Id' header to identify the session.
        store: Dependency-injected in-memory session store.

    Returns:
        HistoryResponse containing the session_id and array of messages.
    """
    session_id, created = _ensure_session_id(x_session_id)
    if created:
        response.headers["X-Session-Id"] = session_id
        response.status_code = status.HTTP_201_CREATED

    history = store.get_history(session_id)
    # Map internal messages to public schema
    messages: List[Message] = [
        Message(role=m["role"], content=m["content"], timestamp=float(m["timestamp"]))
        for m in history
    ]
    return HistoryResponse(session_id=session_id, messages=messages)
