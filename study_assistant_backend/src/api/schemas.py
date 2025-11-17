from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


# PUBLIC_INTERFACE
class ChatRequest(BaseModel):
    """Input payload for submitting a chat question.

    Attributes:
        session_id: Unique identifier for a user's chat session.
        question: The user's question to be answered by the AI model.
        context: Optional context to improve the answer quality (e.g., subject).
        max_history: Optional limit for how many previous messages to include
                     when forming the prompt to the AI model (backend use).
    """
    session_id: str = Field(..., description="Unique identifier for a user's chat session.")
    question: str = Field(..., description="The user's question.")
    context: Optional[str] = Field(default=None, description="Optional context for the question.")
    max_history: Optional[int] = Field(default=10, ge=0, description="Optional limit on previous messages to consider.")

    @field_validator("session_id")
    @classmethod
    def session_id_not_empty(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("session_id must be a non-empty string")
        return v.strip()

    @field_validator("question")
    @classmethod
    def question_not_empty_and_informative(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("question must be a non-empty string")
        # Basic check to avoid low-information inputs; endpoints can tailor messages.
        text = v.strip()
        if len(text) < 3:
            raise ValueError("question is too short; please provide more details")
        return text


# PUBLIC_INTERFACE
class ChatResponse(BaseModel):
    """Response payload containing the AI-generated answer."""
    session_id: str = Field(..., description="Echo of the session ID used.")
    answer: str = Field(..., description="AI-generated answer to the user's question.")


# PUBLIC_INTERFACE
class Message(BaseModel):
    """A chat message in the conversation history."""

    role: Literal["user", "assistant"] = Field(..., description="The role of the message author.")
    content: str = Field(..., description="The message content.")
    timestamp: float = Field(..., description="Unix timestamp (seconds) when the message was stored.")


# PUBLIC_INTERFACE
class HistoryResponse(BaseModel):
    """Response payload containing the message history for a session."""
    session_id: str = Field(..., description="The session whose history is returned.")
    messages: List[Message] = Field(default_factory=list, description="List of messages in chronological order.")
