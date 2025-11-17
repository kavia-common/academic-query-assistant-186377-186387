import os
import uuid
from typing import Optional
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Import the FastAPI app
from src.api.main import app


# Test client fixture
@pytest.fixture(scope="module")
def client():
    # Ensure OPENAI_API_KEY is not set so mock AI is used by default
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]
    return TestClient(app)


def test_health_check_ok(client: TestClient):
    # GET /
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    assert data.get("message") == "Healthy"


def test_session_get_returns_uuid(client: TestClient):
    resp = client.get("/session")
    assert resp.status_code == 200
    payload = resp.json()
    assert "session_id" in payload
    sid = payload["session_id"]
    # Validate UUID format
    uuid_obj = uuid.UUID(sid)
    assert str(uuid_obj) == sid


@pytest.mark.parametrize(
    "question, expected_msg_fragment",
    [
        ("", "must be a non-empty"),  # empty string
        ("  ", "must be a non-empty"),  # whitespace only
        ("??", "unclear"),  # unclear (no alphanumeric) -> custom heuristic in main._validate_question
        ("x" * 1001, "too long"),  # length > 1000
    ],
)
def test_chat_validation_errors(client: TestClient, question: str, expected_msg_fragment: str):
    # No session header to exercise auto-creation path
    payload = {"question": question}
    resp = client.post("/chat", json=payload)
    assert resp.status_code == 422
    # FastAPI returns a standardized error shape
    detail = resp.json().get("detail")
    assert detail is not None
    # Ensure our validation message is present
    # It may be a list of errors with msg fields
    messages = []
    if isinstance(detail, list):
        for err in detail:
            msg = err.get("msg")
            if msg:
                messages.append(msg)
    else:
        messages.append(str(detail))
    joined = " | ".join(messages)
    assert expected_msg_fragment in joined


def test_chat_success_with_auto_created_session_and_mock_ai(client: TestClient):
    # No X-Session-Id header, should auto-create and return 201 with header set
    payload = {"question": "What is the Pythagorean theorem?"}
    resp = client.post("/chat", json=payload)
    # Since new session id should be created
    assert resp.status_code == 201
    # Check header contains X-Session-Id
    session_id = resp.headers.get("X-Session-Id")
    assert session_id is not None
    # Response body contains answer and session_id
    body = resp.json()
    assert "session_id" in body and body["session_id"] == session_id
    assert isinstance(body.get("answer"), str) and len(body["answer"]) > 0
    # Ensure answer came from mock client format
    # Mock client returns string starting with "[MockAnswer:"
    assert body["answer"].startswith("[MockAnswer:")


def test_history_returns_messages_for_session(client: TestClient):
    # First, create a chat turn to generate messages within a session
    payload = {"question": "Explain Newton's first law."}
    resp = client.post("/chat", json=payload)
    assert resp.status_code == 201
    session_id = resp.headers.get("X-Session-Id")
    assert session_id

    # Now retrieve history for that session
    resp_hist = client.get("/history", headers={"X-Session-Id": session_id})
    assert resp_hist.status_code == 200
    data = resp_hist.json()
    assert data.get("session_id") == session_id
    messages = data.get("messages")
    assert isinstance(messages, list)
    # Should contain at least the prior user and assistant messages (2 entries)
    assert len(messages) >= 2
    # Validate message structure
    for m in messages:
        assert m.get("role") in ("user", "assistant")
        assert isinstance(m.get("content"), str)
        # timestamp is numeric
        assert isinstance(m.get("timestamp"), (int, float))


def test_openai_error_results_in_502(client: TestClient):
    # Patch get_ai_client() to raise error when calling chat, simulating upstream failure
    class DummyAIClient:
        def chat(self, messages, model: Optional[str] = None) -> str:
            raise RuntimeError("simulated openai error")

    with patch("src.api.services.openai_client.get_ai_client", return_value=DummyAIClient()):
        payload = {"question": "Trigger upstream error"}
        resp = client.post("/chat", json=payload)
        assert resp.status_code == 502
        detail = resp.json().get("detail")
        assert "simulated openai error" in str(detail)
