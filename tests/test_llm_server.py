import pytest
import requests
from fastapi.testclient import TestClient
from llm_server.server import app
import llm_server.server as server

client = TestClient(app)


def test_chat_completion():
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": server.model_name,
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "max_tokens": 150
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert isinstance(data["choices"][0]["message"]["content"], str)
    assert len(data["choices"][0]["message"]["content"]) > 0  # Verify response is generated


def test_streaming_chat_completion():
    headers = {
        "Accept": "text/event-stream"
    }
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": server.model_name,
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "max_tokens": 150
        },
        headers=headers,
    )
    assert response.status_code == 200
    assert response.headers.get("content-type") == "text/event-stream"
    # Read only first few lines to avoid waiting for full stream
    lines = 0
    for raw in response.iter_lines():
        if not raw:
            continue
        line = raw if isinstance(raw, str) else raw.decode("utf-8")
        if line.startswith("data: "):
            lines += 1
            if lines >= 2:
                break
    assert lines >= 1


def test_list_models():
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert isinstance(data["data"], list)
    assert len(data["data"]) > 0
    model = data["data"][0]
    assert model["id"] == server.model_name
    assert model["object"] == "model"
    assert model["owned_by"] == server.model_name.split("/")[0]


def test_invalid_model():
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "invalid-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "max_tokens": 150
        }
    )
    assert response.status_code == 422
