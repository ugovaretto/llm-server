import pytest
import requests
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_chat_completion():
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "meta-llama/Meta-Llama-3-8B-Instruct",
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
            "model": "meta-llama/Meta-Llama-3-8B-Instruct",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "max_tokens": 150
        },
        headers=headers
    )
    assert response.status_code == 200
    content = response.text
    assert "data: {" in content
    # Check that it's a proper streaming response
    assert response.headers.get("content-type") == "text/event-stream"


def test_list_models():
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert isinstance(data["data"], list)
    assert len(data["data"]) > 0
    model = data["data"][0]
    assert model["id"] == "meta-llama/Meta-Llama-3-8B-Instruct"
    assert model["object"] == "model"
    assert model["owned_by"] == "meta-llama"


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
