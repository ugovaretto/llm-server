import pytest
from fastapi.testclient import TestClient
import llm_server.server as server

class FakeBatch(dict):
    def to(self, device):
        return self

class FakeTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return " ".join([m.get("content", "") for m in messages])

    def __call__(self, prompt, return_tensors="pt"):
        return FakeBatch({"input_ids": [[1, 2, 3]]})

    def decode(self, token_list, skip_special_tokens=True):
        return "".join(["x" for _ in token_list])

class FakeModel:
    def __init__(self):
        self.device = "cpu"

    def generate(self, **kwargs):
        streamer = kwargs.get("streamer")
        max_new_tokens = kwargs.get("max_new_tokens", 5)
        if streamer:
            for _ in range(max_new_tokens):
                streamer.put("x")
            streamer.end()
            return None
        input_ids = kwargs.get("input_ids", [[1, 2, 3]])
        return [input_ids[0] + [4] * max_new_tokens]

server.model_name = "test-model"
server.tokenizer = FakeTokenizer()
server.model = FakeModel()

client = TestClient(server.app)


def test_chat_completion():
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
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
            "model": "test-model",
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
    assert model["id"] == "test-model"
    assert model["object"] == "model"
    assert model["owned_by"] == "unknown"


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
