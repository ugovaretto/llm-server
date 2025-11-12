import pytest
import requests
from fastapi.testclient import TestClient
from llm_server.server import app
import llm_server.server as server

class FakeBatch(dict):
    def to(self, device):
        return self

class FakeTokenizer:
    eos_token_id = 0
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return " ".join([m.get("content", "") for m in messages])
    def __call__(self, prompt, return_tensors="pt", **kwargs):
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

class FakeStreamer:
    def __init__(self, tokenizer, skip_prompt=True, skip_special_tokens=True):
        self.queue = []
        self.closed = False
    def put(self, text):
        self.queue.append(text)
    def end(self):
        self.closed = True
    def __iter__(self):
        idx = 0
        while True:
            if idx < len(self.queue):
                item = self.queue[idx]
                idx += 1
                yield item
            elif self.closed:
                break

# Initialize test fakes and model id expected by tests
server.model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
server.tokenizer = FakeTokenizer()
server.model = FakeModel()
server.TextIteratorStreamer = FakeStreamer

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
