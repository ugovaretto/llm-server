from fastapi.testclient import TestClient
import llm_server.server as server

class FakeBatch(dict):
    def to(self, device):
        return self

class FakeTokenizer:
    eos_token_id = 0
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "prompt"
    def __call__(self, prompt, return_tensors="pt", **kwargs):
        return FakeBatch({"input_ids": [[1, 2, 3]]})
    def decode(self, token_list, skip_special_tokens=True):
        return "x" * len(token_list)

class FakeModel:
    def __init__(self):
        self.device = "cpu"
        self.last_kwargs = None
    def generate(self, **kwargs):
        self.last_kwargs = kwargs
        input_ids = kwargs.get("input_ids", [[1,2,3]])
        return [input_ids[0] + [4] * kwargs.get("max_new_tokens", 5)]

server.model_name = "test-model"

client = TestClient(server.app)

def test_parameters_mapping():
    server.tokenizer = FakeTokenizer()
    server.model = FakeModel()
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "repetition_penalty": 1.1,
        "max_tokens": 8,
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    # Verify last generation kwargs captured by FakeModel
    kwargs = server.model.last_kwargs
    assert kwargs is not None
    assert kwargs.get("do_sample") is True
    assert kwargs.get("temperature") == 0.7
    assert kwargs.get("top_p") == 0.95
    assert kwargs.get("top_k") == 40
    assert kwargs.get("repetition_penalty") == 1.1
