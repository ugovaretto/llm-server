import time
from fastapi.testclient import TestClient
import llm_server.server as server

class FakeBatch(dict):
    def to(self, device):
        return self

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

server.TextIteratorStreamer = FakeStreamer
server.model_name = "test-model"
server.tokenizer = FakeTokenizer()
server.model = FakeModel()

client = TestClient(server.app)

def test_health_endpoint():
    response = client.get('/health')
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'healthy'
    assert 'SOPA' in data['service']

def test_metrics_endpoint():
    response = client.get('/metrics')
    assert response.status_code == 200
    data = response.json()
    assert 'sopa_version' in data
    assert 'performance' in data
    assert 'cache' in data
    assert 'rate_limiter' in data

def test_rate_limiting():
    data = {
        'model': 'test-model',
        'messages': [{'role': 'user', 'content': 'Hi'}],
        'temperature': 0.1,
        'max_tokens': 5
    }
    response = client.post('/v1/chat/completions', json=data)
    assert response.status_code == 200

def test_caching():
    data = {
        'model': 'test-model',
        'messages': [{'role': 'user', 'content': 'What is 2+2?'}],
        'temperature': 0.1,
        'max_tokens': 10
    }
    metrics_before = client.get('/metrics').json()
    cache_before = metrics_before['cache']
    start = time.time()
    response1 = client.post('/v1/chat/completions', json=data)
    latency1 = time.time() - start
    assert response1.status_code == 200
    start = time.time()
    response2 = client.post('/v1/chat/completions', json=data)
    latency2 = time.time() - start
    assert response2.status_code == 200
    metrics_after = client.get('/metrics').json()
    cache_after = metrics_after['cache']
    assert cache_after['hits'] > cache_before['hits']

def test_api_key_auth():
    data = {
        'model': 'test-model',
        'messages': [{'role': 'user', 'content': 'Hello'}],
        'max_tokens': 5
    }
    headers = {'Authorization': 'Bearer sk-test-key-123'}
    response = client.post('/v1/chat/completions', json=data, headers=headers)
    assert response.status_code == 200
    response2 = client.post('/v1/chat/completions', json=data)
    assert response2.status_code == 200

def test_clear_cache_admin():
    response = client.post('/admin/clear-cache')
    assert response.status_code == 401
    headers = {'Authorization': 'Bearer sk-test-key-123'}
    response2 = client.post('/admin/clear-cache', headers=headers)
    assert response2.status_code == 200

def test_streaming_with_sopa():
    data = {
        'model': 'test-model',
        'messages': [{'role': 'user', 'content': 'Count to 3'}],
        'stream': True,
        'max_tokens': 5
    }
    response = client.post('/v1/chat/completions', json=data, headers={'Accept': 'text/event-stream'})
    assert response.status_code == 200
    assert response.headers.get('content-type') == 'text/event-stream'
    content = response.text
    chunks = [line for line in content.splitlines() if line.startswith('data: ')]
    assert len(chunks) > 0
