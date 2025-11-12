#!/usr/bin/env python3
import json
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

def test_streaming():
    headers = {'Accept': 'text/event-stream'}
    data = {
        'model': 'test-model',
        'messages': [{'role': 'user', 'content': 'Hello'}],
        'temperature': 0.7,
        'max_tokens': 10
    }
    response = client.post('/v1/chat/completions', json=data, headers=headers)
    assert response.status_code == 200
    assert response.headers.get('content-type') == 'text/event-stream'
    content_chunks = 0
    for line in response.text.splitlines():
        if line == 'data: [DONE]':
            break
        if line.startswith('data: '):
            body = line[6:].strip()
            if body.startswith('{'):
                try:
                    payload = json.loads(body)
                except Exception:
                    continue
                delta = payload['choices'][0].get('delta', {})
                if 'content' in delta:
                    content_chunks += 1
    assert content_chunks > 0
