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

def test_streaming_format():
    response = client.post(
        '/v1/chat/completions',
        json={
            'model': 'test-model',
            'messages': [{'role': 'user', 'content': 'Count to 3'}],
            'stream': True,
            'max_tokens': 3,
        },
        headers={'Accept': 'text/event-stream'},
    )
    assert response.status_code == 200
    assert response.headers.get('content-type') == 'text/event-stream'
    chunks = []
    content_chunks = 0
    done_received = False
    for line in response.text.splitlines():
        if line == 'data: [DONE]':
            done_received = True
            break
        if line.startswith('data: '):
            data_json = json.loads(line[6:])
            chunks.append(data_json)
            assert 'id' in data_json
            assert 'object' in data_json and data_json['object'] == 'chat.completion.chunk'
            assert 'created' in data_json
            assert 'model' in data_json
            assert 'choices' in data_json
            choice = data_json['choices'][0]
            assert 'index' in choice
            assert 'delta' in choice
            assert 'logprobs' in choice
            assert 'finish_reason' in choice
            if 'content' in choice['delta']:
                content_chunks += 1
    assert content_chunks > 0
    assert done_received
