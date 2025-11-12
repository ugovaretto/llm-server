#!/usr/bin/env python3
import json
from fastapi.testclient import TestClient
import llm_server.server as server

# Real server streaming format test using incremental read and early exit

client = TestClient(server.app)

def test_streaming_format():
    response = client.post(
        '/v1/chat/completions',
        json={
            'model': server.model_name,
            'messages': [{'role': 'user', 'content': 'Count to 3'}],
            'stream': True,
            'max_tokens': 3,
        },
        headers={'Accept': 'text/event-stream'},
        timeout=5.0,
    )
    assert response.status_code == 200
    assert response.headers.get('content-type') == 'text/event-stream'
    content_chunks = 0
    done_received = False
    seen = 0
    for raw in response.iter_lines():
        if not raw:
            continue
        line = raw if isinstance(raw, str) else raw.decode('utf-8')
        if line == 'data: [DONE]':
            done_received = True
            break
        if line.startswith('data: '):
            data_json = json.loads(line[6:])
            assert data_json.get('object') == 'chat.completion.chunk'
            choice = data_json['choices'][0]
            assert 'delta' in choice
            if 'content' in choice['delta']:
                content_chunks += 1
            seen += 1
            if seen >= 3:
                break
    assert content_chunks > 0
    assert done_received or seen >= 1
