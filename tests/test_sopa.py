import time
from fastapi.testclient import TestClient
from llm_server.server import app, model_name

client = TestClient(app)

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
        'model': model_name or 'sshleifer/tiny-gpt2',
        'messages': [{'role': 'user', 'content': 'Hi'}],
        'temperature': 0.1,
        'max_tokens': 5
    }
    response = client.post('/v1/chat/completions', json=data)
    assert response.status_code == 200

def test_caching():
    data = {
        'model': model_name or 'sshleifer/tiny-gpt2',
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
        'model': model_name or 'sshleifer/tiny-gpt2',
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
        'model': model_name or 'sshleifer/tiny-gpt2',
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
