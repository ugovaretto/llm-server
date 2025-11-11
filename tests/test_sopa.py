#!/usr/bin/env python3
"""Test SOPA (Stream Optimization and Performance Accelerator) features"""

import requests
import json
import time

BASE_URL = 'http://127.0.0.1:8000'
VALID_API_KEY = 'sk-test-key-123'

def test_health_endpoint():
    """Test the health check endpoint"""
    print("\n=== Testing Health Endpoint ===")
    response = requests.get(f'{BASE_URL}/health')
    
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'healthy'
    assert 'SOPA' in data['service']
    print(f"✓ Health check passed: {data}")

def test_metrics_endpoint():
    """Test the metrics endpoint"""
    print("\n=== Testing Metrics Endpoint ===")
    response = requests.get(f'{BASE_URL}/metrics')
    
    assert response.status_code == 200
    data = response.json()
    assert 'sopa_version' in data
    assert 'performance' in data
    assert 'cache' in data
    assert 'rate_limiter' in data
    print(f"✓ Metrics endpoint works")
    print(f"  Performance: {data['performance']}")
    print(f"  Cache: {data['cache']}")
    print(f"  Rate Limiter: {data['rate_limiter']}")

def test_rate_limiting():
    """Test rate limiting functionality"""
    print("\n=== Testing Rate Limiting ===")
    
    # Make multiple requests quickly
    url = f'{BASE_URL}/v1/chat/completions'
    data = {
        'model': 'meta-llama/Meta-Llama-3-8B-Instruct',
        'messages': [{'role': 'user', 'content': 'Hi'}],
        'temperature': 0.1,
        'max_tokens': 5
    }
    
    # The server allows 100 requests per 60 seconds
    # We'll just test that it accepts normal usage
    response = requests.post(url, json=data)
    assert response.status_code == 200
    print("✓ Rate limiting allows normal requests")

def test_caching():
    """Test response caching for deterministic requests"""
    print("\n=== Testing Response Caching ===")
    
    url = f'{BASE_URL}/v1/chat/completions'
    data = {
        'model': 'meta-llama/Meta-Llama-3-8B-Instruct',
        'messages': [{'role': 'user', 'content': 'What is 2+2?'}],
        'temperature': 0.1,  # Low temperature for caching
        'max_tokens': 10
    }
    
    # Get initial metrics
    metrics_before = requests.get(f'{BASE_URL}/metrics').json()
    cache_before = metrics_before['cache']
    
    # First request (cache miss)
    start = time.time()
    response1 = requests.post(url, json=data)
    latency1 = time.time() - start
    
    assert response1.status_code == 200
    result1 = response1.json()
    
    # Second identical request (should be cached)
    start = time.time()
    response2 = requests.post(url, json=data)
    latency2 = time.time() - start
    
    assert response2.status_code == 200
    result2 = response2.json()
    
    # Get updated metrics
    metrics_after = requests.get(f'{BASE_URL}/metrics').json()
    cache_after = metrics_after['cache']
    
    print(f"  First request latency: {latency1:.2f}s")
    print(f"  Second request latency: {latency2:.2f}s")
    print(f"  Cache hits: {cache_after['hits']} (was {cache_before['hits']})")
    print(f"  Speedup: {latency1/latency2:.2f}x faster")
    
    # Cache should have at least one hit
    assert cache_after['hits'] > cache_before['hits'], "Cache should register hits"
    print("✓ Response caching works and provides speedup")

def test_api_key_auth():
    """Test API key authentication"""
    print("\n=== Testing API Key Authentication ===")
    
    # Test with valid API key
    url = f'{BASE_URL}/v1/chat/completions'
    data = {
        'model': 'meta-llama/Meta-Llama-3-8B-Instruct',
        'messages': [{'role': 'user', 'content': 'Hello'}],
        'max_tokens': 5
    }
    headers = {'Authorization': f'Bearer {VALID_API_KEY}'}
    
    response = requests.post(url, json=data, headers=headers)
    assert response.status_code == 200
    print("✓ Valid API key accepted")
    
    # Test without API key (should still work, just no auth)
    response = requests.post(url, json=data)
    assert response.status_code == 200
    print("✓ Requests work without API key (optional auth)")

def test_clear_cache_admin():
    """Test admin cache clearing endpoint"""
    print("\n=== Testing Admin Cache Clear ===")
    
    # Try without API key (should fail)
    response = requests.post(f'{BASE_URL}/admin/clear-cache')
    assert response.status_code == 401
    print("✓ Cache clear requires authentication")
    
    # Try with valid API key
    headers = {'Authorization': f'Bearer {VALID_API_KEY}'}
    response = requests.post(f'{BASE_URL}/admin/clear-cache', headers=headers)
    assert response.status_code == 200
    data = response.json()
    print(f"✓ Cache cleared: {data}")

def test_streaming_with_sopa():
    """Test that streaming still works with SOPA"""
    print("\n=== Testing Streaming with SOPA ===")
    
    url = f'{BASE_URL}/v1/chat/completions'
    data = {
        'model': 'meta-llama/Meta-Llama-3-8B-Instruct',
        'messages': [{'role': 'user', 'content': 'Count to 3'}],
        'stream': True,
        'max_tokens': 20
    }
    
    response = requests.post(url, json=data, stream=True)
    assert response.status_code == 200
    assert response.headers.get('content-type') == 'text/event-stream'
    
    chunks = 0
    for line in response.iter_lines():
        if line and line.decode('utf-8').startswith('data: '):
            chunks += 1
            if chunks > 5:  # Just test a few chunks
                break
    
    assert chunks > 0
    print(f"✓ Streaming works with SOPA ({chunks} chunks received)")

if __name__ == '__main__':
    print("=" * 60)
    print("SOPA Feature Test Suite")
    print("=" * 60)
    
    try:
        test_health_endpoint()
        test_metrics_endpoint()
        test_api_key_auth()
        test_rate_limiting()
        test_caching()
        test_clear_cache_admin()
        test_streaming_with_sopa()
        
        print("\n" + "=" * 60)
        print("✓✓✓ ALL SOPA TESTS PASSED ✓✓✓")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("\n✗ Connection Error: Server not running")
        print("Start the server with: .venv/bin/python main.py --port 8000")
        exit(1)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
