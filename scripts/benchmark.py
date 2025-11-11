#!/usr/bin/env python3
"""Benchmark script to measure LLM server performance"""

import requests
import json
import time
from typing import List, Tuple

BASE_URL = 'http://127.0.0.1:8000'

def count_tokens_approx(text: str) -> int:
    """Rough approximation: 1 token ≈ 4 characters"""
    return len(text) // 4

def benchmark_streaming(model: str, prompt: str, max_tokens: int, temperature: float = 0.7) -> Tuple[int, float, str]:
    """Benchmark streaming endpoint and return (tokens, seconds, response)"""
    url = f'{BASE_URL}/v1/chat/completions'
    data = {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'stream': True,
        'max_tokens': max_tokens,
        'temperature': temperature
    }
    
    start_time = time.time()
    response = requests.post(url, json=data, stream=True, timeout=300)
    
    if response.status_code != 200:
        raise Exception(f"Request failed: {response.status_code} - {response.text}")
    
    full_content = ""
    token_count = 0
    first_token_time = None
    
    for line in response.iter_lines():
        if not line:
            continue
        
        decoded = line.decode('utf-8')
        
        if decoded == 'data: [DONE]':
            break
        
        if decoded.startswith('data: '):
            try:
                data_json = json.loads(decoded[6:])
                if 'choices' in data_json and len(data_json['choices']) > 0:
                    delta = data_json['choices'][0].get('delta', {})
                    if 'content' in delta:
                        if first_token_time is None:
                            first_token_time = time.time()
                        content = delta['content']
                        full_content += content
                        token_count += count_tokens_approx(content)
            except json.JSONDecodeError:
                pass
    
    end_time = time.time()
    total_time = end_time - start_time
    ttft = first_token_time - start_time if first_token_time else 0  # Time to first token
    
    return token_count, total_time, full_content, ttft

def benchmark_non_streaming(model: str, prompt: str, max_tokens: int, temperature: float = 0.7) -> Tuple[int, float, str]:
    """Benchmark non-streaming endpoint and return (tokens, seconds, response)"""
    url = f'{BASE_URL}/v1/chat/completions'
    data = {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'stream': False,
        'max_tokens': max_tokens,
        'temperature': temperature
    }
    
    start_time = time.time()
    response = requests.post(url, json=data, timeout=300)
    end_time = time.time()
    
    if response.status_code != 200:
        raise Exception(f"Request failed: {response.status_code} - {response.text}")
    
    result = response.json()
    content = result['choices'][0]['message']['content']
    actual_tokens = result['usage']['completion_tokens']
    total_time = end_time - start_time
    
    return actual_tokens, total_time, content

def get_model_name() -> str:
    """Get the currently loaded model from /v1/models"""
    response = requests.get(f'{BASE_URL}/v1/models')
    if response.status_code == 200:
        data = response.json()
        return data['data'][0]['id']
    return "unknown"

def run_benchmark():
    print("=" * 80)
    print("LLM SERVER PERFORMANCE BENCHMARK")
    print("=" * 80)
    
    # Check server is running
    try:
        model_name = get_model_name()
        print(f"\n✓ Server is running")
        print(f"✓ Model: {model_name}\n")
    except requests.exceptions.ConnectionError:
        print("\n✗ Server is not running!")
        print("Start it with: .venv/bin/python main.py --port 8000\n")
        return
    
    test_cases = [
        {
            "name": "Short Response (50 tokens)",
            "prompt": "Write a haiku about programming.",
            "max_tokens": 50,
            "temperature": 0.7
        },
        {
            "name": "Medium Response (150 tokens)",
            "prompt": "Explain what a REST API is in simple terms.",
            "max_tokens": 150,
            "temperature": 0.7
        },
        {
            "name": "Long Response (300 tokens)",
            "prompt": "Write a short story about a robot learning to paint.",
            "max_tokens": 300,
            "temperature": 0.7
        },
        {
            "name": "Deterministic (Low Temperature)",
            "prompt": "What is 2+2? Answer in one word.",
            "max_tokens": 50,
            "temperature": 0.1
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}/{len(test_cases)}: {test_case['name']}")
        print(f"{'='*80}")
        print(f"Prompt: {test_case['prompt']}")
        print(f"Max tokens: {test_case['max_tokens']}, Temperature: {test_case['temperature']}")
        
        # Streaming test
        print(f"\n[Streaming Mode]")
        try:
            tokens, duration, content, ttft = benchmark_streaming(
                model_name, 
                test_case['prompt'], 
                test_case['max_tokens'],
                test_case['temperature']
            )
            tokens_per_sec = tokens / duration if duration > 0 else 0
            print(f"  Tokens generated: {tokens}")
            print(f"  Total time: {duration:.2f}s")
            print(f"  Time to first token: {ttft:.2f}s")
            print(f"  ⚡ Speed: {tokens_per_sec:.2f} tokens/second")
            print(f"  Response preview: {content[:100]}...")
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        # Non-streaming test
        print(f"\n[Non-Streaming Mode]")
        try:
            tokens, duration, content = benchmark_non_streaming(
                model_name,
                test_case['prompt'],
                test_case['max_tokens'],
                test_case['temperature']
            )
            tokens_per_sec = tokens / duration if duration > 0 else 0
            print(f"  Tokens generated: {tokens}")
            print(f"  Total time: {duration:.2f}s")
            print(f"  ⚡ Speed: {tokens_per_sec:.2f} tokens/second")
            print(f"  Response preview: {content[:100]}...")
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        # Wait a bit between tests
        if i < len(test_cases):
            time.sleep(2)
    
    # Test cache performance
    print(f"\n{'='*80}")
    print(f"CACHE PERFORMANCE TEST")
    print(f"{'='*80}")
    print("Testing same request twice (should be cached on 2nd request)...")
    
    cache_prompt = "What is the capital of France?"
    cache_temp = 0.1  # Low temp for caching
    
    print("\nFirst request (cache miss):")
    try:
        tokens1, duration1, content1 = benchmark_non_streaming(model_name, cache_prompt, 50, cache_temp)
        print(f"  Time: {duration1:.2f}s")
        print(f"  Speed: {tokens1/duration1:.2f} tokens/s")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        duration1 = 0
    
    print("\nSecond request (should be cached):")
    try:
        tokens2, duration2, content2 = benchmark_non_streaming(model_name, cache_prompt, 50, cache_temp)
        print(f"  Time: {duration2:.2f}s")
        print(f"  Speed: {tokens2/duration2:.2f} tokens/s")
        if duration1 > 0 and duration2 > 0:
            speedup = duration1 / duration2
            print(f"  🚀 Cache speedup: {speedup:.1f}x faster")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Get server metrics
    print(f"\n{'='*80}")
    print("SERVER METRICS")
    print(f"{'='*80}")
    try:
        response = requests.get(f'{BASE_URL}/metrics')
        if response.status_code == 200:
            metrics = response.json()
            print(f"\nPerformance:")
            for key, value in metrics['performance'].items():
                print(f"  {key}: {value}")
            print(f"\nCache:")
            for key, value in metrics['cache'].items():
                print(f"  {key}: {value}")
    except Exception as e:
        print(f"✗ Could not fetch metrics: {e}")
    
    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*80}\n")

def main():
    """Entry point for CLI."""
    run_benchmark()


if __name__ == '__main__':
    main()
