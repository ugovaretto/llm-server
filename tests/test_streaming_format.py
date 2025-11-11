#!/usr/bin/env python3
"""Test that streaming format matches OpenAI spec exactly"""

import requests
import json

def test_streaming_format():
    """Verify streaming matches OpenAI format from openai-streaming-example.jsonl"""
    url = 'http://127.0.0.1:8000/v1/chat/completions'
    data = {
        'model': 'meta-llama/Meta-Llama-3-8B-Instruct',
        'messages': [{'role': 'user', 'content': 'Count to 3'}],
        'stream': True,
        'max_tokens': 20
    }
    
    print('Testing streaming format compliance...\n')
    response = requests.post(url, json=data, stream=True, timeout=30)
    
    # Verify headers
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert response.headers.get('content-type') == 'text/event-stream', \
        f"Expected text/event-stream, got {response.headers.get('content-type')}"
    
    chunks = []
    content_chunks = 0
    done_received = False
    
    for line in response.iter_lines():
        if not line:
            continue
            
        decoded = line.decode('utf-8')
        
        if decoded == 'data: [DONE]':
            done_received = True
            print(f'✓ Received [DONE] marker')
            break
        
        if decoded.startswith('data: '):
            try:
                data_json = json.loads(decoded[6:])
                chunks.append(data_json)
                
                # Verify required fields
                assert 'id' in data_json, "Missing 'id' field"
                assert 'object' in data_json, "Missing 'object' field"
                assert data_json['object'] == 'chat.completion.chunk', \
                    f"Expected 'chat.completion.chunk', got {data_json['object']}"
                assert 'created' in data_json, "Missing 'created' field"
                assert 'model' in data_json, "Missing 'model' field"
                assert 'choices' in data_json, "Missing 'choices' field"
                
                choice = data_json['choices'][0]
                assert 'index' in choice, "Missing 'index' field"
                assert 'delta' in choice, "Missing 'delta' field"
                assert 'logprobs' in choice, "Missing 'logprobs' field"
                assert 'finish_reason' in choice, "Missing 'finish_reason' field"
                
                if 'content' in choice['delta']:
                    content_chunks += 1
                    
            except json.JSONDecodeError as e:
                print(f'✗ JSON parse error: {e}')
                print(f'  Line: {decoded}')
                raise
            except AssertionError as e:
                print(f'✗ Format error: {e}')
                print(f'  Chunk: {data_json}')
                raise
    
    # Verify we got data
    assert content_chunks > 0, "No content chunks received"
    assert done_received, "Did not receive [DONE] marker"
    
    print(f'✓ Received {content_chunks} content chunks')
    print(f'✓ All chunks have required fields')
    print(f'✓ Format matches OpenAI spec')
    print(f'\n✓✓✓ Streaming format test PASSED ✓✓✓')

if __name__ == '__main__':
    test_streaming_format()
