#!/usr/bin/env python3
"""Quick streaming test that uses the existing test server in test_llm_server.py"""

import requests
import json

def test_streaming():
    """Test the streaming endpoint"""
    url = 'http://127.0.0.1:8000/v1/chat/completions'
    headers = {'Accept': 'text/event-stream'}
    data = {
        'model': 'meta-llama/Meta-Llama-3-8B-Instruct',
        'messages': [{'role': 'user', 'content': 'Hello'}],
        'temperature': 0.7,
        'max_tokens': 10
    }
    
    print("Testing streaming endpoint at http://127.0.0.1:8000...")
    print(f"Request: {json.dumps(data, indent=2)}")
    print(f"Headers: {headers}\n")
    
    try:
        response = requests.post(url, json=data, headers=headers, stream=True, timeout=10)
        print(f'Status Code: {response.status_code}')
        print(f'Content-Type: {response.headers.get("content-type")}')
        
        if response.status_code == 200:
            print('\n=== STREAMING RESPONSE ===')
            chunk_count = 0
            full_content = ""
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    print(f'Chunk {chunk_count}: {line_str[:100]}...' if len(line_str) > 100 else f'Chunk {chunk_count}: {line_str}')
                    
                    # Parse the data if it starts with "data: "
                    if line_str.startswith('data: '):
                        try:
                            data_json = json.loads(line_str[6:])
                            if 'choices' in data_json and len(data_json['choices']) > 0:
                                delta = data_json['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    full_content += delta['content']
                        except json.JSONDecodeError:
                            pass
                    
                    chunk_count += 1
            
            print(f'\n=== END STREAMING ===')
            print(f'Total chunks received: {chunk_count}')
            print(f'Full content: "{full_content}"')
            print(f'\n✓ Streaming works! Received {chunk_count} chunks.')
            return True
        else:
            print(f'✗ Error: Status code {response.status_code}')
            print(f'Response: {response.text}')
            return False
            
    except requests.exceptions.ConnectionError as e:
        print(f'✗ Connection Error: {e}')
        print('\nMake sure the server is running on port 8000.')
        print('You can start it with: .venv/bin/python main.py --port 8000')
        return False
    except Exception as e:
        print(f'✗ Error: {e}')
        return False

if __name__ == "__main__":
    success = test_streaming()
    exit(0 if success else 1)
