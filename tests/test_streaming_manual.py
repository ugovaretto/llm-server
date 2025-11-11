#!/usr/bin/env python3

import requests
import json
import time
import subprocess
import sys

def start_server():
    """Start the server in background"""
    print("Starting server...")
    env = os.environ.copy()
    env["PATH"] = f"/home/ugo/projects/llm-server/.venv/bin:{env.get('PATH', '')}"
    env["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "0"  # Disable problematic ROCm features
    env["HIP_VISIBLE_DEVICES"] = "-1"  # Disable GPU if problematic
    
    proc = subprocess.Popen([
        sys.executable, "main.py", "--host", "127.0.0.1", "--port", "8002"
    ], cwd="/home/ugo/projects/llm-server", env=env)
    time.sleep(5)  # Wait for server to start
    return proc

def test_streaming():
    """Test the streaming endpoint manually"""
    url = 'http://127.0.0.1:8002/v1/chat/completions'
    headers = {'Accept': 'text/event-stream'}
    data = {
        'model': 'meta-llama/Meta-Llama-3-8B-Instruct',
        'messages': [{'role': 'user', 'content': 'Say hello in exactly 3 words'}],
        'temperature': 0.7,
        'max_tokens': 20
    }
    
    print("Testing streaming endpoint...")
    try:
        response = requests.post(url, json=data, headers=headers, stream=True, timeout=30)
        print(f'Status: {response.status_code}')
        print(f'Content-Type: {response.headers.get("content-type")}')
        
        if response.status_code == 200:
            print('\n=== STREAMING RESPONSE ===')
            count = 0
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    print(f'Line {count}: {line_str}')
                    count += 1
                    if count > 10:  # Show first 10 lines
                        break
            print('=== END STREAMING ===\n')
        else:
            print('Error response:', response.text)
            
    except Exception as e:
        print(f'Error: {e}')

if __name__ == "__main__":
    import os
    server_proc = None
    try:
        server_proc = start_server()
        test_streaming()
    finally:
        if server_proc:
            server_proc.terminate()
            server_proc.wait()
