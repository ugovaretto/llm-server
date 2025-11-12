import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import llm_server.server as server

class _TestStreamer:
    def __init__(self, tokenizer, skip_prompt=True, skip_special_tokens=True):
        self._q = []
        self._closed = False
    def put(self, text):
        self._q.append(text)
    def end(self):
        self._closed = True
    def __iter__(self):
        i = 0
        while True:
            if i < len(self._q):
                item = self._q[i]
                i += 1
                yield item
            elif self._closed:
                break

server.TextIteratorStreamer = _TestStreamer
