import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import llm_server.server as server
import os
import pytest

@pytest.fixture(scope="session", autouse=True)
def setup_model():
    model_id = os.getenv("SOPA_TEST_MODEL", "sshleifer/tiny-gpt2")
    server.load_model(model_id, use_quantization=False, device_preference=os.getenv("SOPA_DEVICE", "cpu"))
    yield
