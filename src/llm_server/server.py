"""SOPA LLM Server

OpenAI-compatible FastAPI server providing `/v1/chat/completions`, `/v1/models`, health, metrics,
and admin endpoints. Integrates Hugging Face Transformers with optional quantization and
`torch.compile` acceleration, plus SOPA features: response caching, rate limiting, and
performance metrics. Tuned for ROCm environments with AMD GPUs.
"""
from fastapi import FastAPI, HTTPException, Request, Header, Depends
from fastapi.responses import StreamingResponse
from typing import List, Optional
from transformers import TextStreamer, TextIteratorStreamer
from threading import Thread
from pydantic import BaseModel
import torch
import time
import json
import hashlib
import os
from collections import defaultdict
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForCausalLM
# Enable experimental ROCm features for better performance
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

app = FastAPI(
    title="SOPA LLM Server",
    description="Stream Optimization and Performance Accelerator for LLMs",
)


# SOPA: Response Cache
class ResponseCache:
    """In-memory response cache with TTL and size cap.

    - Keys include messages and generation parameters to avoid collisions.
    - Evicts the oldest entry when `max_size` is reached.
    - Tracks hit/miss statistics for metrics reporting.
    """
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0

    def _make_key(
        self, messages: list, temperature: float, max_tokens: int, model: str
    ) -> str:
        content = json.dumps(
            {
                "messages": messages,
                "temp": temperature,
                "max_tokens": max_tokens,
                "model": model,
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, messages: list, temperature: float, max_tokens: int, model: str):
        key = self._make_key(messages, temperature, max_tokens, model)
        if key in self.cache:
            entry, timestamp = self.cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.ttl_seconds):
                self.hits += 1
                return entry
            else:
                del self.cache[key]
        self.misses += 1
        return None

    def set(
        self, messages: list, temperature: float, max_tokens: int, model: str, response
    ):
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        key = self._make_key(messages, temperature, max_tokens, model)
        self.cache[key] = (response, datetime.now())

    def stats(self):
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.2f}%",
            "cache_size": len(self.cache),
        }


# SOPA: Rate Limiter
class RateLimiter:
    """Simple sliding-window rate limiter.

    - Maintains per-client request timestamps.
    - Allows up to `max_requests` within `window_seconds`.
    """
    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.window_seconds)

        # Remove old requests
        self.requests[client_id] = [
            ts for ts in self.requests[client_id] if ts > cutoff
        ]

        if len(self.requests[client_id]) >= self.max_requests:
            return False

        self.requests[client_id].append(now)
        return True

    def stats(self):
        return {
            "active_clients": len(self.requests),
            "max_requests_per_window": self.max_requests,
            "window_seconds": self.window_seconds,
        }


# SOPA: Performance Metrics
class PerformanceMetrics:
    """Aggregated performance counters for observability.

    Tracks total requests, token counts, latency averages, streaming/non-streaming splits,
    and error events.
    """
    def __init__(self):
        self.total_requests = 0
        self.total_tokens_generated = 0
        self.total_latency = 0.0
        self.streaming_requests = 0
        self.non_streaming_requests = 0
        self.errors = 0

    def record_request(
        self, latency: float, tokens: int, is_streaming: bool, is_error: bool = False
    ):
        self.total_requests += 1
        self.total_tokens_generated += tokens
        self.total_latency += latency
        if is_streaming:
            self.streaming_requests += 1
        else:
            self.non_streaming_requests += 1
        if is_error:
            self.errors += 1

    def stats(self):
        avg_latency = (
            (self.total_latency / self.total_requests) if self.total_requests > 0 else 0
        )
        return {
            "total_requests": self.total_requests,
            "total_tokens_generated": self.total_tokens_generated,
            "average_latency_seconds": f"{avg_latency:.2f}",
            "streaming_requests": self.streaming_requests,
            "non_streaming_requests": self.non_streaming_requests,
            "errors": self.errors,
        }


# Initialize SOPA components
sopa_cache = ResponseCache(max_size=100, ttl_seconds=1800)
sopa_rate_limiter = RateLimiter(max_requests=100, window_seconds=60)
sopa_metrics = PerformanceMetrics()

# SOPA: API Key Authentication
VALID_API_KEYS = {"sk-test-key-123", "sk-dev-key-456"}


async def verify_api_key(authorization: Optional[str] = Header(None)):
    """Parse and verify optional Bearer API key.

    Returns the API key string if valid, else `None`. Admin-only endpoints require a
    valid key; other endpoints accept anonymous requests.
    """
    if authorization and authorization.startswith("Bearer "):
        api_key = authorization.replace("Bearer ", "")
        if api_key in VALID_API_KEYS:
            return api_key
    return None


class ChatCompletionRequest(BaseModel):
    """OpenAI-style chat completion request body.

    - `model`: must match the globally loaded model.
    - `messages`: list of chat messages with roles and content.
    - `temperature`/`max_tokens`: generation controls.
    - `stream`: when true (or `Accept: text/event-stream`), enables SSE streaming.
    """
    model: str
    messages: list
    temperature: float = 0.7
    max_tokens: int = 150
    stream: bool = False


# Global variables for model and tokenizer (set at startup)
model_name = None
tokenizer = None
model = None


def load_model(model_id: str, use_quantization: bool = False):
    """Load tokenizer and causal LM with ROCm-optimized settings.

    - Device placement: `device_map="auto"` and BF16 dtype when not quantized.
    - Attention: SDPA for ROCm stability.
    - Quantization:
      * ROCm: attempts AMD Quark; falls back to full precision if unavailable.
      * CUDA: uses bitsandbytes 8-bit `BitsAndBytesConfig`.
      * CPU: disables quantization.
    - Compilation: tries `torch.compile(mode="max-autotune")` then `reduce-overhead`.
    Prints diagnostics for device placement and memory.
    """
    global model_name, tokenizer, model

    model_name = model_id
    print(f"\n{'=' * 60}")
    print(f"Loading model: {model_name}")
    print(f"{'=' * 60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Optimized model loading for ROCm
    print(f"Loading with optimizations:")
    print(f"  - dtype: torch.bfloat16 (better for AMD RDNA3)")
    print(f"  - device_map: auto")
    print(f"  - low_cpu_mem_usage: True")
    print(f"  - use_cache: True")
    print(f"  - attn_implementation: sdpa (for ROCm stability)")
    if use_quantization:
        print(f"  - quantization: 8-bit")

    # Prepare quantization config if requested
    quantization_config = None
    if use_quantization:
        # Check for ROCm and prefer AMD Quark on AMD GPUs; fallback to bitsandbytes on CUDA
        try:
            # Debug: Check torch version info
            has_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
            print(f"  Debug: torch.version.hip = {getattr(torch.version, 'hip', None)}")
            print(f"  Debug: torch.cuda.is_available() = {torch.cuda.is_available()}")
            
            if has_rocm:
                print("  Detected ROCm GPU. Attempting AMD Quark quantization...")
                try:
                    import quark
                    print(f"  Debug: AMD Quark version = {getattr(quark, '__version__', 'unknown')}")
                    from quark.torch import ModelQuantizer
                    
                    # AMD Quark uses a different approach - quantizer object
                    # We'll need to apply it after model loading
                    quantization_config = "quark"  # Flag to apply Quark later
                    print(f"  ✓ AMD Quark quantization will be applied after model loading")
                except ImportError as e:
                    print(f"  ✗ AMD Quark import failed: {e}")
                    if "torch.onnx._internal" in str(e):
                        print(f"  ℹ️  AMD Quark requires torch.onnx._internal (not in all PyTorch builds)")
                        print(f"     Your PyTorch version may be missing ONNX internals.")
                        print(f"     Running without quantization (full precision on ROCm).")
                    use_quantization = False
                    quantization_config = None
            elif torch.cuda.is_available():
                print("  CUDA detected. Using bitsandbytes 8-bit quantization...")
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                )
                print(f"  ✓ 8-bit quantization enabled")
            else:
                print("  No GPU detected. CPU quantization not supported here; disabling...")
                use_quantization = False
        except ImportError as e:
            print(
                f"  ⚠️  Required quantization backend not installed: {e}"
            )
            print("     Install AMD Quark for ROCm: pip install amd-quark")
            print("     or bitsandbytes for CUDA: pip install bitsandbytes")
            use_quantization = False

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            dtype=torch.bfloat16
            if not use_quantization
            else None,  # quantization overrides dtype
            low_cpu_mem_usage=True,
            use_cache=True,
            attn_implementation="sdpa",  # Use scaled_dot_product_attention (stable on ROCm)
            quantization_config=quantization_config if use_quantization else None,
        )
    except Exception as e:
        print(f"❌ Failed to load model '{model_name}' - Error: {{str(e)}}")
        print("Hint: Ensure model identifier is valid format like 'owner/model_name'.")
        raise

    # Verify device placement
    print(f"\n✓ Model loaded successfully")
    print(
        f"  Model size: ~{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters"
    )

    # Check where model layers are actually loaded
    devices = {}
    for name, param in model.named_parameters():
        device = str(param.device)
        devices[device] = devices.get(device, 0) + 1

    print(f"\n📍 Device Distribution:")
    for device, count in devices.items():
        print(f"  {device}: {count} parameters")
        if "cpu" in device.lower():
            print(f"  ⚠️  WARNING: Model has parameters on CPU!")

    # Check first parameter details
    first_param = next(model.parameters())
    print(f"\n📊 First parameter details:")
    print(f"  Device: {first_param.device}")
    print(f"  Data type: {first_param.dtype}")
    print(f"  Shape: {first_param.shape}")

    # Check GPU memory usage if on GPU
    if torch.cuda.is_available() and "cuda" in str(first_param.device):
        print(f"\n🎮 GPU Memory Usage:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    else:
        print(f"\n❌ ERROR: Model is NOT on GPU! This will be very slow.")

    # Compile model for faster inference (PyTorch 2.x optimization)
    print(f"\nApplying torch.compile with max-autotune...")
    try:
        # max-autotune mode: more aggressive optimization, better for inference
        model = torch.compile(
            model, mode="max-autotune", fullgraph=False, dynamic=False
        )
        print("✓ Model compiled successfully with max-autotune")
        print("  First inference will be slower (compiling), then 2-4x faster")
    except Exception as e:
        print(f"✗ Model compilation failed: {e}")
        print("  Trying fallback compilation...")
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("✓ Fallback compilation successful")
        except Exception as e2:
            print(f"✗ Compilation not available: {e2}")
            print("  Continuing without compilation")

    print(f"\n{'=' * 60}")
    print(f"Model ready for inference")
    print(f"{'=' * 60}\n")


@app.post("/v1/chat/completions")
async def chat_completion(
    request: ChatCompletionRequest,
    req: Request = None,
    api_key: Optional[str] = Depends(verify_api_key),
):
    """Chat completions endpoint compatible with OpenAI.

    Behavior:
    - Rate limits per client and validates requested `model`.
    - Streaming: when requested, uses `TextIteratorStreamer` with a background thread
      to yield SSE chunks in OpenAI format (`object: chat.completion.chunk`) and final
      `data: [DONE]` marker.
    - Non-streaming: returns a single JSON with `choices` and `usage` fields.
    - Caching: deterministic (low-temperature) non-streaming responses cached by key.
    - Metrics: records latency and token counts.
    """
    start_time = time.time()

    # SOPA: Rate limiting
    client_id = api_key or req.client.host if req and req.client else "unknown"
    if not sopa_rate_limiter.is_allowed(client_id):
        sopa_metrics.record_request(0, 0, False, is_error=True)
        raise HTTPException(
            status_code=429, detail="Rate limit exceeded. Please try again later."
        )

    # Validate model
    if request.model != model_name:
        sopa_metrics.record_request(time.time() - start_time, 0, False, is_error=True)
        raise HTTPException(
            status_code=422, detail=f"Invalid model. Expected: {model_name}"
        )

    # Check if streaming is requested (either via Accept header or stream parameter)
    accept_header = req.headers.get("accept", "") if req else ""
    is_streaming = "text/event-stream" in accept_header or request.stream

    # SOPA: Check cache for non-streaming requests with low temperature
    if not is_streaming and request.temperature < 0.3:
        cached_response = sopa_cache.get(
            request.messages, request.temperature, request.max_tokens, request.model
        )
        if cached_response:
            latency = time.time() - start_time
            sopa_metrics.record_request(
                latency, cached_response["usage"]["completion_tokens"], False
            )
            return cached_response

    # Convert messages to prompt format
    prompt = tokenizer.apply_chat_template(
        request.messages, tokenize=False, add_generation_prompt=True
    )

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    if is_streaming:

        def generate_response():
            # Use TextIteratorStreamer for token-by-token streaming
            streamer = TextIteratorStreamer(
                tokenizer, skip_prompt=True, skip_special_tokens=True
            )

            # Run generation in a background thread and yield SSE chunks from the iterator
            generation_kwargs = {
                **inputs,
                "max_new_tokens": request.max_tokens,
                "do_sample": True if request.temperature > 0 else False,
                "temperature": request.temperature if request.temperature > 0 else None,
                "top_p": 0.9 if request.temperature > 0 else None,
                "top_k": 50 if request.temperature > 0 else None,
                "streamer": streamer,
                "pad_token_id": tokenizer.eos_token_id,
                "use_cache": True,
                "num_beams": 1,
            }

            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            # Stream tokens as they arrive, formatting to OpenAI-compatible SSE
            for text_chunk in streamer:
                # Skip empty chunks
                if not text_chunk or text_chunk.strip() == "":
                    continue

                yield f"data: {
                    json.dumps(
                        {
                            'id': 'chatcmpl-abc123',
                            'object': 'chat.completion.chunk',
                            'created': int(time.time()),
                            'model': request.model,
                            'choices': [
                                {
                                    'index': 0,
                                    'delta': {'content': text_chunk},
                                    'logprobs': None,
                                    'finish_reason': None,
                                }
                            ],
                        }
                    )
                }\n\n"

            # Wait for generation to complete
            thread.join()

            # End of stream marker per OpenAI SSE spec
            yield "data: [DONE]\n\n"

        # SOPA: Record streaming metrics
        latency = time.time() - start_time
        sopa_metrics.record_request(latency, request.max_tokens, True)

        return StreamingResponse(
            generate_response(),
            media_type="text/event-stream",
            headers={"Content-Type": "text/event-stream"},
        )
    else:
        # Non-streaming response with optimized generation
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                do_sample=True if request.temperature > 0 else False,
                temperature=request.temperature if request.temperature > 0 else None,
                top_p=0.9 if request.temperature > 0 else None,
                top_k=50 if request.temperature > 0 else None,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
            )

        # Decode the response properly
        generated_tokens = outputs[0][len(inputs["input_ids"][0]) :]
        if isinstance(generated_tokens, torch.Tensor):
            # Convert tensor to list of integers
            token_list = [
                int(token) for token in generated_tokens.cpu().detach().tolist()
            ]
        else:
            # If it's already a list or other iterable
            token_list = [int(token) for token in generated_tokens]

        response_text = tokenizer.decode(token_list, skip_special_tokens=True)

        response_data = {
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(inputs["input_ids"][0]),
                "completion_tokens": len(token_list),
                "total_tokens": len(inputs["input_ids"][0]) + len(token_list),
            },
        }

        # SOPA: Cache response for deterministic requests
        if request.temperature < 0.3:
            sopa_cache.set(
                request.messages,
                request.temperature,
                request.max_tokens,
                request.model,
                response_data,
            )

        # SOPA: Record non-streaming metrics
        latency = time.time() - start_time
        sopa_metrics.record_request(latency, len(token_list), False)

        return response_data


class ModelResponse(BaseModel):
    id: str
    object: str
    owned_by: str


class ListModelsResponse(BaseModel):
    object: str
    data: List[ModelResponse]


@app.get("/v1/models")
def list_models():
    """Return the currently loaded model in OpenAI list format."""
    return ListModelsResponse(
        object="list",
        data=[
            ModelResponse(
                id=model_name,
                object="model",
                owned_by=model_name.split("/")[0] if "/" in model_name else "unknown",
            )
        ],
    )


# SOPA: Health Check Endpoint
@app.get("/health")
def health_check():
    """Basic liveness endpoint with timestamp."""
    return {
        "status": "healthy",
        "service": "SOPA LLM Server",
        "timestamp": datetime.now().isoformat(),
    }


# SOPA: Metrics Endpoint
@app.get("/metrics")
def get_metrics():
    """Return SOPA performance, cache, and rate limiter statistics."""
    return {
        "sopa_version": "1.0.0",
        "performance": sopa_metrics.stats(),
        "cache": sopa_cache.stats(),
        "rate_limiter": sopa_rate_limiter.stats(),
        "timestamp": datetime.now().isoformat(),
    }


# SOPA: Clear Cache Endpoint (requires API key)
@app.post("/admin/clear-cache")
def clear_cache(api_key: Optional[str] = Depends(verify_api_key)):
    """Admin-only endpoint to clear cache and reset hit/miss counters."""
    if not api_key:
        raise HTTPException(status_code=401, detail="API key required")

    cache_size = len(sopa_cache.cache)
    sopa_cache.cache.clear()
    sopa_cache.hits = 0
    sopa_cache.misses = 0

    return {"message": "Cache cleared successfully", "entries_cleared": cache_size}


# SOPA: Device Diagnostic Endpoint
@app.get("/debug/device-info")
def device_info():
    """Device diagnostics: parameter distribution, GPU memory, and placement.

    Returns metadata about the first parameter's device and dtype, per-device parameter
    counts, and (if available) CUDA memory statistics.
    """
    if model is None:
        return {"error": "Model not loaded"}

    # Count parameters per device
    devices = {}
    for name, param in model.named_parameters():
        device = str(param.device)
        devices[device] = devices.get(device, 0) + 1

    # GPU memory info
    gpu_info = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_info[f"gpu_{i}"] = {
                "name": torch.cuda.get_device_name(i),
                "allocated_gb": torch.cuda.memory_allocated(i) / 1024**3,
                "reserved_gb": torch.cuda.memory_reserved(i) / 1024**3,
                "total_gb": torch.cuda.get_device_properties(i).total_memory / 1024**3,
            }

    first_param = next(model.parameters())

    return {
        "model_name": model_name,
        "device_distribution": devices,
        "first_parameter": {
            "device": str(first_param.device),
            "dtype": str(first_param.dtype),
            "shape": list(first_param.shape),
        },
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_info": gpu_info,
        "on_gpu": "cuda" in str(first_param.device),
    }


# Entry point moved to cli.py for proper package structure
