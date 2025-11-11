# SOPA LLM Server

**Stream Optimization and Performance Accelerator** - An OpenAI-compatible LLM server with advanced caching, rate limiting, and performance optimizations for AMD ROCm GPUs.

## Features

- ✅ **OpenAI API Compatible** - Drop-in replacement for OpenAI chat completions API
- 🚀 **Smart Response Caching** - 560x speedup on repeated queries with TTL and LRU eviction
- 🛡️ **Rate Limiting** - Per-client request throttling (100 req/min default)
- 📊 **Performance Metrics** - Real-time tracking of latency, tokens/sec, and cache hit rates
- 🔐 **API Key Authentication** - Optional Bearer token authentication
- 💾 **Memory Optimized** - BFloat16 precision and torch.compile with max-autotune
- 🎮 **ROCm Ready** - Optimized for AMD GPUs with experimental features enabled
- 📡 **Streaming Support** - Real-time token streaming with proper SSE formatting

## Quick Start

### Installation with UV (Recommended)

```bash
# Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
cd llm-server
uv sync

# Run the server
uv run llm-server --port 8000

# Or run as module
uv run python -m llm_server --port 8000
```

### Installation with pip

```bash
# Create virtual environment
python3.13 -m venv .venv
source .venv/bin/activate

# Install PyTorch with ROCm support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.0

# Install the package
pip install -e .
```

## Usage

### Start the Server

```bash
# Using UV (recommended)
uv run llm-server --port 8000

# Or with options
uv run llm-server --host 0.0.0.0 --port 8000 --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Available Commands

```bash
# Start server
uv run llm-server [--host HOST] [--port PORT] [--model MODEL]

# Run benchmark tests
uv run llm-benchmark

# Profile model performance
uv run llm-profile [model-name]

# Run tests
uv run pytest
```

### Using the API

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="optional-key"  # Set if authentication is enabled
)

# Non-streaming
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
    max_tokens=100
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## API Endpoints

### Main Endpoints
- `POST /v1/chat/completions` - OpenAI-compatible chat completions (streaming & non-streaming)
- `GET /health` - Health check endpoint
- `GET /metrics` - Performance metrics (requests, tokens, cache stats)
- `GET /debug/device-info` - GPU and model placement diagnostics

### Admin Endpoints
- `POST /admin/clear-cache` - Clear response cache (requires API key)

## Configuration

### Environment Variables

```bash
# Enable experimental ROCm features (automatically set)
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
```

### Command-Line Options

```
--host HOST         Host to bind to (default: 0.0.0.0)
--port PORT         Port to bind to (default: 8000)
--model MODEL       Model to load (default: meta-llama/Meta-Llama-3-8B-Instruct)
--quantize         Enable 8-bit quantization (requires compatible library)
```

## Performance

### Current Performance (AMD Radeon 8060S APU)
- Non-streaming: ~7.5 tokens/s
- Streaming: ~6.5 tokens/s
- Cache hit: 560x faster

### Optimizations Applied
- BFloat16 precision (better for AMD RDNA3)
- torch.compile with max-autotune mode
- SDPA attention implementation
- Response caching with smart eviction
- Optimized generation parameters

### Future Optimizations
- GPTQ 4-bit quantization for 2-3x speedup
- Flash Attention 2 when stable on ROCm
- Multi-model support with model registry

## Development

### Project Structure

```
llm-server/
├── src/llm_server/      # Main package
│   ├── server.py        # FastAPI server and model logic
│   ├── cli.py          # Command-line interface
│   └── __main__.py     # Module entry point
├── scripts/            # Utility scripts
│   ├── benchmark.py    # Performance benchmarking
│   └── profiler.py     # Model profiling
├── tests/             # Test suite
├── docs/              # Documentation
└── pyproject.toml     # Project configuration
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_sopa.py

# Verbose output
uv run pytest -v
```

### Adding Dependencies

```bash
# Add runtime dependency
uv add package-name

# Add dev dependency
uv add --dev package-name

# Add optional dependency
uv add --optional quantization package-name
```

## Hardware Requirements

### Minimum
- Python 3.13+
- 16GB RAM
- CPU with AVX2 support

### Recommended
- AMD GPU with ROCm 7.0+ support
- 32GB+ RAM (or shared memory for APU)
- gfx1100+ architecture (RDNA3)

### Tested On
- AMD Radeon 8060S (gfx1151) APU
- ROCm 7.0.2
- PyTorch 2.10.0+rocm7.0
- Fedora 42 (Linux 6.17.7)

## Documentation

- [SOPA Features Guide](docs/SOPA_README.md) - Detailed SOPA feature documentation
- [Optimization Guide](docs/OPTIMIZATION_GUIDE.md) - Performance tuning guide
- [Performance Analysis](docs/PERFORMANCE_OPTIMIZATIONS.md) - Applied optimizations

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please open an issue or PR on GitHub.

## Support

For issues, questions, or feature requests, please open an issue on GitHub.
