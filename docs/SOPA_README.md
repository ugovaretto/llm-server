# SOPA: Stream Optimization and Performance Accelerator

SOPA enhances the OpenAI-compatible LLM server with production-ready features for performance, reliability, and observability.

## Features

### 1. **Response Caching**
- Automatically caches responses for deterministic requests (temperature < 0.3)
- Configurable cache size (default: 100 entries) and TTL (default: 30 minutes)
- Provides significant speedup for repeated queries
- Cache key based on: messages, temperature, max_tokens, and model

### 2. **Rate Limiting**
- Per-client rate limiting to prevent abuse
- Default: 100 requests per 60-second window
- Uses client IP or API key for identification
- Returns HTTP 429 when limit exceeded

### 3. **Performance Metrics**
- Tracks total requests, tokens generated, and latency
- Separates streaming vs non-streaming request stats
- Monitors error rates
- Real-time statistics available via `/metrics` endpoint

### 4. **API Key Authentication** (Optional)
- Bearer token authentication support
- Valid keys: `sk-test-key-123`, `sk-dev-key-456`
- Optional: requests work without auth, but some admin endpoints require it
- Easy to extend with your own key management

### 5. **Health Monitoring**
- `/health` endpoint for service health checks
- `/metrics` endpoint for performance analytics
- Timestamp tracking for all operations

### 6. **Admin Operations**
- `/admin/clear-cache` endpoint to flush cache (requires API key)
- Reset cache statistics on demand

## API Endpoints

### Chat Completions (Original)
```bash
POST /v1/chat/completions
```
Enhanced with caching, rate limiting, and metrics tracking.

### Health Check
```bash
GET /health
```
Response:
```json
{
  "status": "healthy",
  "service": "SOPA LLM Server",
  "timestamp": "2025-11-11T12:00:00.000000"
}
```

### Metrics
```bash
GET /metrics
```
Response:
```json
{
  "sopa_version": "1.0.0",
  "performance": {
    "total_requests": 150,
    "total_tokens_generated": 5000,
    "average_latency_seconds": "2.34",
    "streaming_requests": 80,
    "non_streaming_requests": 70,
    "errors": 2
  },
  "cache": {
    "hits": 45,
    "misses": 105,
    "hit_rate": "30.00%",
    "cache_size": 42
  },
  "rate_limiter": {
    "active_clients": 12,
    "max_requests_per_window": 100,
    "window_seconds": 60
  },
  "timestamp": "2025-11-11T12:00:00.000000"
}
```

### Clear Cache (Admin)
```bash
POST /admin/clear-cache
Authorization: Bearer sk-test-key-123
```
Response:
```json
{
  "message": "Cache cleared successfully",
  "entries_cleared": 42
}
```

### List Models (Original)
```bash
GET /v1/models
```

## Usage Examples

### With API Key Authentication
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-test-key-123" \
  -d '{
    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.1,
    "max_tokens": 50
  }'
```

### Check Server Health
```bash
curl http://localhost:8000/health
```

### View Performance Metrics
```bash
curl http://localhost:8000/metrics
```

### Clear Cache (Admin)
```bash
curl -X POST http://localhost:8000/admin/clear-cache \
  -H "Authorization: Bearer sk-test-key-123"
```

## Configuration

Modify these values in `main.py` to customize SOPA behavior:

```python
# Response Cache
sopa_cache = ResponseCache(
    max_size=100,      # Maximum cached responses
    ttl_seconds=1800   # Cache TTL in seconds (30 min)
)

# Rate Limiter
sopa_rate_limiter = RateLimiter(
    max_requests=100,  # Max requests per window
    window_seconds=60  # Time window in seconds
)

# API Keys
VALID_API_KEYS = {"sk-test-key-123", "sk-dev-key-456"}
```

## Testing

Run the SOPA test suite:

```bash
# Start the server
.venv/bin/python main.py --port 8000

# In another terminal, run tests
.venv/bin/python test_sopa.py
```

The test suite validates:
- Health endpoint functionality
- Metrics collection and reporting
- API key authentication
- Rate limiting behavior
- Response caching and speedup
- Admin operations
- Streaming compatibility

## Performance Benefits

### Caching
- **Cache Hit Speedup**: 10-100x faster for cached responses
- **Reduced Compute**: No model inference for cached queries
- **Best For**: Frequently asked questions, deterministic queries

### Rate Limiting
- **Prevents Abuse**: Protects server from overload
- **Fair Usage**: Ensures resources distributed across clients
- **Cost Control**: Limits unexpected compute costs

### Metrics
- **Visibility**: Real-time performance monitoring
- **Debugging**: Track errors and latency issues
- **Optimization**: Identify bottlenecks and patterns

## Architecture

SOPA integrates seamlessly with the existing OpenAI-compatible server:

1. **Request Flow**:
   - Rate limiter checks client quota
   - Cache checks for existing response (non-streaming, low temp)
   - Model generates response if needed
   - Response cached for future use
   - Metrics recorded

2. **Zero Breaking Changes**: All original endpoints work identically
3. **Optional Features**: Can disable caching, rate limiting independently
4. **Production Ready**: Thread-safe, efficient, minimal overhead

## Future Enhancements

Potential additions:
- Database-backed persistent cache
- Multiple cache strategies (LRU, LFU)
- Advanced rate limiting (token-based, tiered)
- Prometheus metrics export
- Request/response logging
- Content filtering
- Multi-model support
- Batch processing

## Version

SOPA v1.0.0 - Stream Optimization and Performance Accelerator
