# Performance Optimizations Applied

## Changes Made to main.py

### 1. **Model Loading Optimizations**
```python
# Before:
dtype=torch.float16 if torch.cuda.is_available() else torch.float32

# After:
torch_dtype=torch.float16,      # Always use FP16 on ROCm
low_cpu_mem_usage=True,         # Reduce CPU RAM usage during loading
use_cache=True                  # Enable KV cache
```

**Expected Impact:** 
- FP16 is 2x faster than FP32 on AMD GPUs
- Reduces memory bandwidth requirements
- KV cache avoids recomputing attention

### 2. **PyTorch 2.x Compilation**
```python
model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
```

**Expected Impact:**
- 1.5-2x speedup from graph optimization
- Operator fusion for better GPU utilization
- Reduced Python overhead

### 3. **Optimized Generation Parameters**

#### Streaming:
```python
generation_kwargs = {
    "use_cache": True,              # Enable KV cache (crucial!)
    "num_beams": 1,                 # Disable beam search (faster)
    "repetition_penalty": 1.1,      # Reduce repetitions
    "top_p": 0.9,                   # Nucleus sampling
    "temperature": temp if temp > 0 else None
}
```

#### Non-Streaming:
```python
with torch.inference_mode():        # Disables gradient tracking
    outputs = model.generate(...)
```

**Expected Impact:**
- `torch.inference_mode()`: 10-15% speedup
- `use_cache=True`: 2-3x faster for long sequences
- `num_beams=1`: 4x faster than beam_search=4

## Performance Comparison

### Before Optimizations:
- **Speed:** ~7 tokens/s
- **Memory:** Higher VRAM usage
- **Configuration:** FP32, no compilation, suboptimal params

### After Optimizations:
- **Expected Speed:** 15-25 tokens/s (2-3.5x improvement)
- **Memory:** Lower VRAM usage
- **Configuration:** FP16, compiled, optimized generation

## How to Test

1. **Start the optimized server:**
```bash
.venv/bin/python main.py --port 8000
```

2. **Run a quick test:**
```bash
time curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "messages": [{"role": "user", "content": "Write a short story"}],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

3. **Check metrics:**
```bash
curl http://localhost:8000/metrics
```

## Further Optimizations (If Needed)

### Option 1: 8-bit Quantization (30-50% faster)
Install: `pip install bitsandbytes`

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
```

### Option 2: 4-bit Quantization (2-3x faster, uses 1/4 VRAM)
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
```

### Option 3: Flash Attention 2 (1.5-2x faster attention)
Install: `pip install flash-attn --no-build-isolation`

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2",
    ...
)
```

### Option 4: Batch Multiple Requests
Process multiple requests together for better GPU utilization.

### Option 5: ROCm Environment Tuning
```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_ROCM_ARCH=gfx1151
export HSA_ENABLE_SDMA=0
```

## Monitoring Performance

Watch tokens/second in real-time:
```bash
watch -n 1 'curl -s http://localhost:8000/metrics | jq ".performance"'
```

## Expected Results on Radeon 8060S

| Configuration | Tokens/s | VRAM Usage |
|--------------|----------|------------|
| Original (FP32) | ~7 | ~16GB |
| Optimized (FP16 + compile) | 15-25 | ~12GB |
| + 8-bit quant | 25-35 | ~10GB |
| + 4-bit quant | 40-60 | ~6GB |

## Notes

- First request will be slower due to torch.compile warmup
- Subsequent requests benefit from compiled graph
- Performance varies with prompt length and temperature
- Lower temperature (0-0.3) is faster due to greedy decoding
