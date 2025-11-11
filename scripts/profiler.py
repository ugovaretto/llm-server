#!/usr/bin/env python3
"""Profile the model to find performance bottlenecks"""

import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

def profile_model(model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
    print(f"\n{'='*80}")
    print("MODEL PROFILING - Finding Performance Bottlenecks")
    print(f"{'='*80}\n")
    
    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch.float16
    )
    
    print(f"✓ Model loaded")
    
    # Check device placement
    print("\n📍 Device Placement:")
    devices = {}
    for name, param in model.named_parameters():
        device = str(param.device)
        devices[device] = devices.get(device, 0) + 1
    
    for device, count in devices.items():
        print(f"  {device}: {count} parameters")
        if 'cpu' in device.lower():
            print(f"  ⚠️  PROBLEM: Model has parameters on CPU - this is why it's slow!")
    
    # Check memory
    if torch.cuda.is_available():
        print(f"\n🎮 GPU Memory:")
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
    
    # Test inference speed
    print(f"\n⚡ Running inference test...")
    prompt = "Write a short poem about AI."
    messages = [{"role": "user", "content": prompt}]
    
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    # Warmup
    print("  Warming up...")
    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    
    # Actual test
    print("  Generating 50 tokens...")
    start = time.time()
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False, use_cache=True)
    end = time.time()
    
    duration = end - start
    tokens_generated = len(outputs[0]) - len(inputs["input_ids"][0])
    tokens_per_sec = tokens_generated / duration
    
    print(f"\n📊 Results:")
    print(f"  Tokens generated: {tokens_generated}")
    print(f"  Time: {duration:.2f}s")
    print(f"  Speed: {tokens_per_sec:.2f} tokens/second")
    
    if tokens_per_sec < 10:
        print(f"\n❌ SLOW! Expected 15-30 tokens/s on GPU")
        print(f"   Likely issues:")
        if any('cpu' in d.lower() for d in devices.keys()):
            print(f"   - Model is on CPU")
        print(f"   - Shared memory bottleneck (APU)")
        print(f"   - Need quantization")
    elif tokens_per_sec < 20:
        print(f"\n⚠️  Slower than expected, but using GPU")
        print(f"   Recommendations: quantization, BFloat16")
    else:
        print(f"\n✓ Good performance!")
    
    print(f"\n{'='*80}\n")

def main():
    """Entry point for CLI."""
    model_name = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Meta-Llama-3-8B-Instruct"
    profile_model(model_name)


if __name__ == "__main__":
    main()
