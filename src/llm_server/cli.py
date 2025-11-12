"""Command-line interface for SOPA LLM Server."""

import argparse
import os
import torch
import uvicorn
from .server import load_model, app


def print_hardware_info():
    """Print GPU/hardware information."""
    print("\n" + "="*60)
    print("SOPA LLM Server - Hardware Check")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA/ROCm available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Device capability: {torch.cuda.get_device_capability(0)}")
    else:
        print("⚠️  WARNING: No GPU detected! Running on CPU will be very slow.")
    print("="*60 + "\n")


def main():
    """Main entry point for the CLI."""
    print_hardware_info()
    
    parser = argparse.ArgumentParser(
        description="SOPA LLM Server - Stream Optimization and Performance Accelerator"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0", 
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="meta-llama/Meta-Llama-3-8B-Instruct", 
        help="Model to load (default: meta-llama/Meta-Llama-3-8B-Instruct; must be valid HF Hub identifier, e.g., meta-llama/Meta-Llama-3-8B-Instruct)",
    )
    parser.add_argument(
        "--quantize", 
        action="store_true", 
        help="Use 8-bit quantization for 2-3x speedup (requires bitsandbytes)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Default temperature for sampling (0 for greedy)"
    )
    parser.add_argument(
        "--top-p",
        dest="top_p",
        type=float,
        default=0.9,
        help="Default nucleus sampling probability (used when temperature>0)"
    )
    parser.add_argument(
        "--top-k",
        dest="top_k",
        type=int,
        default=50,
        help="Default top-k sampling size (used when temperature>0)"
    )
    parser.add_argument(
        "--repetition-penalty",
        dest="repetition_penalty",
        type=float,
        default=None,
        help="Default repetition penalty (>1 discourages repeats)"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "gpu", "cpu"],
        default="auto",
        help="Select device to use for inference (auto/gpu/cpu)"
    )
    parser.add_argument(
        "--attn-impl",
        dest="attn_impl",
        type=str,
        default=None,
        help="Attention implementation to use (e.g., sdpa)"
    )
    parser.add_argument(
        "--use-cache",
        dest="use_cache",
        choices=["true", "false"],
        default=None,
        help="Enable or disable KV cache (pass explicitly if needed)"
    )
    
    args = parser.parse_args()
    
    # Load the model before starting the server
    use_cache = None if args.use_cache is None else (args.use_cache == "true")
    load_model(
        args.model,
        use_quantization=args.quantize,
        device_preference=args.device,
        use_cache=use_cache,
        attn_impl=args.attn_impl,
    )
    # Set defaults via environment consumed by server when request omits these fields
    os.environ.setdefault("SOPA_DEFAULT_TEMPERATURE", str(args.temperature))
    os.environ.setdefault("SOPA_DEFAULT_TOP_P", str(args.top_p))
    os.environ.setdefault("SOPA_DEFAULT_TOP_K", str(args.top_k))
    if args.repetition_penalty is not None:
        os.environ.setdefault("SOPA_DEFAULT_REPETITION_PENALTY", str(args.repetition_penalty))
    os.environ.setdefault("SOPA_DEVICE", args.device)
    
    # Start the server
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
