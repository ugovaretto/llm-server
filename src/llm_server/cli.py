"""Command-line interface for SOPA LLM Server."""

import argparse
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
    
    args = parser.parse_args()
    
    # Load the model before starting the server
    load_model(args.model, use_quantization=args.quantize)
    
    # Start the server
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
