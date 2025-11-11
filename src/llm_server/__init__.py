"""SOPA LLM Server - Stream Optimization and Performance Accelerator for LLMs."""

__version__ = "0.4.0"
__author__ = "Ugo"
__description__ = "OpenAI-compatible LLM server with advanced caching, rate limiting, and performance optimizations"

from .server import app, load_model

__all__ = ["app", "load_model"]
