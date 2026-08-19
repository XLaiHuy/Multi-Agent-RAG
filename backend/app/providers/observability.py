"""
Langfuse Observability & Tracing Adapter.
Provides optional tracing for Planner, Retrieval, Reranker, Critic, Generator, and Verifier.
Ensures zero runtime impact when Langfuse is unconfigured or disabled.
Supports content redaction for private legal agreements (LANGFUSE_REDACT_CONTENT).
"""
import os
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager

from backend.app.core.config import get_settings

logger = logging.getLogger("observability")

_langfuse_client = None
_langfuse_initialized = False


def get_langfuse_client():
    global _langfuse_client, _langfuse_initialized
    if _langfuse_initialized:
        return _langfuse_client

    _langfuse_initialized = True
    settings = get_settings()

    enabled = os.environ.get("LANGFUSE_ENABLED", "false").lower() in ["true", "1", "yes"]
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not enabled or not public_key or not secret_key:
        _langfuse_client = None
        return None

    try:
        from langfuse import Langfuse
        _langfuse_client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        logger.info("[Observability] Langfuse tracing initialized successfully.")
    except Exception as e:
        logger.warning(f"[Observability] Langfuse initialization skipped: {e}")
        _langfuse_client = None

    return _langfuse_client


class NoOpSpan:
    """Safe no-op fallback when Langfuse is inactive."""
    def end(self, *args, **kwargs):
        pass
    def update(self, *args, **kwargs):
        pass
    def event(self, *args, **kwargs):
        pass
    def score(self, *args, **kwargs):
        pass


@contextmanager
def trace_step(name: str, metadata: Optional[Dict[str, Any]] = None, input_data: Optional[Any] = None):
    """Context manager for tracing an execution step."""
    client = get_langfuse_client()
    if not client:
        yield NoOpSpan()
        return

    redact = os.environ.get("LANGFUSE_REDACT_CONTENT", "true").lower() in ["true", "1", "yes"]
    safe_input = "[REDACTED_FOR_PRIVACY]" if (redact and isinstance(input_data, str)) else input_data

    try:
        span = client.span(name=name, metadata=metadata or {}, input=safe_input)
        yield span
        span.end()
    except Exception as e:
        logger.debug(f"[Observability] Tracing span failed silently: {e}")
        yield NoOpSpan()
