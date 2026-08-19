"""
Unit Tests for Langfuse Observability Hardening.
Verifies single-yield contextmanager contract, no-op behavior when unconfigured,
and clean propagation of business exceptions.
"""
import pytest
import os
from unittest.mock import MagicMock, patch
from backend.app.providers.observability import trace_step, NoOpSpan


def test_trace_disabled_is_noop(monkeypatch):
    """When LANGFUSE_ENABLED=false, trace_step yields a NoOpSpan without network calls."""
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    with patch("backend.app.providers.observability._langfuse_client", None):
        with patch("backend.app.providers.observability._langfuse_initialized", False):
            with trace_step("TestStep", {"meta": 1}) as span:
                assert isinstance(span, NoOpSpan)
                # Calling methods on NoOpSpan should succeed silently
                span.end()
                span.event("test")


def test_business_exception_propagates_through_trace_step(monkeypatch):
    """Exceptions raised inside the business block MUST NOT be swallowed or converted into generator errors."""
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")

    class CustomBusinessException(Exception):
        pass

    with pytest.raises(CustomBusinessException):
        with trace_step("FailingStep"):
            raise CustomBusinessException("Critical business logic failure")


def test_span_end_failure_does_not_break_business_result(monkeypatch):
    """If span.end() raises an exception, the business logic result must still succeed."""
    mock_client = MagicMock()
    mock_span = MagicMock()
    mock_span.end.side_effect = Exception("Langfuse network timeout during flush")
    mock_client.span.return_value = mock_span

    with patch("backend.app.providers.observability.get_langfuse_client", return_value=mock_client):
        result = None
        with trace_step("StepWithFailingEnd"):
            result = 42 + 58
        assert result == 100
        mock_span.end.assert_called_once()


def test_trace_missing_dependency_is_noop(monkeypatch):
    """When langfuse is enabled in env but the package is missing, tracing yields NoOpSpan and does not raise."""
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk_test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk_test")

    with patch("backend.app.providers.observability._langfuse_initialized", False):
        with patch.dict("sys.modules", {"langfuse": None}):
            with trace_step("StepMissingDependency") as span:
                assert isinstance(span, NoOpSpan)
                span.end()


def test_trace_initialization_failure_is_noop(monkeypatch):
    """When Langfuse constructor fails, tracing falls back to NoOpSpan silently."""
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk_test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk_test")

    mock_failing_langfuse = MagicMock(side_effect=RuntimeError("Connection refused by Langfuse backend"))

    with patch("backend.app.providers.observability._langfuse_initialized", False):
        with patch.dict("sys.modules", {"langfuse": MagicMock(Langfuse=mock_failing_langfuse)}):
            with trace_step("StepInitFailure") as span:
                assert isinstance(span, NoOpSpan)
                span.end()
