"""
Centralized Gemini / LLM API Gateway.
Provides concurrency limits, RPM/TPM rate limiting, exponential backoff with jitter,
circuit breaker protection, execution budgeting, and structured observability metrics.
"""
import time
import random
import logging
import threading
from typing import Dict, Any, Optional, Iterator, Type
from google import genai
from google.genai import types
from google.genai.errors import APIError

from backend.app.core.config import Settings, get_settings
from backend.app.providers.interfaces import LLMProvider

logger = logging.getLogger("gemini_gateway")


class CircuitBreakerOpenException(Exception):
    """Raised when circuit breaker is open due to consecutive failures."""
    pass


class BudgetExceededException(Exception):
    """Raised when query execution budget is exhausted."""
    pass


class GeminiAPIGateway(LLMProvider):
    """
    Centralized Gateway for Gemini API interactions with resilience patterns.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.api_key = self.settings.gemini_api_key
        
        # Initialize Google GenAI client if key exists
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None
        
        # Concurrency Semaphore
        self._concurrency_semaphore = threading.Semaphore(self.settings.gemini_concurrency_limit)
        
        # Circuit Breaker state
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0
        self._failure_threshold = 5
        self._cool_down_seconds = 30.0
        self._state_lock = threading.Lock()

        # Simple Rate Limiter (Token bucket for RPM)
        self._rpm_window_start = time.time()
        self._rpm_request_count = 0
        self._rpm_lock = threading.Lock()

        # Metrics Tracker
        self.metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "throttled_429_count": 0,
            "total_tokens": 0,
            "calls_by_model": {},
        }
        self._metrics_lock = threading.Lock()

    def _resolve_model_name(self, model_type: str) -> str:
        """Map logical task role to configured model ID."""
        mapping = {
            "planner": self.settings.planner_model,
            "critic": self.settings.critic_model,
            "rewrite": self.settings.rewrite_model,
            "verifier": self.settings.verifier_model,
            "generation": self.settings.generation_model,
            "ocr": self.settings.ocr_model,
        }
        return mapping.get(model_type, self.settings.generation_model)

    def _check_circuit_breaker(self):
        """Verify circuit breaker state before initiating API request."""
        with self._state_lock:
            if self._circuit_open_until > 0:
                if time.time() < self._circuit_open_until:
                    raise CircuitBreakerOpenException(
                        f"Circuit breaker is OPEN. Fast failing requests until {self._circuit_open_until - time.time():.1f}s remaining."
                    )
                else:
                    # Half-open: allow probe request
                    self._circuit_open_until = 0.0
                    self._consecutive_failures = 0
                    logger.info("[Gateway] Circuit breaker entering HALF-OPEN state.")

    def _record_success(self, model_name: str):
        with self._state_lock:
            self._consecutive_failures = 0
            self._circuit_open_until = 0.0
        with self._metrics_lock:
            self.metrics["total_calls"] += 1
            self.metrics["successful_calls"] += 1
            self.metrics["calls_by_model"][model_name] = (
                self.metrics["calls_by_model"].get(model_name, 0) + 1
            )

    def _record_failure(self, error: Exception, model_name: str):
        with self._metrics_lock:
            self.metrics["total_calls"] += 1
            self.metrics["failed_calls"] += 1
        with self._state_lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._failure_threshold:
                self._circuit_open_until = time.time() + self._cool_down_seconds
                logger.error(
                    f"[Gateway] Circuit breaker TRIPPED! Opened for {self._cool_down_seconds}s due to {self._consecutive_failures} consecutive failures: {error}"
                )

    def _throttle_rpm(self):
        """Enforce RPM limits with sliding time window."""
        with self._rpm_lock:
            now = time.time()
            if now - self._rpm_window_start > 60.0:
                self._rpm_window_start = now
                self._rpm_request_count = 0
            
            if self._rpm_request_count >= self.settings.gemini_rpm_limit:
                sleep_duration = 60.0 - (now - self._rpm_window_start) + 0.1
                if sleep_duration > 0:
                    logger.warning(f"[Gateway] RPM limit reached ({self.settings.gemini_rpm_limit}). Pausing for {sleep_duration:.2f}s...")
                    time.sleep(sleep_duration)
                self._rpm_window_start = time.time()
                self._rpm_request_count = 0

            self._rpm_request_count += 1

    def _execute_with_resilience(self, call_fn, model_name: str):
        """
        Executes API call wrapped in concurrency limits, exponential backoff with jitter,
        and selective error retry.
        """
        # Instant check for placeholder / unconfigured API key
        raw_key = (self.settings.gemini_api_key or "").strip()
        if not raw_key or raw_key == "your_gemini_api_key_here":
            raise RuntimeError("Khóa GEMINI_API_KEY chưa được cấu hình. Vui lòng mở file .env và điền khóa API Gemini hợp lệ.")

        self._check_circuit_breaker()
        self._throttle_rpm()

        retries = self.settings.gemini_max_retries
        base_delay = 1.0

        for attempt in range(retries + 1):
            try:
                with self._concurrency_semaphore:
                    result = call_fn()
                    self._record_success(model_name)
                    return result
            except APIError as e:
                status_code = getattr(e, "code", None)
                # Permanent non-retryable errors
                if status_code in [400, 401, 403, 404]:
                    self._record_failure(e, model_name)
                    logger.error(f"[Gateway] Non-retryable API error {status_code}: {e}")
                    raise e
                
                # 429 Rate limit or 5xx Server error -> Retry with jittered backoff
                if status_code == 429:
                    with self._metrics_lock:
                        self.metrics["throttled_429_count"] += 1
                
            except Exception as e:
                err_str = str(e).lower()
                # Fast fail on invalid API key, auth failure, or unconfigured quota
                if "api_key_invalid" in err_str or "api key not valid" in err_str or "invalid api key" in err_str or "400" in err_str or "403" in err_str:
                    self._record_failure(e, model_name)
                    logger.error(f"[Gateway] Immediate auth/key failure: {e}")
                    raise RuntimeError("Khóa GEMINI_API_KEY trong file .env không hợp lệ hoặc chưa được kích hoạt. Vui lòng kiểm tra lại file .env.") from e

                if attempt == retries:
                    self._record_failure(e, model_name)
                    raise e

                delay = base_delay * (2 ** attempt) + random.uniform(0.1, 0.5)
                logger.warning(
                    f"[Gateway] Connection/Transient error (attempt {attempt + 1}/{retries + 1}): {e}. Retrying in {delay:.2f}s..."
                )
                time.sleep(delay)

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        model_type: str = "generation",
        temperature: float = 0.2,
        max_output_tokens: Optional[int] = None,
    ) -> str:
        """Generate text response using resolved model."""
        if not self.client:
            raise RuntimeError("Gemini API key is not configured.")

        model_name = self._resolve_model_name(model_type)
        
        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
            max_output_tokens=max_output_tokens,
        )

        def _call():
            response = self.client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            return response.text if response and response.text else ""

        return self._execute_with_resilience(_call, model_name)

    def generate_stream(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        model_type: str = "generation",
        temperature: float = 0.2,
    ) -> Iterator[str]:
        """Stream response tokens using resolved model."""
        if not self.client:
            raise RuntimeError("Gemini API key is not configured.")

        model_name = self._resolve_model_name(model_type)
        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
        )

        self._check_circuit_breaker()
        self._throttle_rpm()

        try:
            with self._concurrency_semaphore:
                response = self.client.models.generate_content_stream(
                    model=model_name, contents=prompt, config=config
                )
                for chunk in response:
                    if chunk.text:
                        yield chunk.text
                self._record_success(model_name)
        except Exception as e:
            self._record_failure(e, model_name)
            logger.error(f"[Gateway] Streaming generation error: {e}")
            raise e

    def generate_structured(
        self,
        prompt: str,
        schema: Any,
        system_instruction: Optional[str] = None,
        model_type: str = "planner",
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Generate structured JSON conforming to a Pydantic schema."""
        if not self.client:
            raise RuntimeError("Gemini API key is not configured.")

        model_name = self._resolve_model_name(model_type)
        
        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=schema,
        )

        def _call():
            import json
            response = self.client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            if response and response.text:
                return json.loads(response.text)
            return {}

        return self._execute_with_resilience(_call, model_name)


_gateway_instance: Optional[GeminiAPIGateway] = None


def get_gemini_gateway() -> GeminiAPIGateway:
    global _gateway_instance
    if _gateway_instance is None:
        _gateway_instance = GeminiAPIGateway()
    return _gateway_instance
