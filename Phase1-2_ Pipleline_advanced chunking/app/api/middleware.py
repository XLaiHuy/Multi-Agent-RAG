"""
Rate Limiting + Structured Request Logging Middleware
"""
import time
import json
import logging
import os
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# Đảm bảo thư mục logs tồn tại
os.makedirs("logs", exist_ok=True)

# Cấu hình logger ghi ra file JSON
logging.basicConfig(
    filename="logs/api.log",
    level=logging.INFO,
    format="%(message)s",
)
api_logger = logging.getLogger("api")


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log mọi request vào file logs/api.log theo định dạng JSON."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        # Chỉ log các API call thực sự (bỏ qua OPTIONS/health)
        if request.method not in ("OPTIONS",) and request.url.path != "/health":
            log_entry = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": latency_ms,
                "ip": request.client.host if request.client else "unknown",
            }
            api_logger.info(json.dumps(log_entry, ensure_ascii=False))

        return response
