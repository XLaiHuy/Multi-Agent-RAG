import os
from cachetools import TTLCache
from app.graph.agentic_rag import app as agentic_rag_app
from slowapi import Limiter
from slowapi.util import get_remote_address

# Khởi tạo in-memory cache lưu được 100 câu hỏi trong 1 giờ (3600s)
# Dùng để trả lời siêu tốc nếu người dùng hỏi lại câu cũ
response_cache = TTLCache(maxsize=100, ttl=3600)

# Khởi tạo Rate Limiter chặn spam
limiter = Limiter(key_func=get_remote_address)

def get_rag_app():
    """
    Dependency Injection để lấy instance của AgenticRAG graph.
    FastAPI sẽ gọi hàm này mỗi khi có request tới endpoint.
    """
    return agentic_rag_app

def get_cache():
    return response_cache
