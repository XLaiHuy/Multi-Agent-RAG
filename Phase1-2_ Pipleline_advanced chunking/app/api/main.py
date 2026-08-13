from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.routes import router as api_router
from app.api.auth import auth_router
from app.api.upload import upload_router
from app.api.middleware import LoggingMiddleware
from app.api.deps import limiter

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Tải (Preload) các mô hình AI vào RAM trước khi nhận request
    print("[Lifespan] Preloading AI Models (Zero-latency start)...")
    from app.graph.agentic_rag import get_generator, get_vector_retriever, get_hybrid_retriever
    get_generator()
    get_vector_retriever()
    get_hybrid_retriever()
    print("[Lifespan] AI Models loaded successfully. System ready!")
    yield
    print("[Lifespan] Shutting down...")

app = FastAPI(
    title="Agentic RAG API",
    description="Enterprise Document Intelligence Platform - Module 07",
    version="1.0.0",
    lifespan=lifespan
)

# Gắn Rate Limiter (slowapi)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Thêm LoggingMiddleware để ghi log các request vào logs/api.log
app.add_middleware(LoggingMiddleware)

# Cấu hình CORS để frontend React/NextJS có thể gọi API mà không bị chặn
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong thực tế nên đổi thành domain cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api")
app.include_router(upload_router, prefix="/api")
app.include_router(api_router, prefix="/api")

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok", "message": "Agentic RAG API is running"}
