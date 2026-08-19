"""
Main FastAPI Application Entrypoint.
Registers API routers, CORS, global exception handlers, and persistence lifecycle.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.core.config import get_settings
from backend.app.persistence.database import init_database
from backend.app.api.auth_routes import auth_router
from backend.app.api.document_routes import document_router
from backend.app.api.qa_routes import qa_router, conversation_router
from backend.app.api.compare_routes import compare_router
from backend.app.api.risk_routes import risk_router

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
)
logger = logging.getLogger("app_main")

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application Startup and Shutdown Lifecycle."""
    logger.info(f"Initializing {settings.app_title} v{settings.app_version} (Env: {settings.environment})...")
    init_database()
    yield
    logger.info("Application shutdown complete.")


app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description="Role-Aware Enterprise Contract Intelligence Platform built around Adaptive Multi-Agent RAG.",
    lifespan=lifespan,
)

# CORS Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled Exception on {request.url.path}: {exc}", exc_info=True)
    is_prod = settings.environment.lower() in ["production", "prod"]
    content = {"error": "Internal Server Error"}
    if not is_prod:
        content["detail"] = str(exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=content,
    )


# Health & Readiness Probes
@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "ok",
        "app": settings.app_title,
        "version": settings.app_version,
        "environment": settings.environment,
    }


@app.get("/ready", tags=["Health"])
def readiness_check():
    """Readiness probe: validates database connectivity with a lightweight probe."""
    try:
        from sqlalchemy import text
        from backend.app.persistence.database import engine
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ready", "database": "connected"}
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unavailable", "database": "disconnected"},
        )


# Include Versioned API Routers
API_V1_PREFIX = "/api/v1"
app.include_router(auth_router, prefix=API_V1_PREFIX)
app.include_router(document_router, prefix=API_V1_PREFIX)
app.include_router(qa_router, prefix=API_V1_PREFIX)
app.include_router(conversation_router, prefix=API_V1_PREFIX)
app.include_router(compare_router, prefix=API_V1_PREFIX)
app.include_router(risk_router, prefix=API_V1_PREFIX)
