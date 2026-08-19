"""
Application Configuration and Settings Module.
Derives retrieval baseline defaults from backend.app.core.retrieval_defaults.
"""
import os
import sys
from pathlib import Path
from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

from backend.app.core.retrieval_defaults import (
    DENSE_MODEL_PRODUCTION_DEFAULT,
    DENSE_DIMENSION_PRODUCTION_DEFAULT,
    DENSE_MODEL_EVALUATION_SELECTED,
    DENSE_DIMENSION_EVALUATION_SELECTED,
    CHILD_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE,
    PARENT_CHUNK_OVERLAP,
    RERANKER_MODEL_DEFAULT,
    RERANKER_MAX_SEQ_LENGTH,
)

load_dotenv()


class Settings(BaseSettings):
    # Environment
    environment: str = Field(default="development", alias="ENVIRONMENT")
    debug: bool = Field(default=False, alias="DEBUG")
    app_title: str = "Enterprise Contract Intelligence Platform"
    app_version: str = "3.5.1"

    # Security & JWT
    jwt_secret_key: str = Field(default="dev_insecure_jwt_secret_key_change_in_production_1234567890", alias="JWT_SECRET_KEY")
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = Field(default=60 * 24, alias="ACCESS_TOKEN_EXPIRE_MINUTES")

    # API Keys
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")

    # Model Routing - Decoupled per Task
    llm_provider: str = Field(default="gemini", alias="LLM_PROVIDER")
    planner_model: str = Field(default="gemini-flash-latest", alias="PLANNER_MODEL")
    critic_model: str = Field(default="gemini-flash-latest", alias="CRITIC_MODEL")
    rewrite_model: str = Field(default="gemini-flash-latest", alias="REWRITE_MODEL")
    verifier_model: str = Field(default="gemini-flash-latest", alias="VERIFIER_MODEL")
    generation_model: str = Field(default="gemini-flash-latest", alias="GENERATION_MODEL")
    ocr_model: str = Field(default="gemini-flash-latest", alias="OCR_MODEL")

    # Local Ollama fallback (if configured)
    ollama_base_url: str = Field(default="http://localhost:11434/v1", alias="OLLAMA_BASE_URL")

    # Embedding Settings (Derived from retrieval_defaults)
    embedding_provider: str = Field(default="local", alias="EMBEDDING_PROVIDER")
    gemini_embedding_model: str = Field(default="text-embedding-004", alias="GEMINI_EMBEDDING_MODEL")
    local_embedding_model: str = Field(default=DENSE_MODEL_PRODUCTION_DEFAULT, alias="LOCAL_EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=DENSE_DIMENSION_PRODUCTION_DEFAULT, alias="EMBEDDING_DIMENSION")

    # Optional high-accuracy evaluation model override
    evaluation_dense_model: str = DENSE_MODEL_EVALUATION_SELECTED
    evaluation_dense_dimension: int = DENSE_DIMENSION_EVALUATION_SELECTED

    # Reranker Settings (Derived from retrieval_defaults)
    enable_reranker: bool = Field(default=True, alias="ENABLE_RERANKER")
    reranker_model: str = Field(default=RERANKER_MODEL_DEFAULT, alias="RERANKER_MODEL")
    reranker_max_seq_length: int = Field(default=RERANKER_MAX_SEQ_LENGTH, alias="RERANKER_MAX_SEQ_LENGTH")
    reranker_strict_mode: bool = Field(default=False, alias="RERANKER_STRICT_MODE")

    # Storage Paths & URLs
    database_url: str = Field(default="sqlite:///./data/contracts.db", alias="DATABASE_URL")
    chroma_path: str = Field(default="./data/chroma", alias="CHROMA_PATH")
    chroma_collection: str = Field(default="enterprise_contracts_v2", alias="CHROMA_COLLECTION")
    storage_dir: str = Field(default="./data/storage", alias="STORAGE_DIR")
    redis_url: Optional[str] = Field(default=None, alias="REDIS_URL")

    # Gemini Gateway Rate Limits & Budget
    gemini_rpm_limit: int = Field(default=60, alias="GEMINI_RPM_LIMIT")
    gemini_tpm_limit: int = Field(default=1000000, alias="GEMINI_TPM_LIMIT")
    gemini_concurrency_limit: int = Field(default=10, alias="GEMINI_CONCURRENCY_LIMIT")
    gemini_timeout_seconds: float = Field(default=30.0, alias="GEMINI_TIMEOUT_SECONDS")
    gemini_max_retries: int = Field(default=3, alias="GEMINI_MAX_RETRIES")

    # Execution Budgets (Max LLM calls per query)
    budget_simple_qa: int = 2
    budget_normal_qa: int = 3
    budget_compare_risk: int = 5
    budget_max_retrieval_attempts: int = 2
    budget_max_regenerations: int = 1

    # Chunking Configurations (Tokens - Derived from retrieval_defaults)
    child_chunk_size: int = CHILD_CHUNK_SIZE
    child_chunk_overlap: int = CHILD_CHUNK_OVERLAP
    parent_chunk_size: int = PARENT_CHUNK_SIZE
    parent_chunk_overlap: int = PARENT_CHUNK_OVERLAP

    # Retrieval Confidence Weights
    confidence_weight_bm25_dense_rank: float = 0.25
    confidence_weight_rrf_top: float = 0.20
    confidence_weight_score_margin: float = 0.20
    confidence_weight_rerank_score: float = 0.20
    confidence_weight_metadata_match: float = 0.15

    # CORS Allowed Origins (Comma-separated)
    allowed_origins: str = Field(default="http://localhost:5173", alias="ALLOWED_ORIGINS")

    def get_allowed_origins(self) -> List[str]:
        """Parses comma-separated ALLOWED_ORIGINS string into a list of origins."""
        if not self.allowed_origins:
            return ["http://localhost:5173"]
        origins = [o.strip() for o in self.allowed_origins.split(",") if o.strip()]
        return origins if origins else ["http://localhost:5173"]

    # Optional Advanced Docling Parser Adapter
    use_docling_parser: bool = Field(default=False, alias="USE_DOCLING_PARSER")

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env", "../../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    def validate_security(self):
        """Validates critical security requirements at startup."""
        is_prod = self.environment.lower() in ["production", "prod"]
        if is_prod:
            dev_default_secret = "dev_insecure_jwt_secret_key_change_in_production_1234567890"
            if not self.jwt_secret_key or len(self.jwt_secret_key) < 32:
                raise ValueError("SECURITY FATAL: In production, JWT_SECRET_KEY must be >= 32 characters.")
            if self.jwt_secret_key == dev_default_secret:
                raise ValueError("SECURITY FATAL: In production, JWT_SECRET_KEY must not use the built-in development default secret.")
            if self.llm_provider.lower() == "gemini" and not self.gemini_api_key:
                raise ValueError("SECURITY FATAL: In production with LLM_PROVIDER=gemini, GEMINI_API_KEY must be set.")
            origins = self.get_allowed_origins()
            if "*" in origins:
                raise ValueError("SECURITY FATAL: In production, ALLOWED_ORIGINS must not contain wildcard '*' when credentials are enabled.")


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.validate_security()
    return _settings
