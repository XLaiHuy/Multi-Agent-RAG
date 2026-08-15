"""
Application Configuration and Settings Module.
Implements task-specific model routing, security validation, and environment defaults.
"""
import os
import sys
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Search for .env in current working directory and parent directories
load_dotenv()


class Settings(BaseSettings):
    # Environment
    environment: str = Field(default="development", alias="ENVIRONMENT")
    debug: bool = Field(default=False, alias="DEBUG")
    app_title: str = "Enterprise Contract Intelligence Platform"
    app_version: str = "2.0.0"

    # Security & JWT
    jwt_secret_key: str = Field(default="", alias="JWT_SECRET_KEY")
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = Field(default=60 * 24, alias="ACCESS_TOKEN_EXPIRE_MINUTES")

    # API Keys
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")

    # Model Routing - Decoupled per Task
    llm_provider: str = Field(default="gemini", alias="LLM_PROVIDER") # gemini | ollama | mock
    planner_model: str = Field(default="gemini-flash-latest", alias="PLANNER_MODEL")
    critic_model: str = Field(default="gemini-flash-latest", alias="CRITIC_MODEL")
    rewrite_model: str = Field(default="gemini-flash-latest", alias="REWRITE_MODEL")
    verifier_model: str = Field(default="gemini-flash-latest", alias="VERIFIER_MODEL")
    generation_model: str = Field(default="gemini-flash-latest", alias="GENERATION_MODEL")
    ocr_model: str = Field(default="gemini-flash-latest", alias="OCR_MODEL")

    # Local Ollama fallback (if configured)
    ollama_base_url: str = Field(default="http://localhost:11434/v1", alias="OLLAMA_BASE_URL")

    # Embedding Settings
    embedding_provider: str = Field(default="local", alias="EMBEDDING_PROVIDER") # local | gemini
    gemini_embedding_model: str = Field(default="text-embedding-004", alias="GEMINI_EMBEDDING_MODEL")
    local_embedding_model: str = Field(default="BAAI/bge-small-en-v1.5", alias="LOCAL_EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=384, alias="EMBEDDING_DIMENSION")

    # Reranker Settings
    enable_reranker: bool = Field(default=True, alias="ENABLE_RERANKER")
    reranker_model: str = Field(default="cross-encoder/ms-marco-TinyBERT-L-2-v2", alias="RERANKER_MODEL")

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

    # Chunking Configurations (Tokens)
    child_chunk_size: int = 250
    child_chunk_overlap: int = 50
    parent_chunk_size: int = 1200
    parent_chunk_overlap: int = 100

    # Retrieval Confidence Weights
    confidence_weight_bm25_dense_rank: float = 0.25
    confidence_weight_rrf_top: float = 0.20
    confidence_weight_score_margin: float = 0.20
    confidence_weight_rerank_score: float = 0.20
    confidence_weight_metadata_match: float = 0.15

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env", "../../.env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def validate_security(self):
        """Validates critical security requirements at startup."""
        is_prod = self.environment.lower() in ["production", "prod"]
        
        # Check JWT Secret in production
        if is_prod:
            if not self.jwt_secret_key or len(self.jwt_secret_key) < 32:
                raise RuntimeError(
                    "[FATAL SECURITY ERROR] In production environment, JWT_SECRET_KEY must be set "
                    "with a secure key of at least 32 characters. Application startup aborted."
                )
            if self.jwt_secret_key.startswith("rag_enterprise_dh_mo_2026") or self.jwt_secret_key == "replace_with_your_key":
                raise RuntimeError(
                    "[FATAL SECURITY ERROR] Default/insecure JWT_SECRET_KEY detected in production mode. "
                    "Application startup aborted."
                )
        else:
            # Development fallback
            if not self.jwt_secret_key:
                self.jwt_secret_key = "dev_secret_key_strictly_for_local_testing_only_32bytes_min"

        # Check Gemini API Key if using Gemini
        if self.llm_provider == "gemini" and not self.gemini_api_key:
            if is_prod:
                raise RuntimeError("[FATAL CONFIG ERROR] GEMINI_API_KEY must be set when LLM_PROVIDER is 'gemini'.")


_settings_instance: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
        _settings_instance.validate_security()
    return _settings_instance
