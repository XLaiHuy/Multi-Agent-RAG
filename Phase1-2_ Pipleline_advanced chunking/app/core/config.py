import os
from dataclasses import dataclass
from pathlib import Path
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class Settings:
    gemini_api_key: str
    embedding_provider: str = "gemini"
    embedding_model: str = "gemini-embedding-001"
    embedding_dimension: int = 768
    chroma_path: str = "./data/chroma"
    collection_name: str = "rag_gemini_embedding_001_768"
    max_retries: int = 5
    request_delay_seconds: float = 0.2
    
    llm_provider: str = "gemini"
    llm_model: str = "gemini-3.5-flash"
    ollama_base_url: str = "http://localhost:11434/v1"
    enable_reranker: bool = False

    @classmethod
    def load(cls) -> "Settings":
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        provider = os.getenv("EMBEDDING_PROVIDER", "gemini").strip()
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-004").strip()
        dimension_str = os.getenv("EMBEDDING_DIMENSION", "768").strip()
        chroma_path = os.getenv("CHROMA_PATH", "./data/chroma").strip()
        collection_name = os.getenv("CHROMA_COLLECTION", "rag_gemini_embedding_2_768").strip()
        max_retries_str = os.getenv("GEMINI_MAX_RETRIES", "5").strip()
        delay_str = os.getenv("GEMINI_REQUEST_DELAY_SECONDS", "0.2").strip()

        if provider not in ["gemini", "local"]:
            raise ValueError(f"Invalid EMBEDDING_PROVIDER '{provider}'. Must be 'gemini' or 'local'.")

        if provider == "gemini" and (not api_key or api_key == "your_real_api_key" or api_key == "replace_with_your_key"):
            raise ValueError(
                "GEMINI_API_KEY is not set or invalid in .env! "
                "Please configure a valid Gemini API Key in .env before running with provider='gemini'."
            )

        try:
            dimension = int(dimension_str)
            if dimension <= 0:
                raise ValueError
        except ValueError:
            raise ValueError(f"EMBEDDING_DIMENSION must be a positive integer, got '{dimension_str}'")

        try:
            max_retries = int(max_retries_str)
        except ValueError:
            max_retries = 5

        try:
            request_delay_seconds = float(delay_str)
        except ValueError:
            request_delay_seconds = 0.2

        llm_provider = os.getenv("LLM_PROVIDER", "gemini").strip()
        llm_model = os.getenv("LLM_MODEL", "gemini-3.5-flash").strip()
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1").strip()
        enable_reranker_str = os.getenv("ENABLE_RERANKER", "false").strip().lower()
        enable_reranker = enable_reranker_str in ["true", "1", "yes"]

        return cls(
            gemini_api_key=api_key,
            embedding_provider=provider,
            embedding_model=model,
            embedding_dimension=dimension,
            chroma_path=chroma_path,
            collection_name=collection_name,
            max_retries=max_retries,
            request_delay_seconds=request_delay_seconds,
            llm_provider=llm_provider,
            llm_model=llm_model,
            ollama_base_url=ollama_base_url,
            enable_reranker=enable_reranker,
        )


def get_settings() -> Settings:
    return Settings.load()
