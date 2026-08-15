"""
Provider Interfaces & Abstract Base Classes.
Ensures domain and application business logic are decoupled from specific SDKs / third-party vendors.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, AsyncIterator, Iterator
from pathlib import Path


class LLMProvider(ABC):
    """Abstract interface for LLM text generation and structured outputs."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        model_type: str = "generation",
        temperature: float = 0.2,
        max_output_tokens: Optional[int] = None,
    ) -> str:
        """Generate complete text response."""
        pass

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        model_type: str = "generation",
        temperature: float = 0.2,
    ) -> Iterator[str]:
        """Stream generated text chunks."""
        pass

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        schema: Any,
        system_instruction: Optional[str] = None,
        model_type: str = "planner",
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Generate structured JSON / Pydantic compliant dictionary."""
        pass


class EmbeddingProvider(ABC):
    """Abstract interface for dense embedding generation."""

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding vector for a single query."""
        pass

    @abstractmethod
    def embed_documents_batch(
        self, texts: List[str], batch_size: int = 64
    ) -> List[List[float]]:
        """Generate embedding vectors for a batch of document texts."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding vector dimension."""
        pass


class RerankerProvider(ABC):
    """Abstract interface for CrossEncoder / reranking models."""

    @abstractmethod
    def rerank(
        self, query: str, candidate_texts: List[str], top_n: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Reranks candidates against the query.
        Returns a list of (original_candidate_index, score) sorted by score descending.
        """
        pass


class OCRProvider(ABC):
    """Abstract interface for OCR extraction."""

    @abstractmethod
    def extract_from_image_bytes(
        self, image_bytes: bytes, mime_type: str = "image/png"
    ) -> str:
        """Extract markdown text and tables from image bytes."""
        pass

    @abstractmethod
    def extract_from_page(self, page_image_path: Path) -> str:
        """Extract text from a page image file."""
        pass


class ObjectStore(ABC):
    """Abstract interface for binary document and artifact storage."""

    @abstractmethod
    def save_file(self, file_path_key: str, content: bytes) -> str:
        """Save file content and return storage URI or relative path."""
        pass

    @abstractmethod
    def read_file(self, file_path_key: str) -> bytes:
        """Read file content bytes."""
        pass

    @abstractmethod
    def delete_file(self, file_path_key: str) -> bool:
        """Delete stored file."""
        pass

    @abstractmethod
    def file_exists(self, file_path_key: str) -> bool:
        """Check if file exists."""
        pass


class CacheStore(ABC):
    """Abstract interface for Exact and Semantic Caching."""

    @abstractmethod
    def get_exact(self, namespace: str, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached value by exact key in namespace."""
        pass

    @abstractmethod
    def set_exact(
        self, namespace: str, key: str, value: Dict[str, Any], ttl_seconds: int = 3600
    ) -> None:
        """Store exact key-value with TTL."""
        pass

    @abstractmethod
    def get_semantic(
        self,
        namespace: str,
        query_vector: List[float],
        similarity_threshold: float = 0.96,
    ) -> Optional[Dict[str, Any]]:
        """Lookup semantic cache by cosine similarity within tenant/ACL namespace."""
        pass

    @abstractmethod
    def set_semantic(
        self,
        namespace: str,
        query: str,
        query_vector: List[float],
        result: Dict[str, Any],
        ttl_seconds: int = 86400,
    ) -> None:
        """Store semantic query vector and result within namespace."""
        pass

    @abstractmethod
    def invalidate_namespace(self, namespace: str) -> None:
        """Invalidate all cached items for a specific tenant or corpus version."""
        pass
