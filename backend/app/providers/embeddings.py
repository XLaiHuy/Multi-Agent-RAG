"""
Embedding Providers (Local SentenceTransformers & Gemini API).
Decoupled implementation conforming to EmbeddingProvider interface.
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import time
import logging
from typing import List, Optional
from google import genai
from google.genai import types

from backend.app.core.config import Settings, get_settings
from backend.app.providers.interfaces import EmbeddingProvider

logger = logging.getLogger("embeddings")


class LocalEmbeddingProvider(EmbeddingProvider):
    """Local SentenceTransformer embedding provider (e.g. BAAI/bge-m3)."""

    def __init__(self, model_name: Optional[str] = None):
        settings = get_settings()
        self.model_name = model_name or settings.local_embedding_model
        self._dim = settings.embedding_dimension
        self._model = None

    def _get_model(self):
        if self._model is None:
            logger.info(f"[LocalEmbedding] Loading SentenceTransformer '{self.model_name}'...")
            from sentence_transformers import SentenceTransformer
            try:
                self._model = SentenceTransformer(self.model_name, local_files_only=True)
            except Exception:
                self._model = SentenceTransformer(self.model_name)
            self._dim = self._model.get_embedding_dimension() if hasattr(self._model, "get_embedding_dimension") else self._model.get_sentence_embedding_dimension()
        return self._model

    def embed_query(self, query: str) -> List[float]:
        model = self._get_model()
        vec = model.encode(query, show_progress_bar=False, normalize_embeddings=True)
        return vec.tolist()

    def embed_queries_batch(
        self, queries: List[str], batch_size: int = 32
    ) -> List[List[float]]:
        """Batch encodes queries using the identical normalization and inference protocol as embed_query."""
        if not queries:
            return []
        model = self._get_model()
        vectors = model.encode(
            queries, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True
        )
        return vectors.tolist()

    def embed_documents_batch(
        self, texts: List[str], batch_size: int = 32
    ) -> List[List[float]]:
        if not texts:
            return []
        model = self._get_model()
        vectors = model.encode(
            texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True
        )
        return vectors.tolist()

    @property
    def dimension(self) -> int:
        self._get_model()
        return self._dim


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Google Gemini API embedding provider."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.client = genai.Client(api_key=self.settings.gemini_api_key) if self.settings.gemini_api_key else None
        self.model = self.settings.gemini_embedding_model
        self._dim = self.settings.embedding_dimension

    def embed_query(self, query: str) -> List[float]:
        if not self.client:
            raise RuntimeError("Gemini API key is not configured.")
        res = self.client.models.embed_content(
            model=self.model,
            contents=f"task: contract retrieval | query: {query}",
            config=types.EmbedContentConfig(output_dimensionality=self._dim),
        )
        if not res.embeddings or not res.embeddings[0].values:
            raise ValueError("Empty embedding vector received from Gemini API.")
        return list(res.embeddings[0].values)

    def embed_queries_batch(
        self, queries: List[str], batch_size: int = 32
    ) -> List[List[float]]:
        """Batch encodes queries using the Gemini EmbedContent API."""
        if not self.client:
            raise RuntimeError("Gemini API key is not configured.")
        if not queries:
            return []

        all_vectors = []
        for i in range(0, len(queries), batch_size):
            batch = queries[i : i + batch_size]
            formatted_batch = [f"task: contract retrieval | query: {q}" for q in batch]
            res = self.client.models.embed_content(
                model=self.model,
                contents=formatted_batch,
                config=types.EmbedContentConfig(output_dimensionality=self._dim),
            )
            if not res.embeddings:
                raise ValueError("Empty batch embedding received from Gemini API.")
            all_vectors.extend([list(e.values) for e in res.embeddings])
            if i + batch_size < len(queries):
                time.sleep(0.05)

        return all_vectors

    def embed_documents_batch(
        self, texts: List[str], batch_size: int = 50
    ) -> List[List[float]]:
        if not self.client:
            raise RuntimeError("Gemini API key is not configured.")
        if not texts:
            return []

        all_vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            formatted_batch = [f"task: contract document | text: {t}" for t in batch]
            res = self.client.models.embed_content(
                model=self.model,
                contents=formatted_batch,
                config=types.EmbedContentConfig(output_dimensionality=self._dim),
            )
            if not res.embeddings:
                raise ValueError("Empty batch embedding received from Gemini API.")
            all_vectors.extend([list(e.values) for e in res.embeddings])
            time.sleep(0.1)

        return all_vectors

    @property
    def dimension(self) -> int:
        return self._dim


_embedding_provider_instance: Optional[EmbeddingProvider] = None


def get_embedding_provider() -> EmbeddingProvider:
    global _embedding_provider_instance
    if _embedding_provider_instance is None:
        settings = get_settings()
        if settings.embedding_provider.lower() == "gemini":
            _embedding_provider_instance = GeminiEmbeddingProvider(settings)
        else:
            _embedding_provider_instance = LocalEmbeddingProvider()
    return _embedding_provider_instance
