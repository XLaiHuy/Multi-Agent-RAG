import os
import time

# pyrefly: ignore [missing-import]
from google import genai
# pyrefly: ignore [missing-import]
from google.genai import types
# pyrefly: ignore [missing-import]
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import Settings, get_settings


class GeminiEmbeddingProvider:

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.client = genai.Client(api_key=self.settings.gemini_api_key)
        self.model = self.settings.embedding_model
        self.dimension = self.settings.embedding_dimension
        self.delay = self.settings.request_delay_seconds

    def _embed_with_retry(self, formatted_text: str) -> list[float]:
        @retry(
            reraise=True,
            stop=stop_after_attempt(10),
            wait=wait_exponential(multiplier=2, min=2, max=65),
            retry=retry_if_exception_type(Exception),
        )
        def _call_api():
            response = self.client.models.embed_content(
                model=self.model,
                contents=formatted_text,
                config=types.EmbedContentConfig(
                    output_dimensionality=self.dimension,
                ),
            )
            if not response.embeddings or not response.embeddings[0].values:
                raise ValueError("Received empty embedding vector from Gemini API.")
            values = list(response.embeddings[0].values)
            if len(values) != self.dimension:
                raise ValueError(
                    f"Expected embedding dimension {self.dimension}, got {len(values)}"
                )
            return values

        if self.delay > 0:
            time.sleep(self.delay)

        return _call_api()

    def embed_document(self, text: str, title: str | None = None) -> list[float]:
        if title:
            formatted_text = f"title: {title} | text: {text}"
        else:
            formatted_text = f"text: {text}"

        return self._embed_with_retry(formatted_text)

    def embed_documents_batch(
        self, documents: list[tuple[str, str | None]], batch_size: int = 100
    ) -> list[list[float]]:
        all_vectors: list[list[float]] = []

        total_batches = (len(documents) + batch_size - 1) // batch_size
        for batch_idx, i in enumerate(range(0, len(documents), batch_size), start=1):
            batch = documents[i : i + batch_size]
            formatted_batch = [
                f"title: {title} | text: {text}" if title else f"text: {text}"
                for text, title in batch
            ]

            @retry(
                reraise=True,
                stop=stop_after_attempt(10),
                wait=wait_exponential(multiplier=2, min=2, max=65),
                retry=retry_if_exception_type(Exception),
            )
            def _call_batch():
                response = self.client.models.embed_content(
                    model=self.model,
                    contents=formatted_batch,
                    config=types.EmbedContentConfig(
                        output_dimensionality=self.dimension,
                    ),
                )
                if not response.embeddings:
                    raise ValueError("Received empty response from Gemini API batch embedding.")
                return [list(e.values) for e in response.embeddings]

            print(f"  [Gemini API] Processing Batch {batch_idx}/{total_batches} ({len(batch)} chunks)...", flush=True)
            batch_vectors = _call_batch()
            all_vectors.extend(batch_vectors)
            time.sleep(max(1.0, self.delay))

        return all_vectors

    def embed_query(self, query: str) -> list[float]:
        formatted_text = f"task: question answering | query: {query}"
        return self._embed_with_retry(formatted_text)


class LocalEmbeddingProvider:

    def __init__(self, model_name: str | None = None):
        # pyrefly: ignore [missing-import]
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name or os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-m3").strip()
        print(f"[Local Embedding] Initializing SentenceTransformer model '{self.model_name}'...")
        self.model = SentenceTransformer(self.model_name, trust_remote_code=True)

    def embed_query(self, query: str) -> list[float]:
        vector = self.model.encode(query, show_progress_bar=False)
        return vector.tolist()

    def embed_documents_batch(
        self, documents: list[tuple[str, str | None]], batch_size: int = 64
    ) -> list[list[float]]:
        texts = [text for text, _ in documents]
        print(f"  [Local Embedding] Computing embeddings for {len(texts)} chunks using {self.model_name}...", flush=True)
        vectors = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        return vectors.tolist()


def get_embedding_provider(settings: Settings | None = None):
    settings = settings or get_settings()
    if settings.embedding_provider == "local":
        return LocalEmbeddingProvider()
    else:
        return GeminiEmbeddingProvider(settings)
