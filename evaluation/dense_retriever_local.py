"""
In-Memory Dense Retriever for evaluation use.
Uses LocalEmbeddingProvider to embed documents and compute cosine similarity
without requiring ChromaDB. Used in ablation benchmarks.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from backend.app.providers.embeddings import LocalEmbeddingProvider


class InMemoryDenseRetriever:
    """
    Pure in-memory dense retriever using numpy cosine similarity.
    Independent of ChromaDB — for benchmark use only.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.embedder = LocalEmbeddingProvider(model_name=model_name)
        self.chunk_ids: List[str] = []
        self.chunk_texts: List[str] = []
        self.embeddings: Optional[np.ndarray] = None

    def build_index(
        self,
        chunk_ids: List[str],
        texts: List[str],
        batch_size: int = 64,
    ):
        """Embed all documents and build in-memory L2-normalized matrix."""
        print(f"  [DenseRetriever] Embedding {len(texts)} chunks (batch_size={batch_size})...")
        self.chunk_ids = list(chunk_ids)
        self.chunk_texts = list(texts)
        vecs = self.embedder.embed_documents_batch(texts, batch_size=batch_size)
        arr = np.array(vecs, dtype=np.float32)
        # L2 normalize (embedder already normalizes, but enforce)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self.embeddings = arr / norms
        print(f"  [DenseRetriever] Dense index built: {self.embeddings.shape}")

    def search(
        self,
        query: str,
        top_k: int = 20,
    ) -> List[Tuple[str, float]]:
        """
        Returns List[(chunk_id, cosine_similarity)] sorted descending.
        """
        if self.embeddings is None or len(self.chunk_ids) == 0:
            return []

        q_vec = np.array(self.embedder.embed_query(query), dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        # Cosine similarity via dot product (both vectors are L2-normalized)
        sims = self.embeddings @ q_vec
        top_idxs = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in top_idxs:
            results.append((self.chunk_ids[idx], float(sims[idx])))
        return results
