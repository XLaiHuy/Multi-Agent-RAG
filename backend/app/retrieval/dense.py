"""
Dense Semantic Vector Retriever.
Integrates with ChromaDB and EmbeddingProvider with tenant isolation and ACL pre-filtering.
"""
import os
import logging
import chromadb
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from backend.app.core.config import get_settings
from backend.app.providers.embeddings import get_embedding_provider, EmbeddingProvider

logger = logging.getLogger("dense_retriever")



@dataclass
class DenseSearchResult:
    chunk_id: str
    text: str
    similarity: float
    distance: float
    metadata: Dict[str, Any]


class DenseRetriever:
    """
    Dense Vector Retriever utilizing ChromaDB persistent collection.
    """

    def __init__(self, embedding_provider: Optional[EmbeddingProvider] = None):
        self.settings = get_settings()
        self.embedder = embedding_provider or get_embedding_provider()
        
        os.makedirs(self.settings.chroma_path, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=self.settings.chroma_path)
        
        try:
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.settings.chroma_collection,
                metadata={
                    "hnsw:space": "cosine",
                    "embedding_dimension": self.embedder.dimension,
                },
            )
        except Exception:
            # Recreate collection if incompatible dimension
            try:
                self.chroma_client.delete_collection(self.settings.chroma_collection)
            except Exception:
                pass
            self.collection = self.chroma_client.create_collection(
                name=self.settings.chroma_collection,
                metadata={
                    "hnsw:space": "cosine",
                    "embedding_dimension": self.embedder.dimension,
                },
            )

    def upsert_chunks(
        self,
        chunk_ids: List[str],
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ):
        """Batch upsert child chunks into ChromaDB."""
        if not chunk_ids:
            return
        self.collection.upsert(
            ids=chunk_ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(
        self,
        query: str,
        top_k: int = 20,
        tenant_id: Optional[str] = None,
        allowed_doc_ids: Optional[List[str]] = None,
    ) -> List[DenseSearchResult]:
        """
        Executes semantic vector search with tenant and document ACL filtering.
        """
        if not query.strip():
            return []

        total_count = 0
        try:
            total_count = self.collection.count()
            if total_count <= 0:
                return []
        except Exception:
            return []

        query_vector = self.embedder.embed_query(query)

        # Build Chroma where filter
        where_conditions = []
        if tenant_id:
            where_conditions.append({"tenant_id": tenant_id})
        if allowed_doc_ids:
            if len(allowed_doc_ids) == 1:
                where_conditions.append({"doc_id": allowed_doc_ids[0]})
            elif len(allowed_doc_ids) > 1:
                where_conditions.append({"doc_id": {"$in": allowed_doc_ids}})

        where_filter = None
        if len(where_conditions) == 1:
            where_filter = where_conditions[0]
        elif len(where_conditions) > 1:
            where_filter = {"$and": where_conditions}

        query_k = max(1, min(top_k, total_count))

        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=query_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.warning(f"[DenseRetriever] Chroma query skipped ({e}), falling back to lexical search.")
            return []


        hits: List[DenseSearchResult] = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for cid, doc_text, meta, dist in zip(ids, docs, metas, dists):
            # For cosine distance in Chroma: similarity = 1 - distance
            similarity = max(0.0, min(1.0, 1.0 - float(dist)))
            hits.append(
                DenseSearchResult(
                    chunk_id=cid,
                    text=doc_text,
                    similarity=similarity,
                    distance=float(dist),
                    metadata=meta or {},
                )
            )

        return hits


_dense_retriever_instance: Optional[DenseRetriever] = None


def get_dense_retriever() -> DenseRetriever:
    global _dense_retriever_instance
    if _dense_retriever_instance is None:
        _dense_retriever_instance = DenseRetriever()
    return _dense_retriever_instance
