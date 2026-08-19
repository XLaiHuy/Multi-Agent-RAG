"""
BM25 Sparse Retriever.
Implements Okapi BM25 ranking algorithm over indexed child chunks with tenant and metadata filtering.
"""
import re
import threading
from typing import List, Tuple, Dict, Any, Optional
from rank_bm25 import BM25Okapi


def tokenize_for_bm25(text: str) -> List[str]:
    """Tokenize legal/technical text into normalized alphanumeric words."""
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = [w for w in cleaned.split() if len(w) > 1]
    return tokens


class BM25Retriever:
    """
    In-memory BM25 Index with thread-safe updates and metadata filtering.
    """

    def __init__(self):
        self.chunk_ids: List[str] = []
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Okapi] = None
        self._lock = threading.Lock()

    def build_index(
        self,
        chunk_ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ):
        """Build or replace BM25 Okapi index."""
        with self._lock:
            self.chunk_ids = list(chunk_ids)
            self.documents = list(documents)
            self.metadatas = list(metadatas) if metadatas else [{}] * len(chunk_ids)

            if not self.chunk_ids:
                self.bm25 = None
                return

            tokenized_corpus = [tokenize_for_bm25(doc) for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_corpus)

    def add_chunks(
        self,
        chunk_ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ):
        """Append new chunks and rebuild index."""
        with self._lock:
            self.chunk_ids.extend(chunk_ids)
            self.documents.extend(documents)
            if metadatas:
                self.metadatas.extend(metadatas)
            else:
                self.metadatas.extend([{}] * len(chunk_ids))

            tokenized_corpus = [tokenize_for_bm25(doc) for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_corpus)

    def search(
        self,
        query: str,
        top_k: int = 20,
        tenant_id: Optional[str] = None,
        allowed_doc_ids: Optional[List[str]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Executes BM25 search with metadata and ACL pre-filtering.
        Returns: List of (chunk_id, score, metadata) sorted by score descending.
        """
        with self._lock:
            if not self.bm25 or not query.strip():
                return []

            query_tokens = tokenize_for_bm25(query)
            if not query_tokens:
                return []

            scores = self.bm25.get_scores(query_tokens)

            results: List[Tuple[str, float, Dict[str, Any]]] = []
            allowed_set = set(allowed_doc_ids) if allowed_doc_ids is not None else None

            for i, (cid, raw_score) in enumerate(zip(self.chunk_ids, scores)):
                meta = self.metadatas[i]

                # Strict fail-closed Tenant filter
                if tenant_id is not None:
                    chunk_tenant = meta.get("tenant_id")
                    if not chunk_tenant or chunk_tenant != tenant_id:
                        continue

                # Strict fail-closed Doc ID filter
                if allowed_set is not None:
                    chunk_doc_id = meta.get("doc_id")
                    if not chunk_doc_id or chunk_doc_id not in allowed_set:
                        continue

                # Compute effective score: handle small-corpus BM25 IDF saturation
                tokenized_doc = tokenize_for_bm25(self.documents[i])
                overlap = len(set(query_tokens).intersection(set(tokenized_doc)))
                score = float(raw_score) if raw_score > 0.0 else (0.1 * overlap if overlap > 0 else 0.0)

                if score <= 0.0:
                    continue

                results.append((cid, float(score), meta))

            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]


_bm25_instance: Optional[BM25Retriever] = None


def get_bm25_retriever() -> BM25Retriever:
    global _bm25_instance
    if _bm25_instance is None:
        _bm25_instance = BM25Retriever()
    return _bm25_instance
