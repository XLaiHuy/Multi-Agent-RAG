#!/usr/bin/env python3
import pytest
import numpy as np
from typing import List, Dict, Any

from backend.app.retrieval.bm25 import BM25Retriever, tokenize_for_bm25
from backend.app.retrieval.fusion import reciprocal_rank_fusion
from rank_bm25 import BM25Okapi


def perform_true_document_scoped_retrieval(
    query: str,
    selected_doc_id: str,
    chunks: List[Dict[str, Any]],
    chunk_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    rrf_k: int = 60,
    top_candidates: int = 50,
) -> List[str]:
    """
    True document-scoped first-stage retrieval:
    Prefilters document chunks before Dense and BM25 ranking and fuses within the document scope.
    """
    # 1. Prefilter chunks by selected_doc_id
    scoped_indices = [i for i, c in enumerate(chunks) if c["doc_id"] == selected_doc_id]
    if not scoped_indices:
        return []

    scoped_chunks = [chunks[i] for i in scoped_indices]
    scoped_chunk_ids = [c["chunk_id"] for c in scoped_chunks]

    # 2. Scoped Dense ranking
    scoped_emb = chunk_embeddings[scoped_indices]  # (N_scoped, dim)
    dense_scores = np.dot(scoped_emb, query_embedding)
    dense_order = np.argsort(-dense_scores)
    dense_ranked_ids = [scoped_chunk_ids[i] for i in dense_order[:top_candidates]]

    # 3. Scoped BM25 ranking (index built exclusively over scoped doc chunks)
    tokenized_corpus = [tokenize_for_bm25(c.get("enriched_text", c["text"])) for c in scoped_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    q_tokens = tokenize_for_bm25(query)
    bm25_scores = bm25.get_scores(q_tokens) if q_tokens else np.zeros(len(scoped_chunks))
    bm25_order = np.argsort(-bm25_scores)
    bm25_ranked_ids = [scoped_chunk_ids[i] for i in bm25_order[:top_candidates]]

    # 4. Scoped RRF
    fused = reciprocal_rank_fusion([dense_ranked_ids, bm25_ranked_ids], k=rrf_k)
    return [cid for cid, _ in fused]


def test_document_scoped_no_cross_document_leakage():
    # Setup mock multi-contract corpus
    mock_chunks = [
        {"chunk_id": "doc1_c1", "doc_id": "doc_1", "text": "This Agreement is governed by Delaware law."},
        {"chunk_id": "doc1_c2", "doc_id": "doc_1", "text": "The term is five years from Effective Date."},
        {"chunk_id": "doc2_c1", "doc_id": "doc_2", "text": "This Agreement is governed by New York law."},
        {"chunk_id": "doc2_c2", "doc_id": "doc_2", "text": "Governing Law shall be the State of California."},
    ]
    # Synthetic orthogonal embeddings
    mock_emb = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)

    # Query matching doc2_c1 strongly
    query_emb = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    query = "governed by New York law"

    # Query scoped strictly to doc_1
    result_ids = perform_true_document_scoped_retrieval(
        query=query,
        selected_doc_id="doc_1",
        chunks=mock_chunks,
        chunk_embeddings=mock_emb,
        query_embedding=query_emb,
    )

    # Verify all returned chunks belong exclusively to doc_1
    assert len(result_ids) == 2
    for cid in result_ids:
        assert cid.startswith("doc1_")
    assert "doc2_c1" not in result_ids
    assert "doc2_c2" not in result_ids
