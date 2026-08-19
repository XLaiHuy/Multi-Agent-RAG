"""
Unit Tests for Retrieval Confidence Engine.
"""
import pytest
from backend.app.retrieval.confidence import RetrievalConfidenceEngine


def test_high_confidence_signals():
    engine = RetrievalConfidenceEngine()

    dense_ranked = ["chunk_1", "chunk_2", "chunk_3", "chunk_4", "chunk_5"]
    bm25_ranked = ["chunk_1", "chunk_2", "chunk_3", "chunk_5", "chunk_6"]
    fused_scores = [0.032, 0.020, 0.015, 0.010]
    rerank_scores = [0.92, 0.65, 0.40]

    top_meta = [
        {"section_path": ["Termination", "Notice Period"], "title": "MSA Agreement.pdf"},
        {"section_path": ["Termination Rights"], "title": "MSA Agreement.pdf"},
    ]

    signals = engine.compute_confidence(
        dense_ranked_ids=dense_ranked,
        bm25_ranked_ids=bm25_ranked,
        fused_scores=fused_scores,
        rerank_scores=rerank_scores,
        query="What is the termination notice period?",
        top_candidates_meta=top_meta,
    )

    assert signals.final_confidence >= 0.70
    assert signals.bm25_dense_rank_agreement >= 0.60
    assert signals.top_score_margin > 0.30
    assert signals.reranker_score >= 0.90


def test_low_confidence_signals():
    engine = RetrievalConfidenceEngine()

    # Disjoint rank lists
    dense_ranked = ["chunk_10", "chunk_11", "chunk_12"]
    bm25_ranked = ["chunk_20", "chunk_21", "chunk_22"]
    fused_scores = [0.016, 0.016, 0.015] # Flat scores, negligible margin
    rerank_scores = [0.15, 0.12] # Low CrossEncoder score

    signals = engine.compute_confidence(
        dense_ranked_ids=dense_ranked,
        bm25_ranked_ids=bm25_ranked,
        fused_scores=fused_scores,
        rerank_scores=rerank_scores,
        query="Random complex unrelated query",
        top_candidates_meta=[],
    )

    assert signals.final_confidence < 0.45
    assert signals.bm25_dense_rank_agreement == 0.0
    assert signals.top_score_margin <= 0.10
