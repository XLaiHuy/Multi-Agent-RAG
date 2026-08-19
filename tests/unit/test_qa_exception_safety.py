"""
Unit Tests for QA Candidate Deduplication and Exception Fallback Safety.
"""
import pytest
from backend.app.application.contract_qa import ContractQAService
from backend.app.retrieval.fusion import RetrievedCandidate


def test_qa_deduplicate_candidates_preserves_best_score_and_budget():
    c1 = RetrievedCandidate(
        chunk_id="c_1", doc_id="d1", doc_version=1, text="Text 1",
        is_parent_expanded=False, parent_id=None, page_number=1,
        section_path=["Sec 1"], block_id="b1", bbox=None,
        dense_score=0.7, bm25_score=5.0, rrf_score=0.015, rerank_score=0.80
    )
    c2 = RetrievedCandidate(
        chunk_id="c_2", doc_id="d1", doc_version=1, text="Text 2",
        is_parent_expanded=False, parent_id=None, page_number=2,
        section_path=["Sec 2"], block_id="b2", bbox=None,
        dense_score=0.5, bm25_score=3.0, rrf_score=0.010, rerank_score=0.60
    )
    # Duplicate of c1 with higher score
    c1_improved = RetrievedCandidate(
        chunk_id="c_1", doc_id="d1", doc_version=1, text="Text 1 Improved",
        is_parent_expanded=False, parent_id=None, page_number=1,
        section_path=["Sec 1"], block_id="b1", bbox=None,
        dense_score=0.9, bm25_score=8.0, rrf_score=0.025, rerank_score=0.95
    )
    c3 = RetrievedCandidate(
        chunk_id="c_3", doc_id="d1", doc_version=1, text="Text 3",
        is_parent_expanded=False, parent_id=None, page_number=3,
        section_path=["Sec 3"], block_id="b3", bbox=None,
        dense_score=0.4, bm25_score=2.0, rrf_score=0.008, rerank_score=0.50
    )

    merged = ContractQAService._deduplicate_candidates(
        existing=[c1, c2],
        new_cands=[c1_improved, c3],
        max_k=2,
    )

    assert len(merged) == 2
    assert merged[0].chunk_id == "c_1"
    assert merged[0].rerank_score == 0.95
    assert merged[1].chunk_id == "c_2"
