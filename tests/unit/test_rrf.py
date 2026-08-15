"""
Unit Tests for Reciprocal Rank Fusion (RRF).
"""
import pytest
from backend.app.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_deterministic_ordering():
    list1 = ["doc_a", "doc_b", "doc_c"]
    list2 = ["doc_b", "doc_a", "doc_d"]

    # RRF with k=60
    # doc_a: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.016393 + 0.016129 = 0.032522
    # doc_b: 1/(60+2) + 1/(60+1) = 0.032522
    # doc_c: 1/(60+3) = 0.015873
    # doc_d: 1/(60+3) = 0.015873
    fused = reciprocal_rank_fusion([list1, list2], k=60)

    assert len(fused) == 4
    top_docs = [item[0] for item in fused[:2]]
    assert "doc_a" in top_docs
    assert "doc_b" in top_docs
    assert fused[0][1] > fused[2][1]


def test_rrf_empty_lists():
    fused = reciprocal_rank_fusion([], k=60)
    assert fused == []

    fused2 = reciprocal_rank_fusion([[], []], k=60)
    assert fused2 == []


def test_rrf_single_list():
    list1 = ["doc_1", "doc_2", "doc_3"]
    fused = reciprocal_rank_fusion([list1], k=60)
    assert len(fused) == 3
    assert fused[0][0] == "doc_1"
    assert fused[1][0] == "doc_2"
    assert fused[2][0] == "doc_3"
    assert fused[0][1] == pytest.approx(1.0 / 61.0)
