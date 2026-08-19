"""
Unit Tests for Fail-Closed Multi-Tenant and Document ACL Filtering in Retrievers.
"""
import pytest
from backend.app.retrieval.bm25 import BM25Retriever


def test_bm25_fail_closed_tenant_isolation():
    retriever = BM25Retriever()
    chunk_ids = ["c1", "c2", "c3", "c4"]
    docs = [
        "Payment terms are net 30 days.",
        "Payment terms are net 45 days.",
        "Payment terms are net 60 days.",
        "Payment terms are net 15 days.",
    ]
    metadatas = [
        {"tenant_id": "tenant_alpha", "doc_id": "doc_1"},
        {"tenant_id": "tenant_beta", "doc_id": "doc_2"},
        {"doc_id": "doc_3"}, # Missing tenant_id -> MUST be rejected when tenant_id filter is provided
        {"tenant_id": "tenant_alpha", "doc_id": "doc_4"},
    ]
    retriever.build_index(chunk_ids, docs, metadatas)

    # 1. Search scoped to tenant_alpha
    results_alpha = retriever.search(query="payment terms", tenant_id="tenant_alpha")
    returned_cids = [cid for cid, _, _ in results_alpha]
    assert "c1" in returned_cids
    assert "c4" in returned_cids
    assert "c2" not in returned_cids # tenant_beta
    assert "c3" not in returned_cids # missing tenant_id must fail closed!


def test_bm25_fail_closed_doc_id_isolation():
    retriever = BM25Retriever()
    chunk_ids = ["c1", "c2", "c3"]
    docs = [
        "Liability is limited to fees paid.",
        "Liability is limited to 100000 USD.",
        "Liability is uncapped.",
    ]
    metadatas = [
        {"tenant_id": "t1", "doc_id": "doc_allowed"},
        {"tenant_id": "t1", "doc_id": "doc_forbidden"},
        {"tenant_id": "t1"}, # Missing doc_id -> MUST be rejected when allowed_doc_ids filter is provided
    ]
    retriever.build_index(chunk_ids, docs, metadatas)

    # Search with allowed_doc_ids=["doc_allowed"]
    results = retriever.search(
        query="liability limited",
        tenant_id="t1",
        allowed_doc_ids=["doc_allowed"]
    )
    returned_cids = [cid for cid, _, _ in results]
    assert returned_cids == ["c1"]
    assert "c2" not in returned_cids
    assert "c3" not in returned_cids


def test_bm25_empty_allowed_docs_returns_empty():
    retriever = BM25Retriever()
    retriever.build_index(["c1"], ["Arbitration clause."], [{"tenant_id": "t1", "doc_id": "d1"}])
    results = retriever.search(query="arbitration", tenant_id="t1", allowed_doc_ids=[])
    assert len(results) == 0
