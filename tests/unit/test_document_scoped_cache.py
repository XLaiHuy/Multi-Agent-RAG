"""
Unit Tests for Document-Scoped Cache Identity and Isolation.
Proves that different document scopes never collide on cache, canonical ordering is preserved,
and tenant / role boundaries are strictly enforced.
"""
import pytest
from backend.app.persistence.cache import (
    BoundedSemanticCache,
    compute_acl_scope_hash,
    build_query_cache_identity,
)


def test_document_scope_order_is_canonical():
    """Property B: Ordering of document_ids must be canonical so [A, B] and [B, A] share cache identity."""
    id1 = build_query_cache_identity("tenant_1", "admin", document_ids=["doc_A", "doc_B"])
    id2 = build_query_cache_identity("tenant_1", "admin", document_ids=["doc_B", "doc_A"])
    id_dup = build_query_cache_identity("tenant_1", "admin", document_ids=["doc_A", "doc_B", "doc_A"])
    assert id1 == id2
    assert id1 == id_dup


def test_same_query_different_document_scope_no_cache_collision():
    """Property A: Same query on Contract A and Contract B must have different cache identities and NOT collide."""
    id_docA = build_query_cache_identity("tenant_1", "admin", document_ids=["doc_A"])
    id_docB = build_query_cache_identity("tenant_1", "admin", document_ids=["doc_B"])
    assert id_docA != id_docB

    cache = BoundedSemanticCache()
    cache.set_exact(id_docA, "What is the liability cap?", {"answer": "Liability cap is 1M", "citations": []})

    # Looking up under doc_B namespace must return None
    hit_b = cache.get_exact(id_docB, "What is the liability cap?")
    assert hit_b is None

    # Looking up under doc_A namespace must return doc_A answer
    hit_a = cache.get_exact(id_docA, "What is the liability cap?")
    assert hit_a is not None
    assert hit_a["answer"] == "Liability cap is 1M"


def test_tenant_cache_isolation():
    """Property C: Different tenants must never share cache identity."""
    id_t1 = build_query_cache_identity("tenant_1", "admin", document_ids=["doc_A"])
    id_t2 = build_query_cache_identity("tenant_2", "admin", document_ids=["doc_A"])
    assert id_t1 != id_t2


def test_role_cache_isolation():
    """Property D: Different roles must never share cache identity."""
    id_admin = build_query_cache_identity("tenant_1", "admin", document_ids=["doc_A"])
    id_legal = build_query_cache_identity("tenant_1", "legal", document_ids=["doc_A"])
    assert id_admin != id_legal


def test_unscoped_and_scoped_query_do_not_collide():
    """Property E: Unscoped (all docs) query must not collide with specific document scoped query."""
    id_unscoped_none = build_query_cache_identity("tenant_1", "admin", document_ids=None)
    id_unscoped_empty = build_query_cache_identity("tenant_1", "admin", document_ids=[])
    id_scoped = build_query_cache_identity("tenant_1", "admin", document_ids=["doc_A"])

    assert id_unscoped_none == id_unscoped_empty
    assert id_unscoped_none != id_scoped


def test_cache_identity_changes_with_effective_embedding_model():
    """Cache identity MUST change when the effective embedding model/dimension changes."""
    id_prod = build_query_cache_identity("t1", "admin", ["doc_A"], embedding_identity="local::BAAI/bge-small-en-v1.5::384")
    id_eval = build_query_cache_identity("t1", "admin", ["doc_A"], embedding_identity="local::BAAI/bge-m3::1024")
    id_gemini = build_query_cache_identity("t1", "admin", ["doc_A"], embedding_identity="gemini::text-embedding-004::768")

    assert id_prod != id_eval
    assert id_prod != id_gemini
    assert id_eval != id_gemini


def test_cache_identity_uses_gemini_model_when_provider_is_gemini():
    """When embedding_provider is 'gemini', effective identity must use gemini_embedding_model."""
    from backend.app.persistence.cache import get_effective_embedding_identity
    from backend.app.core.config import Settings

    gemini_settings = Settings(
        EMBEDDING_PROVIDER="gemini",
        GEMINI_EMBEDDING_MODEL="text-embedding-004",
        EMBEDDING_DIMENSION=768,
    )
    eff = get_effective_embedding_identity(gemini_settings)
    assert eff == "gemini::text-embedding-004::768"

    cache_id = build_query_cache_identity("t1", "admin", ["doc_1"], embedding_identity=eff)
    assert "gemini" in eff
    assert len(cache_id) == 24


def test_cache_identity_uses_local_model_when_provider_is_local():
    """When embedding_provider is 'local', effective identity must use local_embedding_model."""
    from backend.app.persistence.cache import get_effective_embedding_identity
    from backend.app.core.config import Settings

    local_settings = Settings(
        EMBEDDING_PROVIDER="local",
        LOCAL_EMBEDDING_MODEL="BAAI/bge-small-en-v1.5",
        EMBEDDING_DIMENSION=384,
    )
    eff = get_effective_embedding_identity(local_settings)
    assert eff == "local::BAAI/bge-small-en-v1.5::384"

    cache_id = build_query_cache_identity("t1", "admin", ["doc_1"], embedding_identity=eff)
    assert "local" in eff
    assert len(cache_id) == 24


def test_unscoped_query_bypasses_exact_cache():
    """Unscoped / ALL-document queries (document_ids=None or []) MUST bypass exact cache lookup and write."""
    from backend.app.application.contract_qa import ContractQAService
    from backend.app.domain.schemas import StructuredAnswer, ExecutionStats
    from unittest.mock import MagicMock, patch

    service = ContractQAService()
    mock_gw = MagicMock()
    mock_gw.generate.return_value = "Unscoped fresh answer"
    service.gateway = mock_gw

    mock_cache = MagicMock()
    service.cache = mock_cache

    # When query is conversational (or general), with document_ids=None
    res = service.answer_query(
        query="Xin chao trợ lý",
        tenant_id="t1",
        role="admin",
        username="u1",
        document_ids=None,
    )

    # get_exact should NOT be called because unscoped queries bypass cache
    mock_cache.get_exact.assert_not_called()
    mock_cache.set_exact.assert_not_called()
    assert res.answer is not None
