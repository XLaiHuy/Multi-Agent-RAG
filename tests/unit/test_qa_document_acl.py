"""
Unit Tests for Document ACL Enforcement on QA Endpoints and Scope Resolution.
Verifies fail-closed behavior at the API boundary, rejecting unauthorized document IDs
and resolving accessible document scopes per tenant and role.
"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi import HTTPException
from sqlalchemy.orm import Session

from backend.app.api.qa_routes import resolve_authorized_document_scope
from backend.app.domain.models import Document, DocumentACL
from backend.app.persistence.database import DocumentRepository


@pytest.fixture
def mock_db_session():
    """Mock SQLite database session with controlled Document & DocumentACL entries."""
    db = MagicMock(spec=Session)
    return db


def test_chat_rejects_same_tenant_document_without_role_acl(mock_db_session):
    """When a same-tenant user requests a document their role cannot read, resolve_authorized_document_scope raises 404."""
    with patch.object(DocumentRepository, "get_document_if_accessible", return_value=None) as mock_get:
        with pytest.raises(HTTPException) as exc_info:
            resolve_authorized_document_scope(
                db=mock_db_session,
                requested_document_ids=["doc_finance_secret"],
                tenant_id="default_tenant",
                role="hr",
            )
        assert exc_info.value.status_code == 404
        assert "not found or access denied" in exc_info.value.detail
        mock_get.assert_called_once_with(mock_db_session, doc_id="doc_finance_secret", tenant_id="default_tenant", role="hr")


def test_stream_rejects_same_tenant_document_without_role_acl(mock_db_session):
    """Streaming endpoint scope resolution also fails closed with HTTP 404 for unauthorized doc."""
    with patch.object(DocumentRepository, "get_document_if_accessible", return_value=None):
        with pytest.raises(HTTPException) as exc_info:
            resolve_authorized_document_scope(
                db=mock_db_session,
                requested_document_ids=["unauthorized_doc"],
                tenant_id="tenant_x",
                role="user",
            )
        assert exc_info.value.status_code == 404


def test_explicit_multi_doc_scope_fails_if_one_document_is_unauthorized(mock_db_session):
    """If requested scope is [allowed_doc, forbidden_doc], MUST fail closed and not execute partially."""
    def fake_get_accessible(db, doc_id, tenant_id, role):
        if doc_id == "doc_allowed":
            return Document(id="doc_allowed", tenant_id=tenant_id, filename="allowed.md")
        return None

    with patch.object(DocumentRepository, "get_document_if_accessible", side_effect=fake_get_accessible):
        with pytest.raises(HTTPException) as exc_info:
            resolve_authorized_document_scope(
                db=mock_db_session,
                requested_document_ids=["doc_allowed", "doc_forbidden"],
                tenant_id="tenant_1",
                role="legal",
            )
        assert exc_info.value.status_code == 404
        assert "doc_forbidden" in exc_info.value.detail


def test_unscoped_chat_resolves_only_role_accessible_documents(mock_db_session):
    """Unscoped query (document_ids=None) resolves all documents matching tenant and role ACL."""
    mock_docs = [
        Document(id="doc_public_1", tenant_id="t1", filename="pub1.pdf"),
        Document(id="doc_legal_1", tenant_id="t1", filename="legal1.pdf"),
    ]
    with patch.object(DocumentRepository, "list_accessible_documents", return_value=mock_docs):
        resolved = resolve_authorized_document_scope(
            db=mock_db_session,
            requested_document_ids=None,
            tenant_id="t1",
            role="legal",
        )
        assert resolved == ["doc_public_1", "doc_legal_1"]


def test_zero_accessible_documents_produces_empty_scope(mock_db_session):
    """When a user has no accessible documents, unscoped query resolves to empty list []."""
    with patch.object(DocumentRepository, "list_accessible_documents", return_value=[]):
        resolved = resolve_authorized_document_scope(
            db=mock_db_session,
            requested_document_ids=[],
            tenant_id="t1",
            role="restricted_user",
        )
        assert resolved == []


def test_cache_identity_changes_when_accessible_corpus_changes():
    """When a new accessible document G is uploaded and resolved scope changes [A, C, F] -> [A, C, F, G], cache identity changes."""
    from backend.app.persistence.cache import build_query_cache_identity

    scope_before = ["doc_A", "doc_C", "doc_F"]
    scope_after = ["doc_A", "doc_C", "doc_F", "doc_G"]

    id_before = build_query_cache_identity("t1", "legal", document_ids=scope_before)
    id_after = build_query_cache_identity("t1", "legal", document_ids=scope_after)

    assert id_before != id_after
