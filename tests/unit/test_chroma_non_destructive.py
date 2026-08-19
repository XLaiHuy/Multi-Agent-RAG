"""
Unit Tests for ChromaDB Non-Destructive Initialization Invariant.
Verifies that arbitrary or unknown initialization exceptions NEVER call delete_collection
and fail loudly while preserving existing persistent data.
"""
import pytest
from unittest.mock import MagicMock, patch
from backend.app.retrieval.dense import DenseRetriever


def test_chroma_init_unknown_error_never_deletes_collection(monkeypatch):
    """An unexpected Chroma initialization exception MUST NOT call delete_collection()."""
    mock_chroma_client = MagicMock()
    mock_chroma_client.get_or_create_collection.side_effect = Exception("Disk I/O error or permission denied")

    mock_embedder = MagicMock()
    mock_embedder.dimension = 384

    with patch("backend.app.retrieval.dense.chromadb.PersistentClient", return_value=mock_chroma_client):
        with pytest.raises(RuntimeError, match="Preserving existing data without destructive recovery"):
            DenseRetriever(embedding_provider=mock_embedder)

    # Assert delete_collection was NEVER called
    mock_chroma_client.delete_collection.assert_not_called()
