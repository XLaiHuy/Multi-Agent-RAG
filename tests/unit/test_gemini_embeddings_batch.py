"""
Unit Tests for GeminiEmbeddingProvider Batch Embedding and Configured Output Dimension.
Verifies that GeminiEmbeddingProvider strictly uses Google GenAI Client, preserves query task prefix,
and accurately derives effective embedding dimension.
"""
import pytest
from unittest.mock import MagicMock, patch
from backend.app.providers.embeddings import GeminiEmbeddingProvider
from backend.app.core.config import Settings
from backend.app.persistence.cache import get_effective_embedding_identity


def test_gemini_embed_queries_batch_uses_gemini_client():
    """embed_queries_batch MUST call client.models.embed_content with task: contract retrieval prefix and output_dimensionality."""
    mock_settings = Settings(
        GEMINI_API_KEY="test_key_abc",
        GEMINI_EMBEDDING_MODEL="text-embedding-004",
        EMBEDDING_DIMENSION=512,
    )

    mock_client = MagicMock()
    mock_item1 = MagicMock()
    mock_item1.values = [0.1] * 512
    mock_item2 = MagicMock()
    mock_item2.values = [0.2] * 512
    mock_res = MagicMock()
    mock_res.embeddings = [mock_item1, mock_item2]
    mock_client.models.embed_content.return_value = mock_res

    with patch("backend.app.providers.embeddings.genai.Client", return_value=mock_client):
        provider = GeminiEmbeddingProvider(settings=mock_settings)
        vectors = provider.embed_queries_batch(["Query 1", "Query 2"], batch_size=10)

        assert len(vectors) == 2
        assert len(vectors[0]) == 512
        assert vectors[0][0] == 0.1

        # Verify arguments passed to Gemini Client
        mock_client.models.embed_content.assert_called_once()
        call_kwargs = mock_client.models.embed_content.call_args[1]
        assert call_kwargs["model"] == "text-embedding-004"
        assert call_kwargs["contents"] == [
            "task: contract retrieval | query: Query 1",
            "task: contract retrieval | query: Query 2",
        ]
        assert call_kwargs["config"].output_dimensionality == 512


def test_gemini_embed_queries_batch_does_not_call_local_model():
    """GeminiEmbeddingProvider must NEVER attempt to load or call SentenceTransformer."""
    mock_settings = Settings(
        GEMINI_API_KEY="test_key_abc",
        GEMINI_EMBEDDING_MODEL="text-embedding-004",
        EMBEDDING_DIMENSION=384,
    )
    mock_client = MagicMock()
    mock_item = MagicMock()
    mock_item.values = [0.5] * 384
    mock_res = MagicMock()
    mock_res.embeddings = [mock_item]
    mock_client.models.embed_content.return_value = mock_res

    with patch("backend.app.providers.embeddings.genai.Client", return_value=mock_client):
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            provider = GeminiEmbeddingProvider(settings=mock_settings)
            provider.embed_queries_batch(["Test query"])
            mock_st.assert_not_called()


def test_gemini_embedding_identity_uses_configured_output_dimension():
    """Effective identity must reflect the actual configured output dimension (e.g. 384 vs 768)."""
    settings_384 = Settings(
        EMBEDDING_PROVIDER="gemini",
        GEMINI_EMBEDDING_MODEL="text-embedding-004",
        EMBEDDING_DIMENSION=384,
    )
    settings_768 = Settings(
        EMBEDDING_PROVIDER="gemini",
        GEMINI_EMBEDDING_MODEL="text-embedding-004",
        EMBEDDING_DIMENSION=768,
    )

    id_384 = get_effective_embedding_identity(settings_384)
    id_768 = get_effective_embedding_identity(settings_768)

    assert id_384 == "gemini::text-embedding-004::384"
    assert id_768 == "gemini::text-embedding-004::768"
    assert id_384 != id_768
