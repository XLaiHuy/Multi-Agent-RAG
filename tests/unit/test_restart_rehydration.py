#!/usr/bin/env python3
"""
Unit tests for production startup and restart BM25 rehydration semantics.
"""
import pytest
from unittest.mock import patch, MagicMock
from backend.app.persistence.database import init_database


def test_production_init_database_skips_demo_user_seeding():
    with patch("backend.app.persistence.database.settings") as mock_settings, \
         patch("backend.app.persistence.database.Base.metadata.create_all"), \
         patch("backend.app.persistence.database.SessionLocal") as mock_session_local, \
         patch("backend.app.ingestion.pipeline.get_ingestion_pipeline") as mock_get_pipeline:

        mock_settings.environment = "production"
        mock_db = MagicMock()
        mock_session_local.return_value.__enter__.return_value = mock_db

        mock_pipeline = MagicMock()
        mock_pipeline.bm25.bm25 = "non_empty_index"
        mock_pipeline.bm25.chunk_ids = ["c1"]
        mock_get_pipeline.return_value = mock_pipeline

        init_database()

        # In production, db.add_all should NOT be called for demo users
        mock_db.add_all.assert_not_called()


def test_restart_rehydrates_bm25_from_chroma_collection():
    with patch("backend.app.persistence.database.settings") as mock_settings, \
         patch("backend.app.persistence.database.Base.metadata.create_all"), \
         patch("backend.app.persistence.database.SessionLocal") as mock_session_local, \
         patch("backend.app.ingestion.pipeline.get_ingestion_pipeline") as mock_get_pipeline:

        mock_settings.environment = "production"
        mock_db = MagicMock()
        mock_session_local.return_value.__enter__.return_value = mock_db

        mock_pipeline = MagicMock()
        # BM25 is initially empty (after container restart)
        mock_pipeline.bm25.bm25 = None
        mock_pipeline.bm25.chunk_ids = []
        # Persistent Chroma collection has 2 chunks
        mock_pipeline.dense.collection.get.return_value = {
            "ids": ["doc1_c1", "doc1_c2"],
            "documents": ["chunk text 1", "chunk text 2"],
            "metadatas": [{"tenant_id": "t1", "doc_id": "doc1"}, {"tenant_id": "t1", "doc_id": "doc1"}],
        }
        mock_get_pipeline.return_value = mock_pipeline

        init_database()

        # BM25 build_index should be called with chunks retrieved from ChromaDB
        mock_pipeline.bm25.build_index.assert_called_once_with(
            chunk_ids=["doc1_c1", "doc1_c2"],
            documents=["chunk text 1", "chunk text 2"],
            metadatas=[{"tenant_id": "t1", "doc_id": "doc1"}, {"tenant_id": "t1", "doc_id": "doc1"}],
        )
