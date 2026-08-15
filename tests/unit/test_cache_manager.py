#!/usr/bin/env python3
import pytest
import numpy as np
from pathlib import Path
from evaluation.cache_manager import compute_cache_key, EvaluationCache


def test_compute_cache_key_deterministic():
    k1 = compute_cache_key("hash123", 250, 30, 1200, 100, "BAAI/bge-m3")
    k2 = compute_cache_key("hash123", 250, 30, 1200, 100, "BAAI/bge-m3")
    assert k1 == k2
    assert len(k1) == 24

    # Any change invalidates key
    k3 = compute_cache_key("hash123", 300, 30, 1200, 100, "BAAI/bge-m3")
    assert k1 != k3

    k4 = compute_cache_key("hash123", 250, 30, 1200, 100, "BAAI/bge-small-en-v1.5")
    assert k1 != k4


def test_cache_save_and_load(tmp_path, monkeypatch):
    import evaluation.cache_manager as cm
    monkeypatch.setattr(cm, "CACHE_DIR", tmp_path)

    cache = EvaluationCache("test_key_abc")
    assert not cache.is_complete()

    # Save mock artifacts
    mock_chunks = [{"chunk_id": "c1", "doc_id": "d1", "text": "hello"}]
    cache.save_corpus_chunks(mock_chunks)
    assert cache.load_corpus_chunks() == mock_chunks

    mock_emb = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    mock_ids = ["c1", "c2"]
    cache.save_dense_embeddings(mock_emb, mock_ids)
    loaded_emb, loaded_ids = cache.load_dense_embeddings()
    assert np.allclose(mock_emb, loaded_emb)
    assert loaded_ids == mock_ids

    cache.save_query_embeddings(np.array([[0.5, 0.6]]), ["q1"])
    cache.save_retrieval_candidates(
        bm25_top100={"0": [("c1", 1.0)]},
        dense_top100={"0": [("c1", 0.9)]},
        rrf_top100={"0": [("c1", 0.03)]},
        gold_mapping={"0": ["c1"]},
    )
    cache.save_metadata({"model": "test"})

    assert cache.is_complete()
