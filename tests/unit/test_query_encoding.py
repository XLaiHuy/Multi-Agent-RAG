#!/usr/bin/env python3
import pytest
import numpy as np
from backend.app.providers.embeddings import LocalEmbeddingProvider

def test_query_encoding_equivalence():
    provider = LocalEmbeddingProvider()
    queries = ["What is the governing law?", "Termination conditions for default."]

    single_vecs = [provider.embed_query(q) for q in queries]
    batch_vecs = provider.embed_queries_batch(queries)

    diff = np.abs(np.array(single_vecs) - np.array(batch_vecs))
    assert np.max(diff) < 1e-5
