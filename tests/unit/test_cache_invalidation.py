#!/usr/bin/env python3
import pytest
from evaluation.cache_manager import compute_cache_key

def test_cache_key_invalidation():
    base_key = compute_cache_key(
        manifest_hash="abc123hash",
        child_target_tokens=250,
        child_overlap_tokens=30,
        parent_target_tokens=1200,
        parent_overlap_tokens=100,
        dense_model="BAAI/bge-m3",
        dense_dimension=1024,
        query_encoding_protocol="v1_normalized",
        bm25_config_version="v1_alphanumeric",
        rrf_k=60,
        broad_candidate_pool_size=100,
        structural_metadata_version="v1",
    )

    # 1. Change manifest hash
    k_manifest = compute_cache_key(
        manifest_hash="diff_hash",
        child_target_tokens=250,
        child_overlap_tokens=30,
        parent_target_tokens=1200,
        parent_overlap_tokens=100,
        dense_model="BAAI/bge-m3",
    )
    assert k_manifest != base_key

    # 2. Change chunk size
    k_chunk = compute_cache_key(
        manifest_hash="abc123hash",
        child_target_tokens=300,
        child_overlap_tokens=30,
        parent_target_tokens=1200,
        parent_overlap_tokens=100,
        dense_model="BAAI/bge-m3",
    )
    assert k_chunk != base_key

    # 3. Change dense model
    k_dense = compute_cache_key(
        manifest_hash="abc123hash",
        child_target_tokens=250,
        child_overlap_tokens=30,
        parent_target_tokens=1200,
        parent_overlap_tokens=100,
        dense_model="BAAI/bge-small-en-v1.5",
    )
    assert k_dense != base_key

    # 4. Change RRF k
    k_rrf = compute_cache_key(
        manifest_hash="abc123hash",
        rrf_k=100,
    )
    assert k_rrf != base_key
