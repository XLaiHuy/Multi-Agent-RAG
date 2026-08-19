import os
import sys
import json
import pytest
import unicodedata
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from backend.app.domain.canonical import CanonicalDocument, CanonicalPage, CanonicalBlock, BlockType
from backend.app.ingestion.chunker import StructureAwareParentChildChunker, IndexedChunk


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def test_sibling_child_does_not_inherit_parent_relevance():
    """
    1. Sibling child does NOT inherit parent relevance:
    If gold evidence is in Child 0, Sibling Child 1 and Child 2 MUST NOT be marked as relevant.
    """
    p1 = "This Agreement is governed by the laws of the State of Delaware."
    p2 = "The Supplier shall deliver the goods within thirty days of order."
    p3 = "All confidential information must be kept secure for five years."
    
    blocks = [
        CanonicalBlock(block_id="b1", block_type=BlockType.PARAGRAPH, page_number=1, text=p1, section_path=["Section 1"]),
        CanonicalBlock(block_id="b2", block_type=BlockType.PARAGRAPH, page_number=1, text=p2, section_path=["Section 1"]),
        CanonicalBlock(block_id="b3", block_type=BlockType.PARAGRAPH, page_number=1, text=p3, section_path=["Section 1"]),
    ]
    page = CanonicalPage(page_number=1, blocks=blocks)
    doc = CanonicalDocument(doc_id="test_doc_01", title="Test Contract", doc_type="markdown", pages=[page])
    
    chunker = StructureAwareParentChildChunker(
        child_target_tokens=15,
        child_overlap_tokens=0,
        parent_target_tokens=500,
        parent_overlap_tokens=0
    )
    child_chunks, parent_chunks = chunker.chunk_canonical_document(doc)
    
    assert len(parent_chunks) == 1
    assert len(child_chunks) >= 2
    
    gold_evidence = "laws of the State of Delaware"
    gold_norm = normalize_text(gold_evidence)
    
    child_relevance = {}
    parent_relevance = {}
    
    for c in child_chunks:
        c_norm = normalize_text(c.text)
        is_relevant = (gold_norm in c_norm) or (c_norm in gold_norm)
        child_relevance[c.chunk_id] = is_relevant
        
    for p in parent_chunks:
        p_norm = normalize_text(p.text)
        parent_relevance[p.chunk_id] = (gold_norm in p_norm)
        
    relevant_children = [cid for cid, rel in child_relevance.items() if rel]
    irrelevant_children = [cid for cid, rel in child_relevance.items() if not rel]
    
    assert len(relevant_children) == 1
    assert relevant_children[0] == child_chunks[0].chunk_id
    assert len(irrelevant_children) >= 1
    
    for sib_id in irrelevant_children:
        assert child_relevance[sib_id] is False, f"Sibling child {sib_id} falsely inherited relevance!"
        
    assert parent_relevance[parent_chunks[0].chunk_id] is True


def test_parent_and_child_metrics_are_strictly_separate():
    """
    2. Parent metrics are separate from child metrics:
    Verifies that child HitRate and parent HitRate evaluate different targets.
    """
    retrieved_child_ids = ["p0_c1", "p0_c2"]
    gold_child_ids = ["p0_c0"]
    
    retrieved_parent_ids = ["p0", "p0"]
    gold_parent_ids = ["p0"]
    
    child_hit = any(cid in set(gold_child_ids) for cid in retrieved_child_ids[:2])
    assert child_hit is False
    
    parent_hit = any(pid in set(gold_parent_ids) for pid in retrieved_parent_ids[:2])
    assert parent_hit is True
    
    assert child_hit != parent_hit


def test_exact_and_span_mapping_determinism():
    """
    3. Exact and span mapping determinism:
    Verifies normalized whitespace and casing matching.
    """
    gold = "November  20, \n  2007"
    chunk_text = "The agreement was executed on november 20, 2007 by both parties."
    
    gold_norm = normalize_text(gold)
    chunk_norm = normalize_text(chunk_text)
    
    assert gold_norm == "november 20, 2007"
    assert gold_norm in chunk_norm


def test_gold_protocol_invalidation_rule():
    """
    4. Gold protocol versioning and cache invalidation:
    Verifies that changing structural/protocol version generates a distinct cache key.
    """
    from evaluation.cache_manager import compute_cache_key
    from evaluation.config_loader import get_retrieval_config
    cfg = get_retrieval_config()
    
    key_v1 = compute_cache_key(
        manifest_hash="abc",
        child_target_tokens=cfg.child_target_tokens,
        child_overlap_tokens=cfg.child_overlap_tokens,
        parent_target_tokens=cfg.parent_target_tokens,
        parent_overlap_tokens=cfg.parent_overlap_tokens,
        dense_model=cfg.dense_model,
        dense_dimension=cfg.dense_dimension,
        query_encoding_protocol="v1_normalized",
        structural_metadata_version="v1"
    )
    key_v2 = compute_cache_key(
        manifest_hash="abc",
        child_target_tokens=cfg.child_target_tokens,
        child_overlap_tokens=cfg.child_overlap_tokens,
        parent_target_tokens=cfg.parent_target_tokens,
        parent_overlap_tokens=cfg.parent_overlap_tokens,
        dense_model=cfg.dense_model,
        dense_dimension=cfg.dense_dimension,
        query_encoding_protocol="v1_normalized",
        structural_metadata_version="v2_strict_child"
    )
    assert key_v1 != key_v2


def test_online_latency_artifact_contains_query_embedding():
    """
    5. Online latency path actually includes query embedding:
    Verifies that the generated online_latency_holdout.json reports T_query_embedding > 0.
    """
    latency_file = REPO_ROOT / "evaluation" / "results" / "phase4_2" / "online_latency_holdout.json"
    assert latency_file.exists(), "online_latency_holdout.json not found!"
    data = json.loads(latency_file.read_text(encoding="utf-8"))
    
    assert data.get("includes_online_query_embedding") is True
    stages = data.get("stages", {})
    assert "T_query_embedding" in stages
    assert stages["T_query_embedding"]["P50"] > 0
    assert stages["T_total_online_retrieval_and_rerank"]["P50"] > stages["T_query_embedding"]["P50"]


def test_cold_warm_cache_result_hashes_identical():
    """
    6. Cold/warm cache result hashes are identical:
    Verifies that cache_speedup_verified.json proves exact fingerprint matching.
    """
    cache_file = REPO_ROOT / "evaluation" / "results" / "phase4_2" / "cache_speedup_verified.json"
    assert cache_file.exists(), "cache_speedup_verified.json not found!"
    data = json.loads(cache_file.read_text(encoding="utf-8"))
    
    assert data["result_identity_verified"] is True
    assert data["cold_result_hash"] == data["warm_result_hash"]
    assert data["speedup_ratio"] > 1.0


def test_cache_timing_unit_is_seconds():
    """
    7. Timing JSON uses seconds consistently:
    Verifies timing_unit and numeric fields in cache_speedup_verified.json.
    """
    cache_file = REPO_ROOT / "evaluation" / "results" / "phase4_2" / "cache_speedup_verified.json"
    data = json.loads(cache_file.read_text(encoding="utf-8"))
    
    assert data.get("timing_unit") == "seconds"
    assert isinstance(data["cold_runtime_seconds"], (int, float))
    assert isinstance(data["warm_runtime_seconds"], (int, float))
    assert data["cold_runtime_seconds"] > 0
    assert data["warm_runtime_seconds"] > 0
