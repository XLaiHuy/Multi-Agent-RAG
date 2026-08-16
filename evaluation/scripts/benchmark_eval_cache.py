#!/usr/bin/env python3
"""
Committed Benchmark Runner for Evaluation Cache Acceleration.
Runs identical cold vs warm workload on canonical CUAD DEV contracts to verify:
1. Speedup ratio (timing in seconds).
2. Exact ranking and metric identity (SHA-256 fingerprint matching).
Saves: evaluation/results/phase4_2/cache_speedup_verified.json
"""
import os
import sys
import time
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from rank_bm25 import BM25Okapi

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
torch.set_num_threads(4)

from evaluation.cache_manager import EvaluationCache, compute_cache_key
from evaluation.config_loader import get_retrieval_config
from backend.app.providers.reranker import LocalCrossEncoderReranker
from backend.app.providers.embeddings import LocalEmbeddingProvider
from backend.app.ingestion.parsers import MasterDocumentParser
from backend.app.ingestion.chunker import StructureAwareParentChildChunker
from backend.app.retrieval.bm25 import tokenize_for_bm25
from backend.app.retrieval.fusion import reciprocal_rank_fusion
from evaluation.metrics.retrieval_metrics import compute_candidate_hit_rate_at_k, compute_reciprocal_rank

cfg = get_retrieval_config()
RESULTS_DIR = REPO_ROOT / "evaluation" / "results" / "phase4_2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CONTRACTS_DIR = REPO_ROOT / "evaluation" / "datasets" / "cuad" / "processed" / "contracts"


def run_workload(
    chunks_data: List[Dict[str, Any]],
    dense_emb: np.ndarray,
    q_emb: np.ndarray,
    ans_queries: List[Dict[str, Any]],
    reranker: LocalCrossEncoderReranker,
    query_indices: List[int]
) -> List[Dict[str, Any]]:
    """Runs identical retrieval and reranking workload across queries."""
    doc_to_chunk_indices: Dict[str, List[int]] = {}
    doc_to_bm25: Dict[str, BM25Okapi] = {}
    chunk_map = {c["chunk_id"]: c for c in chunks_data}

    for idx, c in enumerate(chunks_data):
        doc_id = c["doc_id"]
        doc_to_chunk_indices.setdefault(doc_id, []).append(idx)

    for doc_id, indices in doc_to_chunk_indices.items():
        doc_chunks = [chunks_data[i] for i in indices]
        tokenized = [tokenize_for_bm25(c.get("enriched_text", c["text"])) for c in doc_chunks]
        doc_to_bm25[doc_id] = BM25Okapi(tokenized)

    results = []
    for q_idx in query_indices:
        q = ans_queries[q_idx]
        question = q["question"]
        target_doc_id = q["source_contract_id"]

        doc_indices = doc_to_chunk_indices.get(target_doc_id, [])
        if not doc_indices:
            continue
        scoped_chunks = [chunks_data[i] for i in doc_indices]
        scoped_chunk_ids = [c["chunk_id"] for c in scoped_chunks]

        scoped_dense_sims = np.dot(dense_emb[doc_indices], q_emb[q_idx])
        s_dense_top = [scoped_chunk_ids[i] for i in np.argsort(-scoped_dense_sims)[:20]]

        bm25_scores = doc_to_bm25[target_doc_id].get_scores(tokenize_for_bm25(question))
        s_bm25_top = [scoped_chunk_ids[i] for i in np.argsort(-bm25_scores)[:20]]

        s_rrf_candidates = [cid for cid, _ in reciprocal_rank_fusion([s_dense_top, s_bm25_top], k=cfg.rrf_k)]

        # Parent Dedup
        s_dedup = []
        s_p_count: Dict[str, int] = {}
        for cid in s_rrf_candidates:
            c_obj = chunk_map.get(cid)
            pid = c_obj.get("parent_id") if c_obj else None
            if pid:
                if s_p_count.get(pid, 0) >= 2: continue
                s_p_count[pid] = s_p_count.get(pid, 0) + 1
            s_dedup.append(cid)

        s_budget_20 = s_dedup[:20]
        s_cand_texts = [chunk_map[cid]["text"] for cid in s_budget_20]
        s_rerank_res = reranker.rerank(question, s_cand_texts, top_n=10)
        s_final_ids = [s_budget_20[orig_idx] for orig_idx, _ in s_rerank_res]

        results.append({
            "query_index": q_idx,
            "final_top10": s_final_ids
        })
    return results


def main():
    print("=" * 80)
    print("PHASE 4.2: CACHE SPEEDUP BENCHMARK & RESULT IDENTITY VERIFICATION")
    print("=" * 80)

    manifest_path = REPO_ROOT / "evaluation" / "manifests" / "cuad_dev_manifest.json"
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    contracts_info = manifest_data["contracts"][:3] # 3 canonical DEV contracts
    target_contract_ids = {c["source_contract_id"] for c in contracts_info}
    queries = manifest_data["queries"]
    ans_queries = [q for q in queries if not q.get("is_unanswerable", False) and q.get("source_contract_id") in target_contract_ids]

    workload_indices = list(range(len(ans_queries)))
    print(f"Target Workload: {len(contracts_info)} Contracts, {len(workload_indices)} Answerable Queries")

    reranker = LocalCrossEncoderReranker(
        model_name="cross-encoder/ms-marco-TinyBERT-L-2-v2", max_length=512, strict=True
    )
    emb_provider = LocalEmbeddingProvider(model_name=cfg.dense_model)
    _ = reranker._get_model()
    _ = emb_provider._get_model()

    # 1. Warm Execution (From Cache Key 1a9ef6e99dbb234ff50bcd7e filtered to the 3 target contracts)
    cache = EvaluationCache("1a9ef6e99dbb234ff50bcd7e")
    t_start_warm = time.perf_counter()
    full_warm_chunks = cache.load_corpus_chunks()
    full_warm_dense, _ = cache.load_dense_embeddings()
    
    # Filter warm cache to target 3 contracts
    target_chunk_indices = [i for i, c in enumerate(full_warm_chunks) if c["doc_id"] in target_contract_ids]
    warm_chunks = [full_warm_chunks[i] for i in target_chunk_indices]
    warm_dense = full_warm_dense[target_chunk_indices]
    
    # Pre-encoded query embeddings from cache
    full_q_emb, _ = cache.load_query_embeddings()
    dev_ans_all = [q for q in queries if not q.get("is_unanswerable", False)]
    target_q_indices_in_manifest = [i for i, q in enumerate(dev_ans_all) if q.get("source_contract_id") in target_contract_ids]
    warm_q_emb = full_q_emb[target_q_indices_in_manifest]

    warm_results = run_workload(warm_chunks, warm_dense, warm_q_emb, ans_queries, reranker, workload_indices)
    warm_runtime_seconds = float(time.perf_counter() - t_start_warm)

    # 2. Cold Execution (Full Document Parsing + Chunking + Dense Embedding + Query Embedding + Retrieval + Reranking)
    t_start_cold = time.perf_counter()
    chunker = StructureAwareParentChildChunker(
        child_target_tokens=cfg.child_target_tokens,
        child_overlap_tokens=cfg.child_overlap_tokens,
        parent_target_tokens=cfg.parent_target_tokens,
        parent_overlap_tokens=cfg.parent_overlap_tokens,
    )
    cold_all_texts, cold_chunks = [], []
    for c_info in contracts_info:
        fpath = CONTRACTS_DIR / c_info["filename"]
        if not fpath.exists(): fpath = CONTRACTS_DIR / c_info["filename"].replace(".md", ".txt")
        doc = MasterDocumentParser.parse(fpath, doc_id=c_info["source_contract_id"])
        c_chunks, _ = chunker.chunk_canonical_document(doc, doc_version=1)
        doc_title = c_info.get("original_title", "").replace("_", " ").replace("-", " ")
        for c in c_chunks:
            sec_str = " > ".join(c.section_path) if c.section_path else "General"
            enriched = f"[Document: {doc_title}] [Section: {sec_str}]\n{c.text}"
            cold_all_texts.append(enriched)
            cold_chunks.append({
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "parent_id": c.parent_id,
                "text": c.text,
                "enriched_text": enriched
            })
    cold_dense_list = emb_provider.embed_documents_batch(cold_all_texts, batch_size=32)
    cold_dense = np.array(cold_dense_list, dtype=np.float32)
    
    cold_q_texts = [ans_queries[i]["question"] for i in range(len(ans_queries))]
    cold_q_emb_list = emb_provider.embed_queries_batch(cold_q_texts, batch_size=32)
    cold_q_emb = np.array(cold_q_emb_list, dtype=np.float32)
    
    cold_results = run_workload(cold_chunks, cold_dense, cold_q_emb, ans_queries, reranker, workload_indices)
    cold_runtime_seconds = float(time.perf_counter() - t_start_cold)

    # 3. Fingerprint Verification
    warm_json = json.dumps(warm_results, sort_keys=True)
    cold_json = json.dumps(cold_results, sort_keys=True)
    warm_result_hash = hashlib.sha256(warm_json.encode("utf-8")).hexdigest()
    cold_result_hash = hashlib.sha256(cold_json.encode("utf-8")).hexdigest()
    is_identical = (warm_result_hash == cold_result_hash)

    speedup_ratio = float(cold_runtime_seconds / max(1e-6, warm_runtime_seconds))

    cache_benchmark_data = {
        "benchmark_name": "EVALUATION_CACHE_SPEEDUP_BENCHMARK",
        "workload_description": f"{len(ans_queries)} CUAD DEV queries across {len(contracts_info)} contracts ({len(cold_chunks)} chunks) evaluated under true document-scoped hybrid retrieval + TinyBERT reranker",
        "workload_contracts_count": len(contracts_info),
        "workload_chunks_count": len(cold_chunks),
        "workload_queries_count": len(workload_indices),
        "cold_runtime_seconds": round(cold_runtime_seconds, 2),
        "warm_runtime_seconds": round(warm_runtime_seconds, 4),
        "speedup_ratio": round(speedup_ratio, 2),
        "cold_result_hash": cold_result_hash,
        "warm_result_hash": warm_result_hash,
        "result_identity_verified": is_identical,
        "timing_unit": "seconds",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    out_path = RESULTS_DIR / "cache_speedup_verified.json"
    out_path.write_text(json.dumps(cache_benchmark_data, indent=2), encoding="utf-8")
    print(f"  [OK] Saved {out_path.name}")
    print(f"  Cold Runtime: {cold_runtime_seconds:.2f} s")
    print(f"  Warm Runtime: {warm_runtime_seconds:.4f} s")
    print(f"  Speedup Ratio: {speedup_ratio:.2f}x")
    print(f"  Result Hash Match: {'YES (Exact Match)' if is_identical else 'NO (MISMATCH)'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
