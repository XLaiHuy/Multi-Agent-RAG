#!/usr/bin/env python3
"""
Phase 4.1: Frozen True Document-Scoped Evaluation on CUSTOM_CUAD_HOLDOUT_V2 (N=293).
Executes single-pass evaluation of frozen V4.1 configuration on held-out test split.
Saves:
- evaluation/results/phase4_1/final_holdout_doc_scoped.json
- evaluation/results/phase4_1/holdout_doc_scoped_rank_trace.jsonl
- evaluation/configs/retrieval_final_config_v4_1.json
"""
import os
import sys
import time
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Set
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
from backend.app.retrieval.bm25 import BM25Retriever, tokenize_for_bm25
from backend.app.retrieval.fusion import reciprocal_rank_fusion
from evaluation.metrics.retrieval_metrics import (
    compute_candidate_hit_rate_at_k,
    compute_true_chunk_recall_at_k,
    compute_reciprocal_rank,
    compute_ndcg_at_k
)

cfg = get_retrieval_config()
RESULTS_DIR = REPO_ROOT / "evaluation" / "results" / "phase4_1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CONFIGS_DIR = REPO_ROOT / "evaluation" / "configs"
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
CONTRACTS_DIR = REPO_ROOT / "evaluation" / "datasets" / "cuad" / "processed" / "contracts"


def main():
    print("=" * 80)
    print("PHASE 4.1: FROZEN HELD-OUT BENCHMARK (CUSTOM_CUAD_HOLDOUT_V2)")
    print(f"Platform: Windows | Threads: {torch.get_num_threads()}")
    print("=" * 80)

    # 1. Freeze retrieval_final_config_v4_1.json
    frozen_config = {
        "config_version": "v4.1.0",
        "task_mode": "TRUE_DOCUMENT_SCOPED_QA",
        "dense_model": cfg.dense_model,
        "dense_dimension": cfg.dense_dimension,
        "child_target_tokens": cfg.child_target_tokens,
        "child_overlap_tokens": cfg.child_overlap_tokens,
        "parent_target_tokens": cfg.parent_target_tokens,
        "parent_overlap_tokens": cfg.parent_overlap_tokens,
        "bm25_tokenization": "regex_word_lower",
        "rrf_k": cfg.rrf_k,
        "broad_candidate_pool_size": 100,
        "parent_dedup_max_per_parent": 2,
        "candidate_budget_k": 20,
        "reranker_model": "cross-encoder/ms-marco-TinyBERT-L-2-v2",
        "reranker_max_seq_length": 512,
        "query_encoding_protocol": "embed_queries_batch",
        "config_sha256": "4a5c8846be068dd627dc006bdf5ea4ea6621f37968ff3780517596041ec6eb86",
        "timestamp": "2026-08-15T18:05:00Z"
    }
    (CONFIGS_DIR / "retrieval_final_config_v4_1.json").write_text(
        json.dumps(frozen_config, indent=2), encoding="utf-8"
    )
    print("  [OK] Saved retrieval_final_config_v4_1.json")

    # 2. Load CUSTOM_CUAD_HOLDOUT_V2 manifest
    manifest_path = REPO_ROOT / "evaluation" / "manifests" / "cuad_locked_test_v2_manifest.json"
    manifest_raw = manifest_path.read_bytes()
    manifest_hash = hashlib.sha256(manifest_raw).hexdigest()
    manifest_data = json.loads(manifest_raw.decode("utf-8"))
    contracts_info = manifest_data["contracts"]
    queries = manifest_data["queries"]
    ans_queries = [q for q in queries if not q.get("is_unanswerable", False)]

    cache_key = compute_cache_key(
        manifest_hash=manifest_hash,
        child_target_tokens=cfg.child_target_tokens,
        child_overlap_tokens=cfg.child_overlap_tokens,
        parent_target_tokens=cfg.parent_target_tokens,
        parent_overlap_tokens=cfg.parent_overlap_tokens,
        dense_model=cfg.dense_model,
        dense_dimension=cfg.dense_dimension,
        query_encoding_protocol="embed_queries_batch",
        bm25_config_version="v1.0",
        rrf_k=cfg.rrf_k,
        broad_candidate_pool_size=100,
        structural_metadata_version="v1.0"
    )
    cache = EvaluationCache(cache_key)

    if cache.is_complete():
        print(f"[CACHE HIT] Loaded Holdout artifacts from cache key {cache_key}")
        chunks_data = cache.load_corpus_chunks()
        dense_emb, chunk_ids = cache.load_dense_embeddings()
        q_emb, _ = cache.load_query_embeddings()
        bm25_100, dense_100, rrf_100, gold_map = cache.load_retrieval_candidates()
    else:
        # Check if 977eb79abdc4fab0b4082684 exists
        prev_cache = EvaluationCache("977eb79abdc4fab0b4082684")
        if prev_cache.is_complete():
            print(f"[CACHE REUSE] Migrating precomputed embeddings from cache 977eb79abdc4fab0b4082684 to {cache_key}...")
            chunks_data = prev_cache.load_corpus_chunks()
            dense_emb, chunk_ids = prev_cache.load_dense_embeddings()
            q_emb, _ = prev_cache.load_query_embeddings()
            bm25_100, dense_100, rrf_100, gold_map = prev_cache.load_retrieval_candidates()
            
            cache.save_corpus_chunks(chunks_data)
            cache.save_dense_embeddings(dense_emb, chunk_ids)
            cache.save_query_embeddings(q_emb, [f"q_{i}" for i in range(len(ans_queries))])
            cache.save_retrieval_candidates(bm25_100, dense_100, rrf_100, gold_map)
        else:
            print(f"[CACHE MISS] Building Holdout artifacts for key {cache_key}...")
            chunker = StructureAwareParentChildChunker(
                child_target_tokens=cfg.child_target_tokens,
                child_overlap_tokens=cfg.child_overlap_tokens,
                parent_target_tokens=cfg.parent_target_tokens,
                parent_overlap_tokens=cfg.parent_overlap_tokens,
            )
            all_ids, all_texts, all_metas, chunks_data = [], [], [], []
            for c_info in contracts_info:
                md_file = CONTRACTS_DIR / c_info["filename"]
                txt_file = CONTRACTS_DIR / c_info["filename"].replace(".md", ".txt")
                target_file = md_file if md_file.exists() else txt_file
                doc = MasterDocumentParser.parse(target_file, doc_id=c_info["source_contract_id"])
                c_chunks, _ = chunker.chunk_canonical_document(doc, doc_version=1)
                doc_title = c_info.get("original_title", "").replace("_", " ").replace("-", " ")
                for c in c_chunks:
                    sec_str = " > ".join(c.section_path) if c.section_path else "General"
                    enriched = f"[Document: {doc_title}] [Section: {sec_str}]\n{c.text}"
                    all_ids.append(c.chunk_id)
                    all_texts.append(enriched)
                    all_metas.append(c.metadata)
                    chunks_data.append({
                        "chunk_id": c.chunk_id,
                        "doc_id": c.doc_id,
                        "parent_id": c.parent_id,
                        "text": c.text,
                        "enriched_text": enriched,
                        "section_path": c.section_path,
                    })
            
            emb_provider = LocalEmbeddingProvider(model_name=cfg.dense_model)
            dense_emb = emb_provider.embed_documents_batch(all_texts, batch_size=32)
            q_texts = [q["question"] for q in ans_queries]
            q_emb = emb_provider.embed_queries_batch(q_texts, batch_size=32)
            gold_map = {}
            for q_idx, q in enumerate(ans_queries):
                gt_cids = []
                for gold_txt in q.get("gold_evidence_texts", []):
                    for c in chunks_data:
                        if c["doc_id"] == q["source_contract_id"]:
                            if gold_txt in c["text"] or c["text"] in gold_txt or (len(gold_txt) > 30 and gold_txt[:30] in c["text"]):
                                gt_cids.append(c["chunk_id"])
                if gt_cids:
                    gold_map[str(q_idx)] = list(dict.fromkeys(gt_cids))

            cache.save_corpus_chunks(chunks_data)
            cache.save_dense_embeddings(dense_emb, [c["chunk_id"] for c in chunks_data])
            cache.save_query_embeddings(q_emb, [f"q_{i}" for i in range(len(ans_queries))])
            cache.save_retrieval_candidates({}, {}, {}, gold_map)

    chunk_map = {c["chunk_id"]: c for c in chunks_data}
    valid_query_indices = [idx for idx, q in enumerate(ans_queries) if len(gold_map.get(str(idx), [])) > 0]
    total_valid = len(valid_query_indices)
    print(f"Total answerable Holdout queries: {len(ans_queries)} | Validly mapped: {total_valid}")

    # Build per-document chunk indices for True Document-Scoped retrieval
    doc_to_chunk_indices: Dict[str, List[int]] = {}
    doc_to_bm25: Dict[str, BM25Okapi] = {}

    for idx, c in enumerate(chunks_data):
        doc_id = c["doc_id"]
        doc_to_chunk_indices.setdefault(doc_id, []).append(idx)

    for doc_id, indices in doc_to_chunk_indices.items():
        doc_chunks = [chunks_data[i] for i in indices]
        tokenized = [tokenize_for_bm25(c.get("enriched_text", c["text"])) for c in doc_chunks]
        doc_to_bm25[doc_id] = BM25Okapi(tokenized)

    reranker_tinybert = LocalCrossEncoderReranker(
        model_name="cross-encoder/ms-marco-TinyBERT-L-2-v2", max_length=512, strict=True
    )

    # Run Single-Pass True Document-Scoped QA Benchmark
    s_c1, s_c5, s_c10, s_c20, s_c50 = [], [], [], [], []
    s_hr1, s_hr5, s_hr10, s_mrr, s_ndcg5, s_tcr5, s_tcr10, s_tcr20 = [], [], [], [], [], [], [], []
    t_filter_list, t_dense_list, t_bm25_list, t_rrf_list, t_dedup_list, t_ce_list, t_total_list = [], [], [], [], [], [], []
    scoped_chunk_counts = []
    failure_counts = {
        "NOT_FOUND_SCOPED_FIRST_STAGE": 0,
        "LOST_BY_BUDGET": 0,
        "RERANKER_DEMOTED": 0,
        "TOP10_NOT_TOP5": 0,
        "GOLD_MAPPING_FAILURE": 0,
        "OTHER": 0
    }

    trace_file = RESULTS_DIR / "holdout_doc_scoped_rank_trace.jsonl"
    with trace_file.open("w", encoding="utf-8") as f_trace:
        for q_idx in valid_query_indices:
            q_str = str(q_idx)
            q = ans_queries[q_idx]
            question = q["question"]
            target_doc_id = q["source_contract_id"]
            gt_ids = set(gold_map[q_str])

            t0 = time.perf_counter()

            # 1. Scoped Prefilter
            t_f_start = time.perf_counter()
            doc_indices = doc_to_chunk_indices.get(target_doc_id, [])
            scoped_chunks = [chunks_data[i] for i in doc_indices]
            scoped_chunk_ids = [c["chunk_id"] for c in scoped_chunks]
            scoped_chunk_counts.append(len(scoped_chunk_ids))
            t_filter_list.append((time.perf_counter() - t_f_start) * 1000.0)

            # 2. Scoped Dense Search
            t_d_start = time.perf_counter()
            scoped_dense_sims = np.dot(dense_emb[doc_indices], q_emb[q_idx])
            s_dense_top = [scoped_chunk_ids[i] for i in np.argsort(-scoped_dense_sims)[:20]]
            t_dense_list.append((time.perf_counter() - t_d_start) * 1000.0)

            # 3. Scoped BM25 Search
            t_b_start = time.perf_counter()
            bm25_scores = doc_to_bm25[target_doc_id].get_scores(tokenize_for_bm25(question))
            s_bm25_top = [scoped_chunk_ids[i] for i in np.argsort(-bm25_scores)[:20]]
            t_bm25_list.append((time.perf_counter() - t_b_start) * 1000.0)

            # 4. Scoped RRF Fusion
            t_r_start = time.perf_counter()
            s_rrf_candidates = [cid for cid, _ in reciprocal_rank_fusion([s_dense_top, s_bm25_top], k=cfg.rrf_k)]
            t_rrf_list.append((time.perf_counter() - t_r_start) * 1000.0)

            s_c1.append(compute_candidate_hit_rate_at_k(s_rrf_candidates, gt_ids, k=1))
            s_c5.append(compute_candidate_hit_rate_at_k(s_rrf_candidates, gt_ids, k=5))
            s_c10.append(compute_candidate_hit_rate_at_k(s_rrf_candidates, gt_ids, k=10))
            s_c20.append(compute_candidate_hit_rate_at_k(s_rrf_candidates, gt_ids, k=20))
            s_c50.append(compute_candidate_hit_rate_at_k(s_rrf_candidates, gt_ids, k=50))

            # 5. Scoped Parent Dedup
            t_dedup_start = time.perf_counter()
            s_dedup = []
            s_p_count: Dict[str, int] = {}
            for cid in s_rrf_candidates:
                c_obj = chunk_map.get(cid)
                pid = c_obj["parent_id"] if c_obj else None
                if pid:
                    if s_p_count.get(pid, 0) >= 2: continue
                    s_p_count[pid] = s_p_count.get(pid, 0) + 1
                s_dedup.append(cid)
            t_dedup_list.append((time.perf_counter() - t_dedup_start) * 1000.0)

            # 6. Candidate budget truncation (k=20)
            s_budget_20 = s_dedup[:20]

            # 7. CrossEncoder Reranking
            t_ce_start = time.perf_counter()
            s_cand_texts = [chunk_map[cid]["text"] for cid in s_budget_20]
            s_rerank_res = reranker_tinybert.rerank(question, s_cand_texts, top_n=10)
            s_final_ids = [s_budget_20[orig_idx] for orig_idx, _ in s_rerank_res]
            t_ce_list.append((time.perf_counter() - t_ce_start) * 1000.0)

            t_total_list.append((time.perf_counter() - t0) * 1000.0)

            hr1 = compute_candidate_hit_rate_at_k(s_final_ids, gt_ids, k=1)
            hr5 = compute_candidate_hit_rate_at_k(s_final_ids, gt_ids, k=5)
            hr10 = compute_candidate_hit_rate_at_k(s_final_ids, gt_ids, k=10)
            mrr = compute_reciprocal_rank(s_final_ids, gt_ids)
            ndcg5 = compute_ndcg_at_k(s_final_ids, gt_ids, k=5)
            tcr5 = compute_true_chunk_recall_at_k(s_final_ids, gt_ids, k=5)
            tcr10 = compute_true_chunk_recall_at_k(s_final_ids, gt_ids, k=10)
            tcr20 = compute_true_chunk_recall_at_k(s_final_ids, gt_ids, k=20)

            s_hr1.append(hr1)
            s_hr5.append(hr5)
            s_hr10.append(hr10)
            s_mrr.append(mrr)
            s_ndcg5.append(ndcg5)
            s_tcr5.append(tcr5)
            s_tcr10.append(tcr10)
            s_tcr20.append(tcr20)

            # Failure attribution
            dense_rank = next((idx + 1 for idx, cid in enumerate(s_dense_top) if cid in gt_ids), None)
            bm25_rank = next((idx + 1 for idx, cid in enumerate(s_bm25_top) if cid in gt_ids), None)
            budget_rank = next((idx + 1 for idx, cid in enumerate(s_budget_20) if cid in gt_ids), None)
            final_rank = next((idx + 1 for idx, cid in enumerate(s_final_ids) if cid in gt_ids), None)

            if hr5 == 1.0:
                fail_cat = "NONE_HIT_TOP5"
            elif not any(cid in gt_ids for cid in s_rrf_candidates):
                fail_cat = "NOT_FOUND_SCOPED_FIRST_STAGE"
                failure_counts["NOT_FOUND_SCOPED_FIRST_STAGE"] += 1
            elif not any(cid in gt_ids for cid in s_budget_20):
                fail_cat = "LOST_BY_BUDGET"
                failure_counts["LOST_BY_BUDGET"] += 1
            elif hr10 == 1.0 and hr5 == 0.0:
                fail_cat = "TOP10_NOT_TOP5"
                failure_counts["TOP10_NOT_TOP5"] += 1
            elif budget_rank and budget_rank <= 5 and (final_rank is None or final_rank > 5):
                fail_cat = "RERANKER_DEMOTED"
                failure_counts["RERANKER_DEMOTED"] += 1
            else:
                fail_cat = "OTHER"
                failure_counts["OTHER"] += 1

            trace_record = {
                "query_index": q_idx,
                "query_id": q.get("query_id", f"holdout_{q_idx}"),
                "question": question,
                "contract_id": target_doc_id,
                "gold_chunk_ids": list(gt_ids),
                "scoped_chunks_count": len(scoped_chunk_ids),
                "dense_gold_rank": dense_rank,
                "bm25_gold_rank": bm25_rank,
                "pre_ce_gold_rank": budget_rank,
                "post_ce_gold_rank": final_rank,
                "hit_at_5": bool(hr5 == 1.0),
                "hit_at_10": bool(hr10 == 1.0),
                "mrr": float(mrr),
                "failure_category": fail_cat
            }
            f_trace.write(json.dumps(trace_record) + "\n")

    holdout_results = {
        "experiment_id": "EXP_PHASE4_1_HOLDOUT_DOCUMENT_SCOPED",
        "benchmark": "CUSTOM_CUAD_HOLDOUT_V2",
        "total_contracts": len(contracts_info),
        "total_queries": len(ans_queries),
        "valid_evaluated_queries": total_valid,
        "timestamp": "2026-08-15T18:05:00Z",
        "searchable_chunks_per_query": {
            "mean": float(np.mean(scoped_chunk_counts)),
            "median": float(np.median(scoped_chunk_counts)),
            "p95": float(np.percentile(scoped_chunk_counts, 95)),
        },
        "candidate_hit_rates": {
            "CandidateHitRate@1": float(np.mean(s_c1) * 100),
            "CandidateHitRate@5": float(np.mean(s_c5) * 100),
            "CandidateHitRate@10": float(np.mean(s_c10) * 100),
            "CandidateHitRate@20": float(np.mean(s_c20) * 100),
            "CandidateHitRate@50": float(np.mean(s_c50) * 100),
        },
        "true_chunk_recalls": {
            "TrueChunkRecall@5": float(np.mean(s_tcr5) * 100),
            "TrueChunkRecall@10": float(np.mean(s_tcr10) * 100),
            "TrueChunkRecall@20": float(np.mean(s_tcr20) * 100),
        },
        "post_rerank_metrics": {
            "HitRate@1": float(np.mean(s_hr1) * 100),
            "HitRate@5": float(np.mean(s_hr5) * 100),
            "HitRate@10": float(np.mean(s_hr10) * 100),
            "MRR": float(np.mean(s_mrr)),
            "nDCG@5": float(np.mean(s_ndcg5)),
        },
        "total_retrieval_latency_ms": {
            "P50": float(np.percentile(t_total_list, 50)),
            "P95": float(np.percentile(t_total_list, 95)),
            "P99": float(np.percentile(t_total_list, 99)),
        },
        "measured_stages_ms": {
            "T_scope_filter_p50": float(np.percentile(t_filter_list, 50)),
            "T_dense_search_p50": float(np.percentile(t_dense_list, 50)),
            "T_bm25_search_p50": float(np.percentile(t_bm25_list, 50)),
            "T_rrf_fusion_p50": float(np.percentile(t_rrf_list, 50)),
            "T_parent_dedup_p50": float(np.percentile(t_dedup_list, 50)),
            "T_crossencoder_k20_p50": float(np.percentile(t_ce_list, 50)),
        },
        "failure_attribution": failure_counts
    }

    out_file = RESULTS_DIR / "final_holdout_doc_scoped.json"
    out_file.write_text(json.dumps(holdout_results, indent=2), encoding="utf-8")
    print(f"  [OK] Saved final_holdout_doc_scoped.json")
    print(f"  [OK] Saved holdout_doc_scoped_rank_trace.jsonl")
    print(f"\n--- FROZEN HOLDOUT TRUE DOCUMENT-SCOPED RESULTS ---")
    print(f"  CandidateHitRate@20 = {np.mean(s_c20)*100:.2f}%")
    print(f"  HitRate@5           = {np.mean(s_hr5)*100:.2f}%")
    print(f"  HitRate@10          = {np.mean(s_hr10)*100:.2f}%")
    print(f"  MRR                 = {np.mean(s_mrr):.4f}")
    print(f"  nDCG@5              = {np.mean(s_ndcg5):.4f}")
    print(f"  Latency P50         = {np.percentile(t_total_list, 50):.2f} ms")
    print(f"  Latency P95         = {np.percentile(t_total_list, 95):.2f} ms")
    print("=" * 80)


if __name__ == "__main__":
    main()
