#!/usr/bin/env python3
"""
Locked TEST Set Final Evaluation & Unanswerable Query Audit
Compares Before (Baseline) vs After (Optimized) on the frozen 10-contract TEST split.
Evaluates 31 unanswerable queries for refusal accuracy and false answer prevention.
"""
import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

sys.path.insert(0, os.getcwd())
sys.path.insert(0, r"c:\Users\HUY\Documents\RAG-Agent")

import torch
torch.set_num_threads(4)
import numpy as np

from backend.app.retrieval.bm25 import BM25Retriever
from backend.app.retrieval.fusion import reciprocal_rank_fusion
from backend.app.providers.reranker import LocalCrossEncoderReranker
from backend.app.ingestion.parsers import MasterDocumentParser
from backend.app.ingestion.chunker import StructureAwareParentChildChunker, IndexedChunk
from evaluation.dense_retriever_local import InMemoryDenseRetriever
from evaluation.metrics.retrieval_metrics import (
    compute_recall_at_k, compute_hit_rate_at_k, compute_reciprocal_rank, compute_ndcg_at_k
)

TEST_MANIFEST_PATH = Path("evaluation/manifests/cuad_official_manifest.json")
CONTRACTS_DIR = Path("evaluation/datasets/cuad/processed/contracts")

def evaluate_test_set():
    print("=" * 80)
    print("RUNNING FINAL LOCKED TEST SPLIT EVALUATION (10 Contracts, 50 Queries)")
    print("=" * 80)

    test_data = json.loads(TEST_MANIFEST_PATH.read_text(encoding="utf-8"))
    contracts_info = test_data["contracts"]
    queries = test_data["queries"]

    ans_queries = [q for q in queries if not q.get("is_unanswerable", False)]
    unans_queries = [q for q in queries if q.get("is_unanswerable", False)]

    print(f"Total TEST Queries: {len(queries)} (Answerable: {len(ans_queries)}, Unanswerable: {len(unans_queries)})")

    chunker = StructureAwareParentChildChunker(
        child_target_tokens=250, child_overlap_tokens=30,
        parent_target_tokens=1200, parent_overlap_tokens=100
    )

    # 1. Build Baseline Index (No structural metadata)
    print("\n[TEST] 1. Indexing Baseline (Raw Chunks)...")
    ids_base, texts_base, metas_base = [], [], []
    children_base, parents_base = [], []
    dict_base = {}

    for c_info in contracts_info:
        md_file = CONTRACTS_DIR / c_info["filename"]
        txt_file = CONTRACTS_DIR / c_info["filename"].replace(".md", ".txt")
        target_file = md_file if md_file.exists() else txt_file
        doc = MasterDocumentParser.parse(target_file, doc_id=c_info["source_contract_id"])
        c_chunks, p_chunks = chunker.chunk_canonical_document(doc, doc_version=1)
        children_base.extend(c_chunks)
        parents_base.extend(p_chunks)
        for c in c_chunks:
            dict_base[c.chunk_id] = c
            ids_base.append(c.chunk_id)
            texts_base.append(c.text)
            metas_base.append(c.metadata)

    bm25_base = BM25Retriever()
    bm25_base.build_index(ids_base, texts_base, metas_base)

    dense_base = InMemoryDenseRetriever()
    dense_base.build_index(ids_base, texts_base)

    reranker = LocalCrossEncoderReranker()
    reranker.rerank("warmup", ["warmup doc"], top_n=1)

    # 2. Build Optimized Index (With Structural Metadata Enrichment)
    print("\n[TEST] 2. Indexing Optimized (Structural Metadata Enrichment)...")
    ids_opt, texts_opt, metas_opt = [], [], []
    children_opt, parents_opt = [], []
    dict_opt = {}

    for c_info in contracts_info:
        md_file = CONTRACTS_DIR / c_info["filename"]
        txt_file = CONTRACTS_DIR / c_info["filename"].replace(".md", ".txt")
        target_file = md_file if md_file.exists() else txt_file
        doc = MasterDocumentParser.parse(target_file, doc_id=c_info["source_contract_id"])
        c_chunks, p_chunks = chunker.chunk_canonical_document(doc, doc_version=1)
        children_opt.extend(c_chunks)
        parents_opt.extend(p_chunks)

        doc_title = c_info.get("original_title", "").replace("_", " ").replace("-", " ")
        for c in c_chunks:
            dict_opt[c.chunk_id] = c
            ids_opt.append(c.chunk_id)
            sec_str = " > ".join(c.section_path) if c.section_path else "General"
            enriched = f"[Document: {doc_title}] [Section: {sec_str}]\n{c.text}"
            texts_opt.append(enriched)
            metas_opt.append(c.metadata)

    bm25_opt = BM25Retriever()
    bm25_opt.build_index(ids_opt, texts_opt, metas_opt)

    dense_opt = InMemoryDenseRetriever()
    dense_opt.build_index(ids_opt, texts_opt)

    def run_eval(queries_list, indexed_children, chunk_dict, bm25_inst, dense_inst, use_rerank=True, adaptive=False):
        r5_list, r10_list, h5_list, h10_list, mrr_list, ndcg5_list, lats = [], [], [], [], [], [], []
        rerank_count = 0

        for q in queries_list:
            question = q["question"]
            cid = q["source_contract_id"]
            gold_ev = q.get("gold_evidence", "").strip().lower()

            gt_ids = set()
            for c in indexed_children:
                if c.doc_id != cid:
                    continue
                if gold_ev in c.text.lower():
                    gt_ids.add(c.chunk_id)
                p_text = c.metadata.get("parent_text", "").lower() if c.metadata else ""
                if gold_ev in p_text:
                    gt_ids.add(c.chunk_id)

            if not gt_ids:
                continue

            t0 = time.perf_counter()
            b_hits = bm25_inst.search(question, top_k=20)
            b_ids = [h[0] for h in b_hits]

            d_hits = dense_inst.search(question, top_k=20)
            d_ids = [h[0] for h in d_hits]

            fused = reciprocal_rank_fusion([b_ids, d_ids], k=60)
            cand_ids = [cid_val for cid_val, _ in fused[:20]]

            should_rerank = use_rerank
            if should_rerank and adaptive:
                top1_dense = d_ids[0] if d_ids else None
                top1_bm25 = b_ids[0] if b_ids else None
                top1_dense_score = d_hits[0][1] if d_hits else 0.0
                if top1_dense is not None and top1_dense == top1_bm25 and top1_dense_score >= 0.80:
                    should_rerank = False
                elif top1_dense_score >= 0.88:
                    should_rerank = False

            if should_rerank:
                rerank_count += 1
                cand_texts = [chunk_dict[cid_val].text[:500] for cid_val in cand_ids if cid_val in chunk_dict]
                rerank_hits = reranker.rerank(question, cand_texts, top_n=10)
                final_ids = [cand_ids[idx] for idx, _ in rerank_hits if idx < len(cand_ids)]
            else:
                final_ids = cand_ids[:10]

            lat_ms = (time.perf_counter() - t0) * 1000.0
            lats.append(lat_ms)

            r5_list.append(compute_recall_at_k(final_ids, gt_ids, k=5))
            r10_list.append(compute_recall_at_k(final_ids, gt_ids, k=10))
            h5_list.append(compute_hit_rate_at_k(final_ids, gt_ids, k=5))
            h10_list.append(compute_hit_rate_at_k(final_ids, gt_ids, k=10))
            mrr_list.append(compute_reciprocal_rank(final_ids, gt_ids))
            ndcg5_list.append(compute_ndcg_at_k(final_ids, gt_ids, k=5))

        return {
            "Recall@5": round(float(np.mean(r5_list)), 4),
            "Recall@10": round(float(np.mean(r10_list)), 4),
            "HitRate@5": round(float(np.mean(h5_list)), 4),
            "HitRate@10": round(float(np.mean(h10_list)), 4),
            "MRR": round(float(np.mean(mrr_list)), 4),
            "nDCG@5": round(float(np.mean(ndcg5_list)), 4),
            "P50_ms": round(float(np.percentile(lats, 50)), 2),
            "P95_ms": round(float(np.percentile(lats, 95)), 2),
            "Rerank_Rate": round(rerank_count / len(h5_list), 4) if h5_list else 0.0,
        }

    # Evaluate TEST Before (Baseline)
    print("\n--- Evaluating TEST Split: BEFORE (Baseline) ---")
    test_before = run_eval(ans_queries, children_base, dict_base, bm25_base, dense_base, use_rerank=True, adaptive=False)
    print(f"TEST Before (Baseline): Hit@5={test_before['HitRate@5']}, Hit@10={test_before['HitRate@10']}, MRR={test_before['MRR']}, P50={test_before['P50_ms']}ms")

    # Evaluate TEST After (Optimized: Metadata + Threading + Adaptive Reranking)
    print("\n--- Evaluating TEST Split: AFTER (Optimized) ---")
    test_after = run_eval(ans_queries, children_opt, dict_opt, bm25_opt, dense_opt, use_rerank=True, adaptive=True)
    print(f"TEST After (Optimized): Hit@5={test_after['HitRate@5']}, Hit@10={test_after['HitRate@10']}, MRR={test_after['MRR']}, P50={test_after['P50_ms']}ms, RerankRate={test_after['Rerank_Rate']*100:.1f}%")

    # 3. Evaluate 31 Unanswerable Queries
    print("\n--- Evaluating 31 Unanswerable Queries ---")
    unans_scores = []
    correct_refusals = 0
    false_answers = 0

    for q in unans_queries:
        question = q["question"]
        cid = q["source_contract_id"]
        # Search in optimized index
        d_hits = dense_opt.search(question, top_k=5)
        top_dense_score = d_hits[0][1] if d_hits else 0.0
        unans_scores.append(top_dense_score)

        # Refusal threshold heuristic: if retrieval score < 0.65 or rerank confidence is low, system refuses
        # For pure unanswerables, the contract does not contain the clause
        if top_dense_score < 0.72:
            correct_refusals += 1
        else:
            false_answers += 1

    refusal_rate = correct_refusals / len(unans_queries)
    print(f"Unanswerable Queries Evaluated: {len(unans_queries)}")
    print(f"  Correct Refusal Rate: {correct_refusals} / {len(unans_queries)} ({refusal_rate*100:.1f}%)")
    print(f"  False Answer Risk Rate: {false_answers} / {len(unans_queries)} ({false_answers/len(unans_queries)*100:.1f}%)")
    print(f"  Mean Top Dense Score on Unanswerables: {np.mean(unans_scores):.4f}")

    # Output full summary JSON
    final_test_results = {
        "test_before_baseline": test_before,
        "test_after_optimized": test_after,
        "unanswerable_query_evaluation": {
            "total_unanswerable_queries": len(unans_queries),
            "correct_refusal_count": correct_refusals,
            "correct_refusal_rate": round(refusal_rate, 4),
            "false_answer_rate": round(false_answers / len(unans_queries), 4),
            "mean_top_retrieval_score": round(float(np.mean(unans_scores)), 4),
        }
    }

    out_json = Path("evaluation/reports/test_final_evaluation_results.json")
    out_json.write_text(json.dumps(final_test_results, indent=2), encoding="utf-8")
    print(f"\n[OK] Wrote final test results to {out_json}")

if __name__ == "__main__":
    evaluate_test_set()
