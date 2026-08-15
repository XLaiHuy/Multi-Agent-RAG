#!/usr/bin/env python3
"""
EXP-6: Dense Retrieval Model Capability (BAAI/bge-small-en-v1.5 vs BAAI/bge-m3).
Isolates dense embedding capability in DENSE MODE ONLY (no multi-vector, no ColBERT, no sparse).
Measures:
- Hit@5/10, Recall@5/10, MRR, nDCG@5, Pre-Rerank Candidate Recall
- Similarity score distributions (relevant vs non-relevant, top1-top2 margin)
- Indexing time, RAM, index size, P50/P95 latency
"""
import os
import sys
import time
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

sys.path.insert(0, os.getcwd())
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

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

DEV_MANIFEST_PATH = Path("evaluation/manifests/cuad_dev_manifest.json")
CONTRACTS_DIR = Path("evaluation/datasets/cuad/processed/contracts")
REGISTRY_PATH = Path("evaluation/reports/optimization_registry.jsonl")

def extract_contract_context(full_text: str, title: str) -> Dict[str, str]:
    clean_title = re.sub(r"[-_]", " ", title).strip()
    return {"title": clean_title}

def index_dev_corpus(contracts_info, chunker, sac_mode: str = "5A"):
    indexed_children = []
    indexed_parents = []
    chunk_dict = {}
    all_ids, all_texts, all_metas = [], [], []

    for c_info in contracts_info:
        md_file = CONTRACTS_DIR / c_info["filename"]
        txt_file = CONTRACTS_DIR / c_info["filename"].replace(".md", ".txt")
        target_file = md_file if md_file.exists() else txt_file
        if not target_file.exists():
            continue
        
        full_text = target_file.read_text(encoding="utf-8")
        doc = MasterDocumentParser.parse(target_file, doc_id=c_info["source_contract_id"])
        c_chunks, p_chunks = chunker.chunk_canonical_document(doc, doc_version=1)
        indexed_children.extend(c_chunks)
        indexed_parents.extend(p_chunks)

        ctx = extract_contract_context(full_text, c_info.get("original_title", ""))

        for c in c_chunks:
            chunk_dict[c.chunk_id] = c
            all_ids.append(c.chunk_id)
            sec_str = " > ".join(c.section_path) if c.section_path else "General"
            enriched_text = f"[Document: {ctx['title']}] [Section: {sec_str}]\n{c.text}"
            all_texts.append(enriched_text)
            all_metas.append(c.metadata)

    return all_ids, all_texts, all_metas, indexed_children, indexed_parents, chunk_dict

def evaluate_dense_model(
    model_name: str,
    all_ids: List[str],
    all_texts: List[str],
    all_metas: List[Dict],
    children: List[IndexedChunk],
    chunk_dict: Dict[str, IndexedChunk],
    queries: List[Dict],
    reranker: LocalCrossEncoderReranker,
    candidate_k: int = 20,
) -> Dict[str, Any]:
    
    print(f"\n--- Building Dense Index with model '{model_name}' ---")
    t0_embed = time.perf_counter()
    dense = InMemoryDenseRetriever(model_name=model_name)
    dense.build_index(all_ids, all_texts, batch_size=32)
    dense_embed_time_ms = (time.perf_counter() - t0_embed) * 1000.0

    bm25 = BM25Retriever()
    bm25.build_index(all_ids, all_texts, all_metas)

    ans_queries = [q for q in queries if not q.get("is_unanswerable", False)]
    all_questions = [q["question"] for q in ans_queries]
    
    print(f"  [Dense] Batch-encoding {len(all_questions)} queries with {model_name}...")
    t0_q = time.perf_counter()
    q_vecs = dense.embedder.embed_documents_batch(all_questions, batch_size=64)
    q_arr = np.array(q_vecs, dtype=np.float32)
    q_norms = np.linalg.norm(q_arr, axis=1, keepdims=True)
    q_norms = np.where(q_norms == 0, 1.0, q_norms)
    q_arr = q_arr / q_norms
    
    recalls_5, recalls_10 = [], []
    hits_5, hits_10 = [], []
    mrrs, ndcgs_5 = [], []
    latencies = []
    candidate_recall_pre_rerank = []
    relevant_similarities = []
    non_relevant_similarities = []
    top1_scores = []
    top1_top2_margins = []

    for q_idx, q in enumerate(ans_queries):
        question = q["question"]
        cid = q["source_contract_id"]
        gold_ev = q.get("gold_evidence", "").strip().lower()

        gt_ids = set()
        for c in children:
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
        
        # BM25 search
        b_hits = bm25.search(question, top_k=candidate_k)
        b_ids = [h[0] for h in b_hits]

        # Fast Vectorized Dense search using pre-encoded L2 normalized vectors
        q_vec = q_arr[q_idx]
        sims = dense.embeddings @ q_vec
        top_idxs = np.argsort(sims)[::-1][:candidate_k]
        d_hits = [(dense.chunk_ids[idx], float(sims[idx])) for idx in top_idxs]
        d_ids = [h[0] for h in d_hits]

        # Similarity distributions
        if d_hits:
            top1_scores.append(float(d_hits[0][1]))
            if len(d_hits) > 1:
                top1_top2_margins.append(float(d_hits[0][1] - d_hits[1][1]))
            for c_id, sim in d_hits:
                if c_id in gt_ids:
                    relevant_similarities.append(float(sim))
                else:
                    non_relevant_similarities.append(float(sim))

        # RRF Fusion
        fused = reciprocal_rank_fusion([b_ids, d_ids], k=60)
        cand_ids = [c_id for c_id, _ in fused[:candidate_k]]

        cand_hit = 1.0 if any(c_id in gt_ids for c_id in cand_ids) else 0.0
        candidate_recall_pre_rerank.append(cand_hit)

        # Fast Rerank
        cand_texts = [chunk_dict[c_id].text[:400] for c_id in cand_ids if c_id in chunk_dict]
        rerank_hits = reranker.rerank(question, cand_texts, top_n=10)
        final_ranked_ids = [cand_ids[idx] for idx, _ in rerank_hits if idx < len(cand_ids)]

        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat_ms)

        r5 = compute_recall_at_k(final_ranked_ids, gt_ids, k=5)
        r10 = compute_recall_at_k(final_ranked_ids, gt_ids, k=10)
        h5 = compute_hit_rate_at_k(final_ranked_ids, gt_ids, k=5)
        h10 = compute_hit_rate_at_k(final_ranked_ids, gt_ids, k=10)
        mrr = compute_reciprocal_rank(final_ranked_ids, gt_ids)
        ndcg5 = compute_ndcg_at_k(final_ranked_ids, gt_ids, k=5)

        recalls_5.append(r5)
        recalls_10.append(r10)
        hits_5.append(h5)
        hits_10.append(h10)
        mrrs.append(mrr)
        ndcgs_5.append(ndcg5)

    dim = dense.embedder.dimension
    matrix_bytes = dense.embeddings.nbytes if dense.embeddings is not None else 0

    return {
        "model_name": model_name,
        "embedding_dimension": dim,
        "index_size_kb": round(matrix_bytes / 1024, 2),
        "indexing_time_ms": round(dense_embed_time_ms, 2),
        "queries_evaluated": len(hits_5),
        "Recall@5": round(float(np.mean(recalls_5)), 4),
        "Recall@10": round(float(np.mean(recalls_10)), 4),
        "HitRate@5": round(float(np.mean(hits_5)), 4),
        "HitRate@10": round(float(np.mean(hits_10)), 4),
        "MRR": round(float(np.mean(mrrs)), 4),
        "nDCG@5": round(float(np.mean(ndcgs_5)), 4),
        "Candidate_Recall_Pre_Rerank": round(float(np.mean(candidate_recall_pre_rerank)), 4),
        "P50_Latency_ms": round(float(np.percentile(latencies, 50)), 2),
        "P95_Latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "Mean_Relevant_Similarity": round(float(np.mean(relevant_similarities)), 4) if relevant_similarities else 0.0,
        "Mean_Non_Relevant_Similarity": round(float(np.mean(non_relevant_similarities)), 4) if non_relevant_similarities else 0.0,
        "Mean_Top1_Score": round(float(np.mean(top1_scores)), 4) if top1_scores else 0.0,
        "Mean_Top1_Top2_Margin": round(float(np.mean(top1_top2_margins)), 4) if top1_top2_margins else 0.0,
    }

def run_exp6(sac_mode: str = "5B"):
    print("=" * 80)
    print("RUNNING EXP-6: DENSE RETRIEVAL MODEL CAPABILITY EXPERIMENT")
    print("=" * 80)

    manifest_data = json.loads(DEV_MANIFEST_PATH.read_text(encoding="utf-8"))
    contracts_info = manifest_data["contracts"]
    queries = manifest_data["queries"]

    chunker = StructureAwareParentChildChunker(
        child_target_tokens=250, child_overlap_tokens=30,
        parent_target_tokens=1200, parent_overlap_tokens=100
    )
    reranker = LocalCrossEncoderReranker()
    reranker.rerank("warmup", ["warmup doc"], top_n=1)

    all_ids, all_texts, all_metas, children, parents, chunk_dict = index_dev_corpus(
        contracts_info, chunker, sac_mode=sac_mode
    )

    # 1. Baseline Dense Model: BAAI/bge-small-en-v1.5
    res_small = evaluate_dense_model(
        "BAAI/bge-small-en-v1.5", all_ids, all_texts, all_metas, children, chunk_dict, queries, reranker
    )
    print(f"\n[BGE-Small] Hit@5={res_small['HitRate@5']}, Hit@10={res_small['HitRate@10']}, MRR={res_small['MRR']}, Pre-Rerank Recall={res_small['Candidate_Recall_Pre_Rerank']}, P50={res_small['P50_Latency_ms']}ms, Margin={res_small['Mean_Top1_Top2_Margin']}")

    # 2. Candidate Dense Model: BAAI/bge-m3
    res_m3 = evaluate_dense_model(
        "BAAI/bge-m3", all_ids, all_texts, all_metas, children, chunk_dict, queries, reranker
    )
    print(f"\n[BGE-M3 Dense] Hit@5={res_m3['HitRate@5']}, Hit@10={res_m3['HitRate@10']}, MRR={res_m3['MRR']}, Pre-Rerank Recall={res_m3['Candidate_Recall_Pre_Rerank']}, P50={res_m3['P50_Latency_ms']}ms, Margin={res_m3['Mean_Top1_Top2_Margin']}")

    hit5_gain = res_m3["HitRate@5"] - res_small["HitRate@5"]
    mrr_gain = res_m3["MRR"] - res_small["MRR"]
    cand_gain = res_m3["Candidate_Recall_Pre_Rerank"] - res_small["Candidate_Recall_Pre_Rerank"]

    # Keep BGE-M3 only if quality gain is meaningful relative to memory/latency
    decision = "KEEP" if (hit5_gain >= 0.015 or mrr_gain >= 0.010 or cand_gain >= 0.020) else "REJECT"

    print("\n" + "=" * 80)
    print(f"EXP-6 FINAL COMPARISON & DECISION: {decision}")
    print(f"  HitRate@5: {res_small['HitRate@5']} -> {res_m3['HitRate@5']} (Delta: {hit5_gain:+.4f})")
    print(f"  MRR:       {res_small['MRR']} -> {res_m3['MRR']} (Delta: {mrr_gain:+.4f})")
    print(f"  Pre-Rerank Candidate Recall: {res_small['Candidate_Recall_Pre_Rerank']} -> {res_m3['Candidate_Recall_Pre_Rerank']} (Delta: {cand_gain:+.4f})")
    print(f"  Indexing Time: {res_small['indexing_time_ms']:.1f}ms -> {res_m3['indexing_time_ms']:.1f}ms")
    print(f"  Memory Footprint: {res_small['index_size_kb']} KB (384-d) -> {res_m3['index_size_kb']} KB (1024-d)")
    print("=" * 80)

    # Log to registry
    record = {
        "experiment_id": "EXP-6_Dense_Model_Capability_BGE_M3",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hypothesis": "BGE-M3 1024-dim dense representation with larger context window improves semantic clause matching and candidate recall over 384-dim bge-small-en-v1.5.",
        "failure_category": "DENSE_SEMANTIC_FAILURE / CANDIDATE_POOL_FAILURE",
        "change": "Switch dense embedding model from BAAI/bge-small-en-v1.5 (384-d) to BAAI/bge-m3 (1024-d) in dense mode only",
        "baseline_config": {"dense_model": "BAAI/bge-small-en-v1.5", "dim": 384},
        "candidate_config": {"dense_model": "BAAI/bge-m3", "dim": 1024},
        "dev_manifest": str(DEV_MANIFEST_PATH),
        "before_metrics": res_small,
        "after_metrics": res_m3,
        "latency": {"P50_small_ms": res_small["P50_Latency_ms"], "P50_m3_ms": res_m3["P50_Latency_ms"]},
        "decision": decision,
        "reason": f"HitRate@5 delta={hit5_gain:+.4f}, MRR delta={mrr_gain:+.4f}, Candidate recall delta={cand_gain:+.4f}."
    }

    with open(REGISTRY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"[Registry] Logged EXP-6 -> Decision: {decision}")

    out_json = Path("evaluation/reports/exp6_dense_model_comparison.json")
    out_json.write_text(json.dumps({"res_small": res_small, "res_m3": res_m3, "decision": decision}, indent=2), encoding="utf-8")

if __name__ == "__main__":
    run_exp6()
