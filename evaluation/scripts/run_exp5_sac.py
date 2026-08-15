#!/usr/bin/env python3
"""
EXP-5: Summary / Context-Augmented Chunking (SAC) Evaluation on DEV Split.
Compares:
- EXP-5A: Structural Metadata Baseline ([Document: Title] [Section: Path])
- EXP-5B: Context-Augmented Chunking ([Document: Title] [Type: Agreement Type] [Context: Preamble] [Section: Path])

Evaluates candidate recall before reranker, Hit@5/10, Recall@5/10, MRR, nDCG@5,
wrong-contract retrieval rate, indexing time, index size, and latency.
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

DEV_MANIFEST_PATH = Path("evaluation/manifests/cuad_dev_manifest.json")
CONTRACTS_DIR = Path("evaluation/datasets/cuad/processed/contracts")
REGISTRY_PATH = Path("evaluation/reports/optimization_registry.jsonl")

def extract_contract_context(full_text: str, title: str) -> Dict[str, str]:
    """
    Extracts deterministic, document-derivable context during ingestion without query-time LLM.
    """
    clean_title = re.sub(r"[-_]", " ", title).strip()
    
    # 1. Infer agreement type
    ag_type = "Agreement"
    type_match = re.search(r"(?i)\b([A-Z\s]+AGREEMENT|[A-Z\s]+CONTRACT|[A-Z\s]+PLAN|[A-Z\s]+LEASE)\b", title)
    if type_match:
        ag_type = type_match.group(1).strip().title()
    elif "AGREEMENT" in full_text[:500].upper():
        m = re.search(r"(?i)([A-Z\s]{4,30}\s+(?:AGREEMENT|CONTRACT))", full_text[:500])
        if m:
            ag_type = m.group(1).strip().title()

    # 2. Extract preamble snippet (first 60-80 words of preamble establishing parties/date)
    lines = [l.strip() for l in full_text.split("\n") if l.strip() and not l.startswith("#")]
    preamble_snippet = " ".join(lines[:3]) if lines else ""
    preamble_words = preamble_snippet.split()
    compact_preamble = " ".join(preamble_words[:50])

    return {
        "title": clean_title,
        "agreement_type": ag_type,
        "preamble_context": compact_preamble
    }

def index_dev_corpus(contracts_info, chunker, sac_mode: str = "5A"):
    indexed_children = []
    indexed_parents = []
    chunk_dict = {}
    
    all_ids, all_texts, all_metas = [], [], []

    t0 = time.perf_counter()
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

            if sac_mode == "5A":
                # EXP-5A: Structural Metadata
                enriched_text = f"[Document: {ctx['title']}] [Section: {sec_str}]\n{c.text}"
            elif sac_mode == "5B":
                # EXP-5B: Context/Summary-Augmented Chunk (SAC)
                enriched_text = (
                    f"[Document: {ctx['title']}] | [Type: {ctx['agreement_type']}] | "
                    f"[Preamble: {ctx['preamble_context'][:120]}] | [Section: {sec_str}]\n{c.text}"
                )
            else:
                enriched_text = c.text

            all_texts.append(enriched_text)
            all_metas.append(c.metadata)

    indexing_time = (time.perf_counter() - t0) * 1000.0
    return all_ids, all_texts, all_metas, indexed_children, indexed_parents, chunk_dict, indexing_time

def evaluate_sac_variant(
    sac_mode: str,
    contracts_info: List[Dict],
    queries: List[Dict],
    chunker: StructureAwareParentChildChunker,
    reranker: LocalCrossEncoderReranker,
    candidate_k: int = 20,
) -> Dict[str, Any]:
    
    print(f"\n--- Indexing & Evaluating Variant EXP-{sac_mode} ---")
    all_ids, all_texts, all_metas, children, parents, chunk_dict, index_time = index_dev_corpus(
        contracts_info, chunker, sac_mode=sac_mode
    )

    t_bm25 = time.perf_counter()
    bm25 = BM25Retriever()
    bm25.build_index(all_ids, all_texts, all_metas)
    bm25_index_time = (time.perf_counter() - t_bm25) * 1000.0

    t_dense = time.perf_counter()
    dense = InMemoryDenseRetriever()
    dense.build_index(all_ids, all_texts)
    dense_index_time = (time.perf_counter() - t_dense) * 1000.0

    ans_queries = [q for q in queries if not q.get("is_unanswerable", False)]
    
    recalls_5, recalls_10 = [], []
    hits_5, hits_10 = [], []
    mrrs, ndcgs_5 = [], []
    latencies = []
    wrong_contract_candidates_count = 0
    total_candidates_evaluated = 0
    candidate_recall_pre_rerank = []

    for q in ans_queries:
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
        
        # 1. First-stage retrieval
        b_hits = bm25.search(question, top_k=candidate_k)
        b_ids = [h[0] for h in b_hits]

        d_hits = dense.search(question, top_k=candidate_k)
        d_ids = [h[0] for h in d_hits]

        fused = reciprocal_rank_fusion([b_ids, d_ids], k=60)
        cand_ids = [c_id for c_id, _ in fused[:candidate_k]]

        # Candidate recall before reranker
        cand_hit = 1.0 if any(c_id in gt_ids for c_id in cand_ids) else 0.0
        candidate_recall_pre_rerank.append(cand_hit)

        # Count wrong contract candidates
        for c_id in cand_ids:
            total_candidates_evaluated += 1
            if chunk_dict[c_id].doc_id != cid:
                wrong_contract_candidates_count += 1

        # 2. Second-stage reranking
        cand_texts = [chunk_dict[c_id].text[:500] for c_id in cand_ids if c_id in chunk_dict]
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

    eval_count = len(hits_5)
    wrong_contract_rate = (wrong_contract_candidates_count / total_candidates_evaluated) if total_candidates_evaluated else 0.0

    return {
        "variant": f"EXP-{sac_mode}",
        "queries_evaluated": eval_count,
        "Recall@5": round(float(np.mean(recalls_5)), 4),
        "Recall@10": round(float(np.mean(recalls_10)), 4),
        "HitRate@5": round(float(np.mean(hits_5)), 4),
        "HitRate@10": round(float(np.mean(hits_10)), 4),
        "MRR": round(float(np.mean(mrrs)), 4),
        "nDCG@5": round(float(np.mean(ndcgs_5)), 4),
        "Candidate_Recall_Pre_Rerank": round(float(np.mean(candidate_recall_pre_rerank)), 4),
        "Wrong_Contract_Candidate_Rate": round(wrong_contract_rate, 4),
        "P50_Latency_ms": round(float(np.percentile(latencies, 50)), 2),
        "P95_Latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "Total_Indexing_Time_ms": round(index_time + bm25_index_time + dense_index_time, 2),
        "Indexed_Chunks_Count": len(all_ids),
    }

def run_exp5():
    print("=" * 80)
    print("RUNNING EXP-5: SUMMARY / CONTEXT-AUGMENTED CHUNKING (SAC) EXPERIMENT")
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

    # 1. Run EXP-5A (Structural Metadata Baseline)
    res_5a = evaluate_sac_variant("5A", contracts_info, queries, chunker, reranker)
    print(f"\n[EXP-5A Structural Metadata] Hit@5={res_5a['HitRate@5']}, Hit@10={res_5a['HitRate@10']}, MRR={res_5a['MRR']}, Pre-Rerank Recall={res_5a['Candidate_Recall_Pre_Rerank']}, Wrong-Contract Rate={res_5a['Wrong_Contract_Candidate_Rate']*100:.1f}%")

    # 2. Run EXP-5B (Summary/Context-Augmented Chunking)
    res_5b = evaluate_sac_variant("5B", contracts_info, queries, chunker, reranker)
    print(f"\n[EXP-5B Context-Augmented (SAC)] Hit@5={res_5b['HitRate@5']}, Hit@10={res_5b['HitRate@10']}, MRR={res_5b['MRR']}, Pre-Rerank Recall={res_5b['Candidate_Recall_Pre_Rerank']}, Wrong-Contract Rate={res_5b['Wrong_Contract_Candidate_Rate']*100:.1f}%")

    # Compare and decide
    hit5_gain = res_5b["HitRate@5"] - res_5a["HitRate@5"]
    mrr_gain = res_5b["MRR"] - res_5a["MRR"]
    wrong_contract_drop = res_5a["Wrong_Contract_Candidate_Rate"] - res_5b["Wrong_Contract_Candidate_Rate"]

    decision = "KEEP" if (res_5b["HitRate@5"] >= res_5a["HitRate@5"] and res_5b["MRR"] >= res_5a["MRR"]) else "REJECT"

    print("\n" + "=" * 80)
    print(f"EXP-5 FINAL COMPARISON & DECISION: {decision}")
    print(f"  HitRate@5: {res_5a['HitRate@5']} -> {res_5b['HitRate@5']} (Delta: {hit5_gain:+.4f})")
    print(f"  MRR:       {res_5a['MRR']} -> {res_5b['MRR']} (Delta: {mrr_gain:+.4f})")
    print(f"  Pre-Rerank Candidate Recall: {res_5a['Candidate_Recall_Pre_Rerank']} -> {res_5b['Candidate_Recall_Pre_Rerank']}")
    print(f"  Wrong Contract Candidate Rate: {res_5a['Wrong_Contract_Candidate_Rate']*100:.1f}% -> {res_5b['Wrong_Contract_Candidate_Rate']*100:.1f}% (Reduction: {wrong_contract_drop*100:+.1f}%)")
    print("=" * 80)

    # Log to optimization registry
    record = {
        "experiment_id": "EXP-5_Context_Augmented_Chunking_SAC",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hypothesis": "Lightweight ingestion-time document context (agreement type + preamble snippet) reduces cross-contract ambiguity and improves candidate recall without query-time LLM.",
        "failure_category": "QUERY_AMBIGUITY / CROSS_CONTRACT_DISTRACTION",
        "change": "Enrich child chunk search text with [Document Title], [Agreement Type], [Preamble Context], and [Section Path]",
        "baseline_config": {"sac_mode": "5A", "dense_model": "BAAI/bge-small-en-v1.5", "candidate_k": 20},
        "candidate_config": {"sac_mode": "5B", "dense_model": "BAAI/bge-small-en-v1.5", "candidate_k": 20},
        "dev_manifest": str(DEV_MANIFEST_PATH),
        "before_metrics": res_5a,
        "after_metrics": res_5b,
        "latency": {"P50_5A_ms": res_5a["P50_Latency_ms"], "P50_5B_ms": res_5b["P50_Latency_ms"]},
        "decision": decision,
        "reason": f"HitRate@5 delta={hit5_gain:+.4f}, MRR delta={mrr_gain:+.4f}, Wrong-contract rate delta={-wrong_contract_drop*100:+.1f} percentage points."
    }

    with open(REGISTRY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"[Registry] Logged EXP-5 -> Decision: {decision}")

    # Output detailed report json
    out_json = Path("evaluation/reports/exp5_sac_comparison.json")
    out_json.write_text(json.dumps({"res_5a": res_5a, "res_5b": res_5b, "decision": decision}, indent=2), encoding="utf-8")

if __name__ == "__main__":
    run_exp5()
