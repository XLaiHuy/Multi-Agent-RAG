#!/usr/bin/env python3
"""
EXP-11: Candidate Pool Diagnostic across Top-5, 10, 20, 30, 50, 100.
Measures first-stage candidate recall, HitRate, MRR, rank distribution bins, and latency before reranking.
Generates evaluation/reports/CANDIDATE_POOL_DIAGNOSTIC.md.
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
from backend.app.ingestion.parsers import MasterDocumentParser
from backend.app.ingestion.chunker import StructureAwareParentChildChunker, IndexedChunk
from evaluation.dense_retriever_local import InMemoryDenseRetriever
from evaluation.metrics.retrieval_metrics import (
    compute_recall_at_k, compute_hit_rate_at_k, compute_reciprocal_rank, compute_ndcg_at_k
)

DEV_MANIFEST_PATH = Path("evaluation/manifests/cuad_dev_manifest.json")
CONTRACTS_DIR = Path("evaluation/datasets/cuad/processed/contracts")
REPORT_PATH = Path("evaluation/reports/CANDIDATE_POOL_DIAGNOSTIC.md")
REGISTRY_PATH = Path("evaluation/reports/optimization_registry.jsonl")

def run_candidate_pool_diagnostic(dense_model: str = "BAAI/bge-small-en-v1.5"):
    print("=" * 80)
    print(f"RUNNING EXP-11: CANDIDATE POOL DIAGNOSTIC (Model: {dense_model})")
    print("=" * 80)

    manifest_data = json.loads(DEV_MANIFEST_PATH.read_text(encoding="utf-8"))
    contracts_info = manifest_data["contracts"]
    queries = manifest_data["queries"]
    ans_queries = [q for q in queries if not q.get("is_unanswerable", False)]

    chunker = StructureAwareParentChildChunker(
        child_target_tokens=250, child_overlap_tokens=30,
        parent_target_tokens=1200, parent_overlap_tokens=100
    )

    # Ingest with structural metadata
    all_ids, all_texts, all_metas = [], [], []
    indexed_children, indexed_parents = [], []
    chunk_dict = {}

    for c_info in contracts_info:
        md_file = CONTRACTS_DIR / c_info["filename"]
        txt_file = CONTRACTS_DIR / c_info["filename"].replace(".md", ".txt")
        target_file = md_file if md_file.exists() else txt_file
        doc = MasterDocumentParser.parse(target_file, doc_id=c_info["source_contract_id"])
        c_chunks, p_chunks = chunker.chunk_canonical_document(doc, doc_version=1)
        indexed_children.extend(c_chunks)
        indexed_parents.extend(p_chunks)

        doc_title = c_info.get("original_title", "").replace("_", " ").replace("-", " ")
        for c in c_chunks:
            chunk_dict[c.chunk_id] = c
            all_ids.append(c.chunk_id)
            sec_str = " > ".join(c.section_path) if c.section_path else "General"
            enriched = f"[Document: {doc_title}] [Section: {sec_str}]\n{c.text}"
            all_texts.append(enriched)
            all_metas.append(c.metadata)

    bm25 = BM25Retriever()
    bm25.build_index(all_ids, all_texts, all_metas)

    dense = InMemoryDenseRetriever(model_name=dense_model)
    dense.build_index(all_ids, all_texts)

    # Batch encode queries
    all_questions = [q["question"] for q in ans_queries]
    print(f"  [Dense] Batch-encoding {len(all_questions)} queries with {dense_model}...")
    q_vecs = dense.embedder.embed_documents_batch(all_questions, batch_size=64)
    q_arr = np.array(q_vecs, dtype=np.float32)
    q_norms = np.linalg.norm(q_arr, axis=1, keepdims=True)
    q_norms = np.where(q_norms == 0, 1.0, q_norms)
    q_arr = q_arr / q_norms

    k_values = [5, 10, 20, 30, 50, 100]
    metrics_per_k = {k: {"cand_recall": [], "hit_rate": [], "mrr": [], "latencies": []} for k in k_values}

    # Track first appearance rank of gold evidence
    first_rank_distribution = {
        "1_5": 0,
        "6_10": 0,
        "11_20": 0,
        "21_30": 0,
        "31_50": 0,
        "51_100": 0,
        "never_or_gt100": 0,
    }

    total_valid_queries = 0

    for q_idx, q in enumerate(ans_queries):
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

        total_valid_queries += 1

        t0 = time.perf_counter()
        # Retrieve broad candidate pool up to Top-100
        b_hits = bm25.search(question, top_k=100)
        b_ids = [h[0] for h in b_hits]

        q_vec = q_arr[q_idx]
        sims = dense.embeddings @ q_vec
        top_idxs = np.argsort(sims)[::-1][:100]
        d_ids = [dense.chunk_ids[idx] for idx in top_idxs]

        fused_100 = reciprocal_rank_fusion([b_ids, d_ids], k=60)
        fused_ids_100 = [c_id for c_id, _ in fused_100[:100]]

        lat_ms = (time.perf_counter() - t0) * 1000.0

        # Find first rank of gold chunk in fused_ids_100 (1-based)
        first_gold_rank = None
        for r_idx, c_id in enumerate(fused_ids_100):
            if c_id in gt_ids:
                first_gold_rank = r_idx + 1
                break

        if first_gold_rank is not None:
            if 1 <= first_gold_rank <= 5:
                first_rank_distribution["1_5"] += 1
            elif 6 <= first_gold_rank <= 10:
                first_rank_distribution["6_10"] += 1
            elif 11 <= first_gold_rank <= 20:
                first_rank_distribution["11_20"] += 1
            elif 21 <= first_gold_rank <= 30:
                first_rank_distribution["21_30"] += 1
            elif 31 <= first_gold_rank <= 50:
                first_rank_distribution["31_50"] += 1
            elif 51 <= first_gold_rank <= 100:
                first_rank_distribution["51_100"] += 1
        else:
            first_rank_distribution["never_or_gt100"] += 1

        # Calculate metrics for each k
        for k in k_values:
            top_k_ids = fused_ids_100[:k]
            cand_recall = 1.0 if any(c_id in gt_ids for c_id in top_k_ids) else 0.0
            hit_k = compute_hit_rate_at_k(top_k_ids, gt_ids, k=k)
            mrr_k = compute_reciprocal_rank(top_k_ids, gt_ids)

            metrics_per_k[k]["cand_recall"].append(cand_recall)
            metrics_per_k[k]["hit_rate"].append(hit_k)
            metrics_per_k[k]["mrr"].append(mrr_k)
            metrics_per_k[k]["latencies"].append(lat_ms)

    # Summarize results
    table_rows = []
    print("\n--- Summary of Candidate Pool Diagnostic across k ---")
    print(f"{'Pool k':<10} | {'Candidate Recall':<18} | {'HitRate@k':<12} | {'MRR':<10} | {'P50 Latency (ms)':<18}")
    print("-" * 75)

    summary_dict = {}
    for k in k_values:
        mean_cr = np.mean(metrics_per_k[k]["cand_recall"]) * 100
        mean_hr = np.mean(metrics_per_k[k]["hit_rate"]) * 100
        mean_mrr = np.mean(metrics_per_k[k]["mrr"])
        p50_lat = np.percentile(metrics_per_k[k]["latencies"], 50)

        summary_dict[f"Top_{k}"] = {
            "candidate_recall": round(mean_cr / 100, 4),
            "hit_rate": round(mean_hr / 100, 4),
            "mrr": round(float(mean_mrr), 4),
            "p50_latency_ms": round(float(p50_lat), 2),
        }

        print(f"Top-{k:<6} | {mean_cr:6.2f}%{'':<11} | {mean_hr:6.2f}%{'':<5} | {mean_mrr:.4f}{'':<4} | {p50_lat:6.2f} ms")
        table_rows.append(f"| **Top-{k}** | **{mean_cr:.2f}%** | {mean_hr:.2f}% | {mean_mrr:.4f} | {p50_lat:.2f} ms |")

    table_md_str = "\n".join(table_rows)

    # Rank bin distribution table
    bin_rows = [
        f"| **Ranks 1 – 5** | {first_rank_distribution['1_5']} | {first_rank_distribution['1_5']/total_valid_queries*100:.2f}% |",
        f"| **Ranks 6 – 10** | {first_rank_distribution['6_10']} | {first_rank_distribution['6_10']/total_valid_queries*100:.2f}% |",
        f"| **Ranks 11 – 20** | {first_rank_distribution['11_20']} | {first_rank_distribution['11_20']/total_valid_queries*100:.2f}% |",
        f"| **Ranks 21 – 30** | {first_rank_distribution['21_30']} | {first_rank_distribution['21_30']/total_valid_queries*100:.2f}% |",
        f"| **Ranks 31 – 50** | {first_rank_distribution['31_50']} | {first_rank_distribution['31_50']/total_valid_queries*100:.2f}% |",
        f"| **Ranks 51 – 100** | {first_rank_distribution['51_100']} | {first_rank_distribution['51_100']/total_valid_queries*100:.2f}% |",
        f"| **> 100 / Never Found** | {first_rank_distribution['never_or_gt100']} | {first_rank_distribution['never_or_gt100']/total_valid_queries*100:.2f}% |",
    ]
    bin_md_str = "\n".join(bin_rows)

    # Interpretation:
    cr_20 = summary_dict["Top_20"]["candidate_recall"]
    cr_100 = summary_dict["Top_100"]["candidate_recall"]
    delta_cr = (cr_100 - cr_20) * 100

    if cr_100 >= 0.60 or delta_cr >= 15.0:
        case_diagnosis = "CASE A: Candidate recall rises strongly with broader first-stage pool (from Top-20 to Top-100)."
        action_recommendation = "Deploy broad first-stage retrieval (Top-50/100) paired with cheap pruning and CrossEncoder reranking."
    elif cr_100 < 0.50:
        case_diagnosis = "CASE B: Candidate recall remains low even at Top-100, indicating retriever semantic / terminology representation limits."
        action_recommendation = "Deploy document/section soft routing boost + legal-specific embedding + hybrid union."
    else:
        case_diagnosis = "CASE C: Candidate recall is moderate; ranking and reranking calibration is the key lever."
        action_recommendation = "Calibrate candidate pruning and two-stage reranker gating."

    report_md = f"""# Candidate Pool Diagnostic Report (EXP-11)

**Evaluation Split:** CUAD DEV Set (20 Contracts, 244 Answerable Queries)  
**Retriever:** Hybrid RRF (BM25Okapi + Dense `{dense_model}` with Structural Metadata)  
**Evaluation Scope:** First-stage retrieval candidate generation across $k \\in [5, 10, 20, 30, 50, 100]$ before reranking  
**Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%SZ')}

---

## 1. Candidate Recall & Metric Progression across Pool Size $k$

| Candidate Pool Size ($k$) | Candidate Recall@k | HitRate@k | MRR | P50 First-Stage Latency |
|:---|:---:|:---:|:---:|:---:|
{table_md_str}

---

## 2. Gold Evidence Rank Appearance Distribution

Where does the first relevant evidence chunk appear in the first-stage retrieved list?

| Rank Range Bin | Query Count | Percentage of Total Queries |
|:---|:---:|:---:|
{bin_md_str}

---

## 3. Empirical Diagnosis & Action Plan

- **Top-20 Candidate Recall:** **`{cr_20*100:.2f}%`**
- **Top-100 Candidate Recall:** **`{cr_100*100:.2f}%`** (Recall Gain: **`+{delta_cr:.2f}%`**)
- **Never Found in Top-100:** **`{first_rank_distribution['never_or_gt100']/total_valid_queries*100:.2f}%`**

### Case Interpretation:
**{case_diagnosis}**

### Recommended Engineering Roadmap:
1. **{action_recommendation}**
2. Combine **broad first-stage candidate generation (Top-50/100)** with **EXP-12 Soft Document/Section Routing Boost** to filter out distractor contracts before passing to CrossEncoder.
3. Test **Dense $\\cup$ Sparse Candidate Union** (EXP-16) to maximize unique candidate capture.
"""

    REPORT_PATH.write_text(report_md.strip() + "\n", encoding="utf-8")
    print(f"\n[OK] Wrote Candidate Pool Diagnostic Report to {REPORT_PATH}")

    # Log to optimization registry
    record = {
        "experiment_id": "EXP-11_Candidate_Pool_Diagnostic",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hypothesis": "Evaluating candidate recall across k in [5..100] isolates whether first-stage loss is due to candidate cutoff vs retriever capability.",
        "failure_category": "CANDIDATE_POOL_FAILURE / RETRIEVAL_CUTOFF",
        "change": "Sweep candidate pool size k in [5, 10, 20, 30, 50, 100] on DEV",
        "baseline_config": {"candidate_k": 20},
        "candidate_config": {"k_sweep": summary_dict},
        "dev_manifest": str(DEV_MANIFEST_PATH),
        "before_metrics": {"candidate_recall_top20": cr_20},
        "after_metrics": {"candidate_recall_top50": summary_dict["Top_50"]["candidate_recall"], "candidate_recall_top100": cr_100},
        "decision": "KEEP_DIAGNOSTIC",
        "reason": f"Top-20 Recall: {cr_20*100:.2f}% -> Top-50: {summary_dict['Top_50']['candidate_recall']*100:.2f}% -> Top-100: {cr_100*100:.2f}%."
    }

    with open(REGISTRY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"[Registry] Logged EXP-11 -> Diagnosis: {case_diagnosis[:40]}...")

if __name__ == "__main__":
    run_candidate_pool_diagnostic()
