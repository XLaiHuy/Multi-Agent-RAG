#!/usr/bin/env python3
"""
EXP-11: Candidate Pool Recall & HitRate Diagnostic across candidate budgets k in [5, 10, 20, 30, 50, 100].
Measures CandidateHitRate@k and TrueChunkRecall@k on CUAD DEV split (20 contracts, 238 valid answerable queries).
Exports raw JSON to evaluation/results/phase3_5_1/candidate_pool.json.
"""
import os
import sys
import time
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Set

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "4"

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
torch.set_num_threads(4)
import numpy as np

from evaluation.config_loader import get_retrieval_config
from backend.app.retrieval.bm25 import BM25Retriever
from backend.app.retrieval.fusion import reciprocal_rank_fusion
from backend.app.ingestion.parsers import MasterDocumentParser
from backend.app.ingestion.chunker import StructureAwareParentChildChunker, IndexedChunk
from evaluation.dense_retriever_local import InMemoryDenseRetriever
from evaluation.metrics.retrieval_metrics import (
    compute_candidate_hit_rate_at_k, compute_true_chunk_recall_at_k, compute_reciprocal_rank
)

cfg = get_retrieval_config()
DEV_MANIFEST_PATH = REPO_ROOT / "evaluation" / "manifests" / "cuad_dev_manifest.json"
CONTRACTS_DIR = REPO_ROOT / "evaluation" / "datasets" / "cuad" / "processed" / "contracts"
OUTPUT_REPORT_PATH = REPO_ROOT / "evaluation" / "reports" / "CANDIDATE_POOL_DIAGNOSTIC.md"
OUTPUT_JSON_DIR = REPO_ROOT / "evaluation" / "results" / "phase3_5_1"

def run_diagnostic():
    start_wall_time = time.perf_counter()
    print("=" * 80)
    print("EXP-11: CANDIDATE POOL COVERAGE & RECALL DIAGNOSTIC (DEV SPLIT)")
    print(f"Model: {cfg.dense_model} ({cfg.dense_dimension}-d) | Chunker: {cfg.child_target_tokens}/{cfg.child_overlap_tokens}")
    print("=" * 80)

    manifest_raw = DEV_MANIFEST_PATH.read_bytes()
    manifest_hash = hashlib.sha256(manifest_raw).hexdigest()
    manifest_data = json.loads(manifest_raw.decode("utf-8"))
    contracts_info = manifest_data["contracts"]
    queries = manifest_data["queries"]
    ans_queries = [q for q in queries if not q.get("is_unanswerable", False)]

    chunker = StructureAwareParentChildChunker(
        child_target_tokens=cfg.child_target_tokens, child_overlap_tokens=cfg.child_overlap_tokens,
        parent_target_tokens=cfg.parent_target_tokens, parent_overlap_tokens=cfg.parent_overlap_tokens
    )

    all_ids, all_texts, all_metas = [], [], []
    indexed_children = []
    chunk_dict = {}

    for c_info in contracts_info:
        md_file = CONTRACTS_DIR / c_info["filename"]
        txt_file = CONTRACTS_DIR / c_info["filename"].replace(".md", ".txt")
        target_file = md_file if md_file.exists() else txt_file
        doc = MasterDocumentParser.parse(target_file, doc_id=c_info["source_contract_id"])
        c_chunks, p_chunks = chunker.chunk_canonical_document(doc, doc_version=1)
        indexed_children.extend(c_chunks)

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

    dense = InMemoryDenseRetriever(model_name=cfg.dense_model)
    dense.build_index(all_ids, all_texts)

    all_questions = [q["question"] for q in ans_queries]
    print(f"  [Dense] Batch-encoding {len(all_questions)} queries with {cfg.dense_model}...")
    q_vecs = dense.embedder.embed_documents_batch(all_questions, batch_size=64)
    q_arr = np.array(q_vecs, dtype=np.float32)
    q_norms = np.linalg.norm(q_arr, axis=1, keepdims=True)
    q_norms = np.where(q_norms == 0, 1.0, q_norms)
    q_arr = q_arr / q_norms

    k_list = [5, 10, 20, 30, 50, 100]
    results = {k: {"hit_rates": [], "recalls": [], "mrrs": []} for k in k_list}
    valid_queries = 0

    for q_idx, q in enumerate(ans_queries):
        question = q["question"]
        cid = q["source_contract_id"]
        gold_ev = q.get("gold_evidence", "").strip().lower()

        gt_ids = set()
        for c in indexed_children:
            if c.doc_id != cid:
                continue
            if gold_ev in c.text.lower() or (c.metadata and gold_ev in c.metadata.get("parent_text", "").lower()):
                gt_ids.add(c.chunk_id)

        if not gt_ids:
            continue
        valid_queries += 1

        b_hits = bm25.search(question, top_k=100)
        b_ids = [h[0] for h in b_hits]

        q_vec = q_arr[q_idx]
        sims = dense.embeddings @ q_vec
        top_idxs = np.argsort(sims)[::-1][:100]
        d_ids = [dense.chunk_ids[idx] for idx in top_idxs]

        fused = reciprocal_rank_fusion([b_ids, d_ids], k=cfg.rrf_k)
        fused_ids = [c_id for c_id, _ in fused[:100]]

        for k in k_list:
            cands = fused_ids[:k]
            results[k]["hit_rates"].append(compute_candidate_hit_rate_at_k(cands, gt_ids, k=k))
            results[k]["recalls"].append(compute_true_chunk_recall_at_k(cands, gt_ids, k=k))
            results[k]["mrrs"].append(compute_reciprocal_rank(cands, gt_ids))

    elapsed_s = time.perf_counter() - start_wall_time
    print(f"\n--- EXP-11 DIAGNOSTIC RESULTS (N = {valid_queries} DEV QUERIES) ---")
    print(f"{'Candidate Budget k':<20} | {'CandidateHitRate@k':<20} | {'TrueChunkRecall@k':<20} | {'MRR@k':<10}")
    print("-" * 75)

    table_rows = []
    json_metrics = {}
    for k in k_list:
        hr = float(np.mean(results[k]["hit_rates"]) * 100)
        rec = float(np.mean(results[k]["recalls"]) * 100)
        mrr = float(np.mean(results[k]["mrrs"]))
        print(f"Top-{k:<16} | {hr:6.2f}%{'':<13} | {rec:6.2f}%{'':<13} | {mrr:.4f}")
        table_rows.append(f"| **Top-{k}** | {hr:.2f}% | {rec:.2f}% | {mrr:.4f} |")
        json_metrics[f"CandidateHitRate@{k}"] = hr
        json_metrics[f"TrueChunkRecall@{k}"] = rec
        json_metrics[f"MRR@{k}"] = mrr

    # Save Machine-Readable JSON
    OUTPUT_JSON_DIR.mkdir(parents=True, exist_ok=True)
    raw_record = {
        "experiment_id": "EXP-11",
        "benchmark_name": "CUAD_DEV_SPLIT",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": round(elapsed_s, 2),
        "manifest_hash": manifest_hash,
        "dense_model": cfg.dense_model,
        "dense_dimension": cfg.dense_dimension,
        "sparse_retriever": cfg.sparse_retriever,
        "fusion": f"RRF_k{cfg.rrf_k}",
        "total_queries": len(queries),
        "valid_answerable_queries": valid_queries,
        "metrics": json_metrics,
    }
    json_path = OUTPUT_JSON_DIR / "candidate_pool.json"
    json_path.write_text(json.dumps(raw_record, indent=2), encoding="utf-8")

    report_md = f"""# EXP-11: Candidate Pool Diagnostic Report

**Evaluation Split:** CUAD DEV Split (20 Contracts, {valid_queries} Evaluated Answerable Queries)  
**Dense Model:** `{cfg.dense_model}` ({cfg.dense_dimension}-d)  
**Sparse Model:** `{cfg.sparse_retriever}`  
**Fusion:** Equal RRF ($k={cfg.rrf_k}$)  
**Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%SZ')}  
**Runtime:** {elapsed_s:.2f}s  

---

## 1. Candidate Pool Diagnostic Matrix

| Candidate Pool Size $k$ | CandidateHitRate@k (Any-Gold) | TrueChunkRecall@k (All-Gold) | First-Stage MRR |
|:---|:---:|:---:|:---:|
{chr(10).join(table_rows)}

---

## 2. Scientific Takeaways

1. **Candidate Coverage Pattern:** Candidate coverage (`CandidateHitRate@k`) increases monotonically through Top-100 (reaching {json_metrics['CandidateHitRate@100']:.2f}% at $k=100$).
2. **HitRate vs True Chunk Recall Distinction:**
   - `CandidateHitRate@k`: whether **at least one** relevant chunk is present in the first-stage pool ({json_metrics['CandidateHitRate@100']:.2f}% at $k=100$).
   - `TrueChunkRecall@k`: fraction of **all** relevant chunk spans captured ({json_metrics['TrueChunkRecall@100']:.2f}% at $k=100$).
"""
    OUTPUT_REPORT_PATH.write_text(report_md.strip() + "\n", encoding="utf-8")
    print(f"\n[OK] Wrote EXP-11 report to {OUTPUT_REPORT_PATH} and raw JSON to {json_path}")

if __name__ == "__main__":
    run_diagnostic()
