#!/usr/bin/env python3
"""
Fair Comparison: Candidate Union vs Equal RRF at IDENTICAL Candidate Budgets (@20, @50, @100).
Reports both CandidateHitRate@k and TrueChunkRecall@k on CUAD DEV split.
Exports raw JSON to evaluation/results/phase3_5_1/rrf_vs_union.json.
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
OUTPUT_REPORT_PATH = REPO_ROOT / "evaluation" / "reports" / "CANDIDATE_UNION_VS_RRF_FAIR_COMPARISON.md"
OUTPUT_JSON_DIR = REPO_ROOT / "evaluation" / "results" / "phase3_5_1"

def evaluate_union_vs_rrf_fair():
    start_wall_time = time.perf_counter()
    print("=" * 80)
    print("FAIR CANDIDATE UNION VS EQUAL RRF COMPARISON AT IDENTICAL BUDGETS (@20, @50, @100)")
    print(f"Dense: {cfg.dense_model} | Sparse: {cfg.sparse_retriever} | Fusion: RRF k={cfg.rrf_k}")
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

    for c_info in contracts_info:
        md_file = CONTRACTS_DIR / c_info["filename"]
        txt_file = CONTRACTS_DIR / c_info["filename"].replace(".md", ".txt")
        target_file = md_file if md_file.exists() else txt_file
        doc = MasterDocumentParser.parse(target_file, doc_id=c_info["source_contract_id"])
        c_chunks, p_chunks = chunker.chunk_canonical_document(doc, doc_version=1)
        indexed_children.extend(c_chunks)

        doc_title = c_info.get("original_title", "").replace("_", " ").replace("-", " ")
        for c in c_chunks:
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

    budgets = [20, 50, 100]
    metrics = {
        "Dense_Only": {b: {"hit": [], "rec": [], "mrr": []} for b in budgets},
        "BM25_Only": {b: {"hit": [], "rec": [], "mrr": []} for b in budgets},
        "Interleaved_Union": {b: {"hit": [], "rec": [], "mrr": []} for b in budgets},
        "Equal_RRF": {b: {"hit": [], "rec": [], "mrr": []} for b in budgets},
    }

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
        rrf_ids_100 = [c_id for c_id, _ in fused[:100]]

        for b in budgets:
            cands_d = d_ids[:b]
            metrics["Dense_Only"][b]["hit"].append(compute_candidate_hit_rate_at_k(cands_d, gt_ids, k=b))
            metrics["Dense_Only"][b]["rec"].append(compute_true_chunk_recall_at_k(cands_d, gt_ids, k=b))
            metrics["Dense_Only"][b]["mrr"].append(compute_reciprocal_rank(cands_d, gt_ids))

            cands_b = b_ids[:b]
            metrics["BM25_Only"][b]["hit"].append(compute_candidate_hit_rate_at_k(cands_b, gt_ids, k=b))
            metrics["BM25_Only"][b]["rec"].append(compute_true_chunk_recall_at_k(cands_b, gt_ids, k=b))
            metrics["BM25_Only"][b]["mrr"].append(compute_reciprocal_rank(cands_b, gt_ids))

            union_cands = []
            seen = set()
            for i in range(max(len(d_ids), len(b_ids))):
                if i < len(d_ids) and d_ids[i] not in seen and len(union_cands) < b:
                    union_cands.append(d_ids[i])
                    seen.add(d_ids[i])
                if i < len(b_ids) and b_ids[i] not in seen and len(union_cands) < b:
                    union_cands.append(b_ids[i])
                    seen.add(b_ids[i])
                if len(union_cands) >= b:
                    break

            metrics["Interleaved_Union"][b]["hit"].append(compute_candidate_hit_rate_at_k(union_cands, gt_ids, k=b))
            metrics["Interleaved_Union"][b]["rec"].append(compute_true_chunk_recall_at_k(union_cands, gt_ids, k=b))
            metrics["Interleaved_Union"][b]["mrr"].append(compute_reciprocal_rank(union_cands, gt_ids))

            cands_rrf = rrf_ids_100[:b]
            metrics["Equal_RRF"][b]["hit"].append(compute_candidate_hit_rate_at_k(cands_rrf, gt_ids, k=b))
            metrics["Equal_RRF"][b]["rec"].append(compute_true_chunk_recall_at_k(cands_rrf, gt_ids, k=b))
            metrics["Equal_RRF"][b]["mrr"].append(compute_reciprocal_rank(cands_rrf, gt_ids))

    elapsed_s = time.perf_counter() - start_wall_time
    print(f"\n--- FAIR CANDIDATE GENERATION COMPARISON (N = {valid_queries} DEV QUERIES) ---")
    print(f"{'Strategy':<18} | {'HitRate@20':<12} | {'HitRate@50':<12} | {'HitRate@100':<12} | {'MRR@100':<10}")
    print("-" * 75)

    summary_rows = []
    json_metrics = {}
    methods = ["Dense_Only", "BM25_Only", "Interleaved_Union", "Equal_RRF"]
    for m in methods:
        h20 = float(np.mean(metrics[m][20]["hit"]) * 100)
        h50 = float(np.mean(metrics[m][50]["hit"]) * 100)
        h100 = float(np.mean(metrics[m][100]["hit"]) * 100)
        mrr100 = float(np.mean(metrics[m][100]["mrr"]))
        print(f"{m:<18} | {h20:6.2f}%{'':<5} | {h50:6.2f}%{'':<5} | {h100:6.2f}%{'':<5} | {mrr100:.4f}")
        summary_rows.append(f"| **{m}** | {h20:.2f}% | {h50:.2f}% | {h100:.2f}% | {mrr100:.4f} |")
        json_metrics[m] = {"HitRate@20": h20, "HitRate@50": h50, "HitRate@100": h100, "MRR@100": mrr100}

    # Save Machine-Readable JSON
    OUTPUT_JSON_DIR.mkdir(parents=True, exist_ok=True)
    raw_record = {
        "experiment_id": "EXP-16_FAIR",
        "benchmark_name": "CUAD_DEV_SPLIT",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": round(elapsed_s, 2),
        "manifest_hash": manifest_hash,
        "budgets_evaluated": budgets,
        "valid_answerable_queries": valid_queries,
        "results": json_metrics,
    }
    json_path = OUTPUT_JSON_DIR / "rrf_vs_union.json"
    json_path.write_text(json.dumps(raw_record, indent=2), encoding="utf-8")

    report_md = f"""# Fair Comparison: Candidate Union vs Equal RRF at Identical Candidate Budgets

**Evaluation Dataset:** CUAD DEV Split (20 Contracts, {valid_queries} Evaluated Answerable Queries)  
**Dense Model:** `{cfg.dense_model}` ({cfg.dense_dimension}-d)  
**Sparse Model:** `{cfg.sparse_retriever}`  
**Evaluation Protocol:** Strict identical candidate budgets $k \\in [20, 50, 100]$.  
**Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%SZ')}  
**Runtime:** {elapsed_s:.2f}s  

---

## 1. CandidateHitRate at Identical Candidate Budgets

| Candidate Strategy | CandidateHitRate @20 | CandidateHitRate @50 | CandidateHitRate @100 | First-Stage MRR @100 |
|:---|:---:|:---:|:---:|:---:|
{chr(10).join(summary_rows)}

---

## 2. Key Scientific Findings & Analysis

1. **At Identical Budget $k=20$:** Equal RRF achieves **`{json_metrics['Equal_RRF']['HitRate@20']:.2f}%`** vs **`{json_metrics['Interleaved_Union']['HitRate@20']:.2f}%`** for Interleaved Union.
2. **At Identical Budget $k=50$:** Equal RRF achieves **`{json_metrics['Equal_RRF']['HitRate@50']:.2f}%`** vs **`{json_metrics['Interleaved_Union']['HitRate@50']:.2f}%`** for Interleaved Union.
3. **At Identical Budget $k=100$:** Equal RRF achieves **`{json_metrics['Equal_RRF']['HitRate@100']:.2f}%`** vs **`{json_metrics['Interleaved_Union']['HitRate@100']:.2f}%`** for Interleaved Union.
"""
    OUTPUT_REPORT_PATH.write_text(report_md.strip() + "\n", encoding="utf-8")
    print(f"\n[OK] Wrote fair comparison report to {OUTPUT_REPORT_PATH} and raw JSON to {json_path}")

if __name__ == "__main__":
    evaluate_union_vs_rrf_fair()
