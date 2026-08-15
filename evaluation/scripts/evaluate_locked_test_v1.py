#!/usr/bin/env python3
"""
Evaluation on Frozen LEGACY_LOCKED_TEST_V1 (10 Contracts, 50 queries, 19 answerable).
Exports raw JSON to evaluation/results/phase3_5_1/locked_test_v1.json.
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
from backend.app.providers.reranker import LocalCrossEncoderReranker
from backend.app.ingestion.parsers import MasterDocumentParser
from backend.app.ingestion.chunker import StructureAwareParentChildChunker, IndexedChunk
from evaluation.dense_retriever_local import InMemoryDenseRetriever
from evaluation.metrics.retrieval_metrics import (
    compute_candidate_hit_rate_at_k, compute_true_chunk_recall_at_k, compute_reciprocal_rank
)

cfg = get_retrieval_config()
MANIFEST_PATH = REPO_ROOT / "evaluation" / "manifests" / "cuad_official_manifest.json"
CONTRACTS_DIR = REPO_ROOT / "evaluation" / "datasets" / "cuad" / "processed" / "contracts"
REPORT_PATH = REPO_ROOT / "evaluation" / "reports" / "RETRIEVAL_BENCHMARK_LOCKED_TEST.md"
OUTPUT_JSON_DIR = REPO_ROOT / "evaluation" / "results" / "phase3_5_1"

def run_legacy_test():
    start_wall_time = time.perf_counter()
    print("=" * 80)
    print("EVALUATING ON LEGACY_LOCKED_TEST_V1 (10 CONTRACTS)")
    print(f"Dense: {cfg.dense_model} | Reranker: {cfg.reranker_model} (strict=True)")
    print("=" * 80)

    manifest_raw = MANIFEST_PATH.read_bytes()
    manifest_hash = hashlib.sha256(manifest_raw).hexdigest()
    manifest_data = json.loads(manifest_raw.decode("utf-8"))
    contracts_info = manifest_data["contracts"]
    queries = manifest_data["queries"]
    ans_queries = [q for q in queries if not q.get("is_unanswerable", False)]

    chunker = StructureAwareParentChildChunker(
        child_target_tokens=cfg.child_target_tokens, child_overlap_tokens=cfg.child_overlap_tokens,
        parent_target_tokens=cfg.parent_target_tokens, parent_overlap_tokens=cfg.parent_overlap_tokens
    )
    reranker = LocalCrossEncoderReranker(
        model_name=cfg.reranker_model, max_length=cfg.reranker_max_seq_length, strict=True
    )
    reranker.rerank("warmup", ["warmup text"], top_n=1)

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

    hr1_list, hr5_list, hr10_list, mrr_list, c100_list = [], [], [], [], []
    valid_queries = 0
    reranker_failure_count = 0

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
        cand_ids_100 = [c_id for c_id, _ in fused[:100]]

        c100_list.append(compute_candidate_hit_rate_at_k(cand_ids_100, gt_ids, k=100))

        dedup_candidates = []
        parent_count = {}
        for c_id in cand_ids_100:
            c_obj = chunk_dict.get(c_id)
            p_id = c_obj.parent_id if c_obj else None
            if p_id:
                if parent_count.get(p_id, 0) >= cfg.max_child_chunks_per_parent:
                    continue
                parent_count[p_id] = parent_count.get(p_id, 0) + 1
            dedup_candidates.append(c_id)

        pruned_top20 = dedup_candidates[:cfg.reranker_input_budget]

        cand_texts = [chunk_dict[c_id].text for c_id in pruned_top20 if c_id in chunk_dict]
        try:
            rerank_hits = reranker.rerank(question, cand_texts, top_n=cfg.reranker_top_n)
        except Exception as e:
            reranker_failure_count += 1
            raise
        final_ids = [pruned_top20[idx] for idx, _ in rerank_hits if idx < len(pruned_top20)]

        hr1_list.append(compute_candidate_hit_rate_at_k(final_ids, gt_ids, k=1))
        hr5_list.append(compute_candidate_hit_rate_at_k(final_ids, gt_ids, k=5))
        hr10_list.append(compute_candidate_hit_rate_at_k(final_ids, gt_ids, k=10))
        mrr_list.append(compute_reciprocal_rank(final_ids, gt_ids))

    elapsed_s = time.perf_counter() - start_wall_time
    c100 = float(np.mean(c100_list) * 100)
    hr1 = float(np.mean(hr1_list) * 100)
    hr5 = float(np.mean(hr5_list) * 100)
    hr10 = float(np.mean(hr10_list) * 100)
    mrr = float(np.mean(mrr_list))

    print(f"\n--- LEGACY_LOCKED_TEST_V1 RESULTS (N = {valid_queries} Answerable Queries) ---")
    print(f"  CandidateHitRate @100: {c100:.2f}%")
    print(f"  HitRate @1:            {hr1:.2f}%")
    print(f"  HitRate @5:            {hr5:.2f}%")
    print(f"  HitRate @10:           {hr10:.2f}%")
    print(f"  MRR:                   {mrr:.4f}")

    # Save Machine-Readable JSON
    OUTPUT_JSON_DIR.mkdir(parents=True, exist_ok=True)
    raw_record = {
        "experiment_id": "LOCKED_TEST_V1",
        "benchmark_name": "CUAD_OFFICIAL_10_CONTRACT_SPLIT",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": round(elapsed_s, 2),
        "manifest_hash": manifest_hash,
        "dense_model": cfg.dense_model,
        "reranker_model": cfg.reranker_model,
        "reranker_failure_count": reranker_failure_count,
        "total_queries": len(queries),
        "valid_answerable_queries": valid_queries,
        "metrics": {
            "CandidateHitRate@100": c100,
            "HitRate@1": hr1,
            "HitRate@5": hr5,
            "HitRate@10": hr10,
            "MRR": mrr,
        }
    }
    json_path = OUTPUT_JSON_DIR / "locked_test_v1.json"
    json_path.write_text(json.dumps(raw_record, indent=2), encoding="utf-8")

    report_md = f"""# Locked Test V1 Retrieval Benchmark Report

**Dataset:** CUAD Locked Test V1 (10 Contracts, {valid_queries} Answerable Queries)  
**Dense Model:** `{cfg.dense_model}` ({cfg.dense_dimension}-d)  
**Reranker Model:** `{cfg.reranker_model}` (strict=True, full text input)  
**Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%SZ')}  
**Runtime:** {elapsed_s:.2f}s  

---

## 1. Measured Retrieval Performance

| Metric | Measured Value | Scope |
|:---|:---:|:---|
| **CandidateHitRate@100** | **{c100:.2f}%** | First-Stage $RRF_{{60}}$ Broad Pool |
| **HitRate@1** | **{hr1:.2f}%** | Top-1 Exact Clause Accuracy |
| **HitRate@5** | **{hr5:.2f}%** | Top-5 Context Window |
| **HitRate@10** | **{hr10:.2f}%** | Top-10 Output |
| **MRR** | **{mrr:.4f}** | Mean Reciprocal Rank |
"""
    REPORT_PATH.write_text(report_md.strip() + "\n", encoding="utf-8")
    print(f"\n[OK] Wrote locked test report to {REPORT_PATH} and raw JSON to {json_path}")

if __name__ == "__main__":
    run_legacy_test()
