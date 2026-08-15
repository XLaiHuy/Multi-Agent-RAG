#!/usr/bin/env python3
"""
Post-Pruning Recall & HitRate Waterfall Audit.
Evaluates the 5-stage funnel using strict evaluation mode and full chunk text:
Stage 1: Raw Top-100 First-Stage Retrieval (RRF k=60)
Stage 2: After Parent Deduplication (Max 2 child chunks per parent block)
Stage 3: After Top-20 Truncation (RRF-order budget reduction for reranker)
Stage 4: After CrossEncoder Reranking (Top-10 output)
Stage 5: Final Top-5 Context Window (Top-5 output)
Exports raw JSON to evaluation/results/phase3_5_1/waterfall.json.
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
DEV_MANIFEST_PATH = REPO_ROOT / "evaluation" / "manifests" / "cuad_dev_manifest.json"
CONTRACTS_DIR = REPO_ROOT / "evaluation" / "datasets" / "cuad" / "processed" / "contracts"
REPORT_PATH = REPO_ROOT / "evaluation" / "reports" / "POST_PRUNING_RECALL_WATERFALL.md"
OUTPUT_JSON_DIR = REPO_ROOT / "evaluation" / "results" / "phase3_5_1"

def run_waterfall_audit():
    start_wall_time = time.perf_counter()
    print("=" * 80)
    print("POST-PRUNING RECALL & HITRATE WATERFALL AUDIT")
    print(f"Dense: {cfg.dense_model} ({cfg.dense_dimension}-d) | Reranker: {cfg.reranker_model} (strict=True, max_length={cfg.reranker_max_seq_length})")
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
    reranker = LocalCrossEncoderReranker(
        model_name=cfg.reranker_model, max_length=cfg.reranker_max_seq_length, strict=True
    )
    reranker.rerank("warmup query", ["warmup document clause text"], top_n=1)

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

    funnel = {
        "stage1_top100_raw": {"hit": [], "rec": []},
        "stage2_after_parent_dedup": {"hit": [], "rec": []},
        "stage3_after_trunc_top20": {"hit": [], "rec": []},
        "stage4_after_rerank_top10": {"hit": [], "rec": []},
        "stage5_after_rerank_top5": {"hit": [], "rec": []},
    }

    valid_queries = 0
    reranker_failure_count = 0
    reranker_latencies_ms = []

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

        funnel["stage1_top100_raw"]["hit"].append(compute_candidate_hit_rate_at_k(cand_ids_100, gt_ids, k=100))
        funnel["stage1_top100_raw"]["rec"].append(compute_true_chunk_recall_at_k(cand_ids_100, gt_ids, k=100))

        # Stage 2: Parent Dedup
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

        funnel["stage2_after_parent_dedup"]["hit"].append(compute_candidate_hit_rate_at_k(dedup_candidates, gt_ids, k=len(dedup_candidates)))
        funnel["stage2_after_parent_dedup"]["rec"].append(compute_true_chunk_recall_at_k(dedup_candidates, gt_ids, k=len(dedup_candidates)))

        # Stage 3: Top-20 Truncation
        pruned_top20 = dedup_candidates[:cfg.reranker_input_budget]
        funnel["stage3_after_trunc_top20"]["hit"].append(compute_candidate_hit_rate_at_k(pruned_top20, gt_ids, k=len(pruned_top20)))
        funnel["stage3_after_trunc_top20"]["rec"].append(compute_true_chunk_recall_at_k(pruned_top20, gt_ids, k=len(pruned_top20)))

        # Stage 4 & 5: Strict CrossEncoder Rerank (FULL chunk text passed)
        cand_texts = [chunk_dict[c_id].text for c_id in pruned_top20 if c_id in chunk_dict]
        t_rerank_start = time.perf_counter()
        try:
            rerank_hits = reranker.rerank(question, cand_texts, top_n=cfg.reranker_top_n)
        except Exception as e:
            reranker_failure_count += 1
            raise
        reranker_latencies_ms.append((time.perf_counter() - t_rerank_start) * 1000.0)

        final_ids_10 = [pruned_top20[idx] for idx, _ in rerank_hits if idx < len(pruned_top20)]
        final_ids_5 = final_ids_10[:5]

        funnel["stage4_after_rerank_top10"]["hit"].append(compute_candidate_hit_rate_at_k(final_ids_10, gt_ids, k=10))
        funnel["stage4_after_rerank_top10"]["rec"].append(compute_true_chunk_recall_at_k(final_ids_10, gt_ids, k=10))

        funnel["stage5_after_rerank_top5"]["hit"].append(compute_candidate_hit_rate_at_k(final_ids_5, gt_ids, k=5))
        funnel["stage5_after_rerank_top5"]["rec"].append(compute_true_chunk_recall_at_k(final_ids_5, gt_ids, k=5))

    elapsed_s = time.perf_counter() - start_wall_time
    s1_h = float(np.mean(funnel["stage1_top100_raw"]["hit"]) * 100)
    s1_r = float(np.mean(funnel["stage1_top100_raw"]["rec"]) * 100)

    s2_h = float(np.mean(funnel["stage2_after_parent_dedup"]["hit"]) * 100)
    s2_r = float(np.mean(funnel["stage2_after_parent_dedup"]["rec"]) * 100)

    s3_h = float(np.mean(funnel["stage3_after_trunc_top20"]["hit"]) * 100)
    s3_r = float(np.mean(funnel["stage3_after_trunc_top20"]["rec"]) * 100)

    s4_h = float(np.mean(funnel["stage4_after_rerank_top10"]["hit"]) * 100)
    s4_r = float(np.mean(funnel["stage4_after_rerank_top10"]["rec"]) * 100)

    s5_h = float(np.mean(funnel["stage5_after_rerank_top5"]["hit"]) * 100)
    s5_r = float(np.mean(funnel["stage5_after_rerank_top5"]["rec"]) * 100)

    p50_latency_ms = float(np.median(reranker_latencies_ms))
    p95_latency_ms = float(np.percentile(reranker_latencies_ms, 95))

    print(f"\n--- POST-PRUNING WATERFALL FUNNEL (N = {valid_queries} DEV QUERIES) ---")
    print(f"Stage 1: Top-100 Raw Retrieval:           HitRate={s1_h:6.2f}%, TrueRecall={s1_r:6.2f}%")
    print(f"Stage 2: After Parent Deduplication:     HitRate={s2_h:6.2f}%, TrueRecall={s2_r:6.2f}% (Hit Loss: {s1_h-s2_h:+.2f}%)")
    print(f"Stage 3: After Top-20 Truncation:        HitRate={s3_h:6.2f}%, TrueRecall={s3_r:6.2f}% (Hit Loss: {s2_h-s3_h:+.2f}%)")
    print(f"Stage 4: After CrossEncoder Top-10:      HitRate={s4_h:6.2f}%, TrueRecall={s4_r:6.2f}% (Hit Loss: {s3_h-s4_h:+.2f}%)")
    print(f"Stage 5: After CrossEncoder Top-5:       HitRate={s5_h:6.2f}%, TrueRecall={s5_r:6.2f}% (Hit Loss: {s4_h-s5_h:+.2f}%)")
    print(f"Reranker Latency P50: {p50_latency_ms:.1f}ms | P95: {p95_latency_ms:.1f}ms | Failure Count: {reranker_failure_count}")

    # Save Machine-Readable JSON
    OUTPUT_JSON_DIR.mkdir(parents=True, exist_ok=True)
    raw_record = {
        "experiment_id": "EXP-17_WATERFALL",
        "benchmark_name": "CUAD_DEV_SPLIT",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": round(elapsed_s, 2),
        "manifest_hash": manifest_hash,
        "dense_model": cfg.dense_model,
        "reranker_model": cfg.reranker_model,
        "reranker_failure_count": reranker_failure_count,
        "reranker_p50_ms": round(p50_latency_ms, 2),
        "reranker_p95_ms": round(p95_latency_ms, 2),
        "valid_answerable_queries": valid_queries,
        "stages": {
            "Stage 1 - Raw Top-100": {"CandidateHitRate": s1_h, "TrueChunkRecall": s1_r},
            "Stage 2 - Parent Dedup": {"CandidateHitRate": s2_h, "TrueChunkRecall": s2_r},
            "Stage 3 - Top-20 Truncation": {"CandidateHitRate": s3_h, "TrueChunkRecall": s3_r},
            "Stage 4 - CrossEncoder Top-10": {"CandidateHitRate": s4_h, "TrueChunkRecall": s4_r},
            "Stage 5 - CrossEncoder Top-5": {"CandidateHitRate": s5_h, "TrueChunkRecall": s5_r},
        }
    }
    json_path = OUTPUT_JSON_DIR / "waterfall.json"
    json_path.write_text(json.dumps(raw_record, indent=2), encoding="utf-8")

    report_md = f"""# Post-Pruning Recall & HitRate Waterfall Audit

**Evaluation Dataset:** CUAD DEV Split (20 Contracts, {valid_queries} Evaluated Answerable Queries)  
**Dense Model:** `{cfg.dense_model}` ({cfg.dense_dimension}-d)  
**Reranker Model:** `{cfg.reranker_model}` (`strict=True`, `max_seq_length={cfg.reranker_max_seq_length}`, full text passed)  
**Reranker Failures:** `{reranker_failure_count}`  
**Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%SZ')}  
**Runtime:** {elapsed_s:.2f}s  

---

## 1. Step-by-Step Recall & Coverage Waterfall Funnel

| Funnel Stage | Description | CandidateHitRate (Any-Gold) | TrueChunkRecall (All-Gold) | HitRate Loss vs Prev | HitRate Retention vs Prev |
|:---|:---|:---:|:---:|:---:|:---:|
| **Stage 1** | Raw Top-100 First-Stage Retrieval ($RRF_{{60}}$) | **{s1_h:.2f}%** | **{s1_r:.2f}%** | Baseline | 100.0% |
| **Stage 2** | After Parent Deduplication (Max 2 chunks/parent) | **{s2_h:.2f}%** | **{s2_r:.2f}%** | {s1_h-s2_h:+.2f}% | {s2_h/s1_h*100:.1f}% |
| **Stage 3** | After Top-20 Truncation (RRF-Order Budget Input) | **{s3_h:.2f}%** | **{s3_r:.2f}%** | {s2_h-s3_h:+.2f}% | {s3_h/s2_h*100:.1f}% |
| **Stage 4** | After CrossEncoder Reranking (Top-10 Output) | **{s4_h:.2f}%** | **{s4_r:.2f}%** | {s3_h-s4_h:+.2f}% | {s4_h/s3_h*100:.1f}% |
| **Stage 5** | Final Top-5 Context Window (Top-5 Output) | **{s5_h:.2f}%** | **{s5_r:.2f}%** | {s4_h-s5_h:+.2f}% | {s5_h/s3_h*100:.1f}% |
"""
    REPORT_PATH.write_text(report_md.strip() + "\n", encoding="utf-8")
    print(f"\n[OK] Wrote Waterfall report to {REPORT_PATH} and raw JSON to {json_path}")

if __name__ == "__main__":
    run_waterfall_audit()
