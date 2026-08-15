#!/usr/bin/env python3
"""
Final Frozen V4 Evaluation on Locked Test Sets.
Runs once after configuration freeze (v4 with Pareto-optimal candidate budget k=40).
Saves:
- evaluation/results/phase4/locked_test_v1_v4.json
- evaluation/results/phase4/custom_cuad_holdout_v2_v4.json
"""
import os
import sys
import time
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Set
import numpy as np

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
from backend.app.ingestion.parsers import MasterDocumentParser
from backend.app.ingestion.chunker import StructureAwareParentChildChunker
from backend.app.retrieval.bm25 import BM25Retriever
from backend.app.retrieval.fusion import reciprocal_rank_fusion
from evaluation.dense_retriever_local import InMemoryDenseRetriever
from evaluation.metrics.retrieval_metrics import (
    compute_candidate_hit_rate_at_k, compute_true_chunk_recall_at_k, compute_reciprocal_rank
)

cfg = get_retrieval_config()
RESULTS_DIR = REPO_ROOT / "evaluation" / "results" / "phase4"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CONTRACTS_DIR = REPO_ROOT / "evaluation" / "datasets" / "cuad" / "processed" / "contracts"


def evaluate_v4_on_manifest(manifest_path: Path, exp_id: str, out_filename: str):
    print(f"\n--- EVALUATING FROZEN V4 ON {exp_id} ({manifest_path.name}) ---")
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
    )
    cache = EvaluationCache(cache_key)

    if cache.is_complete():
        print(f"[CACHE HIT] Loaded {exp_id} from cache {cache_key}")
        chunks_data = cache.load_corpus_chunks()
        dense_emb, chunk_ids = cache.load_dense_embeddings()
        q_emb, _ = cache.load_query_embeddings()
        bm25_100, dense_100, rrf_100, gold_map = cache.load_retrieval_candidates()
    else:
        print(f"[CACHE MISS] Building {exp_id} artifacts...")
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
                    "metadata": c.metadata,
                })
        cache.save_corpus_chunks(chunks_data)

        bm25 = BM25Retriever()
        bm25.build_index(all_ids, all_texts, all_metas)

        dense = InMemoryDenseRetriever(model_name=cfg.dense_model)
        dense.build_index(all_ids, all_texts, batch_size=16)
        cache.save_dense_embeddings(dense.embeddings, all_ids)

        all_questions = [q["question"] for q in ans_queries]
        q_vecs = dense.embedder.embed_documents_batch(all_questions, batch_size=16)
        q_arr = np.array(q_vecs, dtype=np.float32)
        q_norms = np.linalg.norm(q_arr, axis=1, keepdims=True)
        q_norms = np.where(q_norms == 0, 1.0, q_norms)
        q_arr = q_arr / q_norms
        cache.save_query_embeddings(q_arr, all_questions)

        bm25_100, dense_100, rrf_100, gold_map = {}, {}, {}, {}
        for q_idx, q in enumerate(ans_queries):
            q_str = str(q_idx)
            question = q["question"]
            cid = q["source_contract_id"]
            gold_ev = q.get("gold_evidence", "").strip().lower()

            gt_ids = []
            for c in chunks_data:
                if c["doc_id"] != cid:
                    continue
                if gold_ev in c["text"].lower() or (c["metadata"] and gold_ev in c["metadata"].get("parent_text", "").lower()):
                    gt_ids.append(c["chunk_id"])
            gold_map[q_str] = gt_ids

            b_hits = bm25.search(question, top_k=100)
            bm25_100[q_str] = b_hits

            q_vec = q_arr[q_idx]
            sims = dense.embeddings @ q_vec
            top_idxs = np.argsort(sims)[::-1][:100]
            dense_100[q_str] = [(dense.chunk_ids[idx], float(sims[idx])) for idx in top_idxs]

            b_ids = [h[0] for h in b_hits]
            d_ids = [h[0] for h in dense_100[q_str]]
            fused = reciprocal_rank_fusion([b_ids, d_ids], k=cfg.rrf_k)
            rrf_100[q_str] = fused[:100]

        cache.save_retrieval_candidates(bm25_100, dense_100, rrf_100, gold_map)
        cache.save_metadata({"manifest_hash": manifest_hash, "exp_id": exp_id})

    chunk_map = {c["chunk_id"]: c for c in chunks_data}
    valid_query_indices = [idx for idx, q in enumerate(ans_queries) if len(gold_map.get(str(idx), [])) > 0]

    reranker = LocalCrossEncoderReranker(
        model_name=cfg.reranker_model, max_length=cfg.reranker_max_seq_length, strict=True
    )

    # In V4, candidate budget is expanded to k=40
    budget_k = cfg.reranker_input_budget
    c100_list, hr1_list, hr5_list, hr10_list, mrr_list = [], [], [], [], []

    for q_idx in valid_query_indices:
        q_str = str(q_idx)
        q = ans_queries[q_idx]
        question = q["question"]
        gt_ids = set(gold_map[q_str])

        cand_ids_100 = [c_id for c_id, _ in rrf_100[q_str]]
        c100_list.append(compute_candidate_hit_rate_at_k(cand_ids_100, gt_ids, k=100))

        dedup_candidates = []
        parent_count = {}
        for c_id in cand_ids_100:
            c_obj = chunk_map.get(c_id)
            p_id = c_obj["parent_id"] if c_obj else None
            if p_id:
                if parent_count.get(p_id, 0) >= 2:
                    continue
                parent_count[p_id] = parent_count.get(p_id, 0) + 1
            dedup_candidates.append(c_id)

        pruned_k = dedup_candidates[:budget_k]
        cand_texts = [chunk_map[c_id]["text"] for c_id in pruned_k if c_id in chunk_map]
        rerank_hits = reranker.rerank(question, cand_texts, top_n=10)
        final_ids = [pruned_k[idx] for idx, _ in rerank_hits if idx < len(pruned_k)]

        hr1_list.append(compute_candidate_hit_rate_at_k(final_ids, gt_ids, k=1))
        hr5_list.append(compute_candidate_hit_rate_at_k(final_ids, gt_ids, k=5))
        hr10_list.append(compute_candidate_hit_rate_at_k(final_ids, gt_ids, k=10))
        mrr_list.append(compute_reciprocal_rank(final_ids, gt_ids))

    c100 = float(np.mean(c100_list) * 100)
    hr1 = float(np.mean(hr1_list) * 100)
    hr5 = float(np.mean(hr5_list) * 100)
    hr10 = float(np.mean(hr10_list) * 100)
    mrr = float(np.mean(mrr_list))

    print(f"Results on {exp_id} (N={len(valid_query_indices)}):")
    print(f"  CandidateHitRate@100: {c100:.2f}%")
    print(f"  HitRate@1:            {hr1:.2f}%")
    print(f"  HitRate@5:            {hr5:.2f}%")
    print(f"  HitRate@10:           {hr10:.2f}%")
    print(f"  MRR:                   {mrr:.4f}")

    record = {
        "experiment_id": f"{exp_id}_V4",
        "pipeline_version": "v4.0.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manifest_hash": manifest_hash,
        "dense_model": cfg.dense_model,
        "reranker_model": cfg.reranker_model,
        "reranker_input_budget": budget_k,
        "total_queries": len(queries),
        "valid_answerable_queries": len(valid_query_indices),
        "metrics": {
            "CandidateHitRate@100": c100,
            "HitRate@1": hr1,
            "HitRate@5": hr5,
            "HitRate@10": hr10,
            "MRR": mrr
        }
    }
    out_file = RESULTS_DIR / out_filename
    out_file.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"[OK] Saved {out_file}")


def main():
    locked_v1 = REPO_ROOT / "evaluation" / "manifests" / "cuad_official_manifest.json"
    holdout_v2 = REPO_ROOT / "evaluation" / "manifests" / "cuad_locked_test_v2_manifest.json"

    evaluate_v4_on_manifest(locked_v1, "LOCKED_TEST_V1", "locked_test_v1_v4.json")
    evaluate_v4_on_manifest(holdout_v2, "CUSTOM_CUAD_HOLDOUT_V2", "custom_cuad_holdout_v2_v4.json")


if __name__ == "__main__":
    main()
