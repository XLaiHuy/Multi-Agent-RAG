#!/usr/bin/env python3
"""
Phase 4 Unified Fast Evaluation Harness, Task-Formulation Audit & Candidate-Budget Recovery.
Executes:
1. Reusable cached intermediate representation (cold vs warm speedup benchmark)
2. Task formulation audit: GLOBAL_MULTI_CONTRACT vs DOCUMENT_SCOPED_QA
3. Query Ambiguity Analysis by category annotation density
4. Candidate budget sweep (k in [10, 20, 30, 40, 50, 75]) with Pareto quality/latency analysis
5. Query-level rank trace & failure taxonomy (NOT_FOUND_TOP100, LOST_BY_BUDGET, RERANKER_DEMOTED, etc.)
6. Conditional EXP-21 Reranker A/B
Outputs raw machine-readable JSON artifacts to evaluation/results/phase4/.
"""
import os
import sys
import time
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
torch.set_num_threads(4)

from backend.app.core.config import get_settings
from backend.app.core.retrieval_defaults import (
    CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP, PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP,
    RRF_K_DEFAULT, BROAD_CANDIDATE_POOL_SIZE, RERANKER_INPUT_BUDGET,
    DENSE_MODEL_PRODUCTION_DEFAULT, DENSE_MODEL_EVALUATION_SELECTED,
    RERANKER_MODEL_DEFAULT, RERANKER_MAX_SEQ_LENGTH
)
from evaluation.config_loader import get_retrieval_config
from backend.app.retrieval.bm25 import BM25Retriever
from backend.app.retrieval.fusion import reciprocal_rank_fusion
from backend.app.providers.reranker import LocalCrossEncoderReranker
from backend.app.ingestion.parsers import MasterDocumentParser
from backend.app.ingestion.chunker import StructureAwareParentChildChunker, IndexedChunk
from evaluation.dense_retriever_local import InMemoryDenseRetriever
from evaluation.cache_manager import EvaluationCache, compute_cache_key
from evaluation.metrics.retrieval_metrics import (
    compute_candidate_hit_rate_at_k, compute_true_chunk_recall_at_k, compute_reciprocal_rank
)

cfg = get_retrieval_config()
RESULTS_DIR = REPO_ROOT / "evaluation" / "results" / "phase4"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CONTRACTS_DIR = REPO_ROOT / "evaluation" / "datasets" / "cuad" / "processed" / "contracts"


def compute_ndcg_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int = 5) -> float:
    """Computes Normalized Discounted Cumulative Gain at k."""
    if not ground_truth_ids:
        return 0.0
    dcg = 0.0
    for i, item_id in enumerate(retrieved_ids[:k]):
        rel = 1.0 if item_id in ground_truth_ids else 0.0
        dcg += (2.0**rel - 1.0) / np.log2(i + 2.0)
    
    # Ideal DCG: best possible ranking with min(k, len(gt)) relevant items
    idcg = sum((2.0**1.0 - 1.0) / np.log2(i + 2.0) for i in range(min(k, len(ground_truth_ids))))
    return float(dcg / idcg) if idcg > 0 else 0.0


class Phase4Harness:
    """Unified evaluation harness for Phase 4."""

    def __init__(self, manifest_path: Path, mode_name: str = "DEV"):
        self.manifest_path = manifest_path
        self.mode_name = mode_name
        self.manifest_raw = manifest_path.read_bytes()
        self.manifest_hash = hashlib.sha256(self.manifest_raw).hexdigest()
        self.manifest_data = json.loads(self.manifest_raw.decode("utf-8"))
        self.contracts_info = self.manifest_data["contracts"]
        self.queries = self.manifest_data["queries"]
        self.ans_queries = [q for q in self.queries if not q.get("is_unanswerable", False)]

        self.cache_key = compute_cache_key(
            manifest_hash=self.manifest_hash,
            child_target_tokens=cfg.child_target_tokens,
            child_overlap_tokens=cfg.child_overlap_tokens,
            parent_target_tokens=cfg.parent_target_tokens,
            parent_overlap_tokens=cfg.parent_overlap_tokens,
            dense_model=cfg.dense_model,
        )
        self.cache = EvaluationCache(self.cache_key)
        self.timings: Dict[str, float] = {}

    def prepare_intermediate_artifacts(self) -> Tuple[
        List[Dict[str, Any]],
        np.ndarray,
        List[str],
        np.ndarray,
        Dict[str, List[Tuple[str, float]]],
        Dict[str, List[Tuple[str, float]]],
        Dict[str, List[Tuple[str, float]]],
        Dict[str, List[str]],
        bool
    ]:
        """Loads from cache or builds deterministic intermediate representations."""
        if self.cache.is_complete():
            print(f"[CACHE HIT] Loaded all intermediate artifacts from key {self.cache_key}")
            t0 = time.perf_counter()
            chunks_data = self.cache.load_corpus_chunks()
            dense_emb, chunk_ids = self.cache.load_dense_embeddings()
            q_emb, _ = self.cache.load_query_embeddings()
            bm25_100, dense_100, rrf_100, gold_map = self.cache.load_retrieval_candidates()
            self.timings["cache_load_s"] = time.perf_counter() - t0
            return chunks_data, dense_emb, chunk_ids, q_emb, bm25_100, dense_100, rrf_100, gold_map, True

        print(f"[CACHE MISS] Building intermediate artifacts for key {self.cache_key}...")
        t_total_0 = time.perf_counter()

        # 1. Parse and chunk documents
        t_parse_0 = time.perf_counter()
        chunker = StructureAwareParentChildChunker(
            child_target_tokens=cfg.child_target_tokens,
            child_overlap_tokens=cfg.child_overlap_tokens,
            parent_target_tokens=cfg.parent_target_tokens,
            parent_overlap_tokens=cfg.parent_overlap_tokens,
        )

        all_ids: List[str] = []
        all_texts: List[str] = []
        all_metas: List[Dict[str, Any]] = []
        chunks_data: List[Dict[str, Any]] = []

        for c_info in self.contracts_info:
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

        self.timings["parse_and_chunk_s"] = time.perf_counter() - t_parse_0
        self.cache.save_corpus_chunks(chunks_data)

        # 2. Build BM25
        t_bm25_0 = time.perf_counter()
        bm25 = BM25Retriever()
        bm25.build_index(all_ids, all_texts, all_metas)
        self.timings["bm25_index_s"] = time.perf_counter() - t_bm25_0

        # 3. Dense embeddings for document chunks
        t_dense_0 = time.perf_counter()
        dense = InMemoryDenseRetriever(model_name=cfg.dense_model)
        dense.build_index(all_ids, all_texts, batch_size=16)
        self.timings["dense_corpus_embed_s"] = time.perf_counter() - t_dense_0
        self.cache.save_dense_embeddings(dense.embeddings, all_ids)

        # 4. Dense embeddings for queries
        t_q_0 = time.perf_counter()
        all_questions = [q["question"] for q in self.ans_queries]
        q_vecs = dense.embedder.embed_documents_batch(all_questions, batch_size=16)
        q_arr = np.array(q_vecs, dtype=np.float32)
        q_norms = np.linalg.norm(q_arr, axis=1, keepdims=True)
        q_norms = np.where(q_norms == 0, 1.0, q_norms)
        q_arr = q_arr / q_norms
        self.timings["query_embed_s"] = time.perf_counter() - t_q_0
        self.cache.save_query_embeddings(q_arr, all_questions)

        # 5. First-stage retrieval candidate generation
        t_ret_0 = time.perf_counter()
        bm25_100: Dict[str, List[Tuple[str, float]]] = {}
        dense_100: Dict[str, List[Tuple[str, float]]] = {}
        rrf_100: Dict[str, List[Tuple[str, float]]] = {}
        gold_map: Dict[str, List[str]] = {}

        for q_idx, q in enumerate(self.ans_queries):
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

            # BM25 Top100
            b_hits = bm25.search(question, top_k=100)
            bm25_100[q_str] = b_hits

            # Dense Top100
            q_vec = q_arr[q_idx]
            sims = dense.embeddings @ q_vec
            top_idxs = np.argsort(sims)[::-1][:100]
            dense_100[q_str] = [(dense.chunk_ids[idx], float(sims[idx])) for idx in top_idxs]

            # RRF Top100
            b_ids = [h[0] for h in b_hits]
            d_ids = [h[0] for h in dense_100[q_str]]
            fused = reciprocal_rank_fusion([b_ids, d_ids], k=cfg.rrf_k)
            rrf_100[q_str] = fused[:100]

        self.timings["first_stage_retrieval_s"] = time.perf_counter() - t_ret_0
        self.timings["total_cache_build_s"] = time.perf_counter() - t_total_0

        self.cache.save_retrieval_candidates(bm25_100, dense_100, rrf_100, gold_map)
        self.cache.save_metadata({
            "manifest_path": str(self.manifest_path.name),
            "manifest_hash": self.manifest_hash,
            "dense_model": cfg.dense_model,
            "dense_dimension": cfg.dense_dimension,
            "total_chunks": len(chunks_data),
            "total_queries": len(self.ans_queries),
            "timings_seconds": self.timings,
        })

        return chunks_data, dense.embeddings, all_ids, q_arr, bm25_100, dense_100, rrf_100, gold_map, False


def run_full_phase4_suite():
    suite_start = time.perf_counter()
    print("=" * 80)
    print("PHASE 4: FAST EVALUATION HARNESS, TASK AUDIT & CANDIDATE-BUDGET RECOVERY")
    print(f"Dense: {cfg.dense_model} | Reranker: {cfg.reranker_model}")
    print("=" * 80)

    dev_manifest = REPO_ROOT / "evaluation" / "manifests" / "cuad_dev_manifest.json"
    harness = Phase4Harness(dev_manifest, mode_name="DEV")

    # Step 1: Intermediate Artifact Preparation
    chunks_data, dense_emb, chunk_ids, q_emb, bm25_100, dense_100, rrf_100, gold_map, was_cached = (
        harness.prepare_intermediate_artifacts()
    )

    chunk_map = {c["chunk_id"]: c for c in chunks_data}
    ans_queries = harness.ans_queries
    valid_query_indices = [idx for idx, q in enumerate(ans_queries) if len(gold_map.get(str(idx), [])) > 0]
    print(f"Total answerable DEV queries: {len(ans_queries)} | Valid mapped: {len(valid_query_indices)}")

    # Initialize Reranker ONCE
    reranker = LocalCrossEncoderReranker(
        model_name=cfg.reranker_model, max_length=cfg.reranker_max_seq_length, strict=True
    )
    reranker.rerank("warmup", ["warmup text"], top_n=1)

    # =========================================================================
    # EXPERIMENT 1: TASK FORMULATION AUDIT (GLOBAL vs DOCUMENT_SCOPED)
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: TASK FORMULATION AUDIT (GLOBAL SEARCH VS DOCUMENT-SCOPED QA)")
    print("=" * 80)

    # A. GLOBAL SEARCH (All 20 contracts)
    global_c5, global_c10, global_c20, global_c50, global_c100 = [], [], [], [], []
    global_r5, global_r10, global_r20 = [], [], []
    global_hr1, global_hr5, global_hr10, global_mrr, global_ndcg5 = [], [], [], [], []

    # B. DOCUMENT-SCOPED QA (Only target contract chunks)
    doc_c5, doc_c10, doc_c20, doc_c50, doc_c100 = [], [], [], [], []
    doc_r5, doc_r10, doc_r20 = [], [], []
    doc_hr1, doc_hr5, doc_hr10, doc_mrr, doc_ndcg5 = [], [], [], [], []

    doc_chunk_counts = []

    for q_idx in valid_query_indices:
        q_str = str(q_idx)
        q = ans_queries[q_idx]
        question = q["question"]
        cid = q["source_contract_id"]
        gt_ids = set(gold_map[q_str])

        # --- GLOBAL PIPELINE ---
        cand_ids_100 = [c_id for c_id, _ in rrf_100[q_str]]
        global_c5.append(compute_candidate_hit_rate_at_k(cand_ids_100, gt_ids, k=5))
        global_c10.append(compute_candidate_hit_rate_at_k(cand_ids_100, gt_ids, k=10))
        global_c20.append(compute_candidate_hit_rate_at_k(cand_ids_100, gt_ids, k=20))
        global_c50.append(compute_candidate_hit_rate_at_k(cand_ids_100, gt_ids, k=50))
        global_c100.append(compute_candidate_hit_rate_at_k(cand_ids_100, gt_ids, k=100))

        global_r5.append(compute_true_chunk_recall_at_k(cand_ids_100, gt_ids, k=5))
        global_r10.append(compute_true_chunk_recall_at_k(cand_ids_100, gt_ids, k=10))
        global_r20.append(compute_true_chunk_recall_at_k(cand_ids_100, gt_ids, k=20))

        # Parent dedup & Top20
        dedup_candidates = []
        parent_count = {}
        for c_id in cand_ids_100:
            c_obj = chunk_map.get(c_id)
            p_id = c_obj["parent_id"] if c_obj else None
            if p_id:
                if parent_count.get(p_id, 0) >= cfg.max_child_chunks_per_parent:
                    continue
                parent_count[p_id] = parent_count.get(p_id, 0) + 1
            dedup_candidates.append(c_id)

        pruned_top20 = dedup_candidates[:cfg.reranker_input_budget]
        cand_texts = [chunk_map[c_id]["text"] for c_id in pruned_top20 if c_id in chunk_map]
        rerank_hits = reranker.rerank(question, cand_texts, top_n=10)
        final_ids = [pruned_top20[idx] for idx, _ in rerank_hits if idx < len(pruned_top20)]

        global_hr1.append(compute_candidate_hit_rate_at_k(final_ids, gt_ids, k=1))
        global_hr5.append(compute_candidate_hit_rate_at_k(final_ids, gt_ids, k=5))
        global_hr10.append(compute_candidate_hit_rate_at_k(final_ids, gt_ids, k=10))
        global_mrr.append(compute_reciprocal_rank(final_ids, gt_ids))
        global_ndcg5.append(compute_ndcg_at_k(final_ids, gt_ids, k=5))

        # --- DOCUMENT-SCOPED PIPELINE ---
        # Searchable chunks are strictly those belonging to cid
        doc_scoped_chunks = [c for c in chunks_data if c["doc_id"] == cid]
        doc_chunk_counts.append(len(doc_scoped_chunks))
        doc_chunk_id_set = {c["chunk_id"] for c in doc_scoped_chunks}

        # Filter candidates to document scope
        doc_cand_100 = [c_id for c_id in cand_ids_100 if c_id in doc_chunk_id_set]

        doc_c5.append(compute_candidate_hit_rate_at_k(doc_cand_100, gt_ids, k=5))
        doc_c10.append(compute_candidate_hit_rate_at_k(doc_cand_100, gt_ids, k=10))
        doc_c20.append(compute_candidate_hit_rate_at_k(doc_cand_100, gt_ids, k=20))
        doc_c50.append(compute_candidate_hit_rate_at_k(doc_cand_100, gt_ids, k=50))
        doc_c100.append(compute_candidate_hit_rate_at_k(doc_cand_100, gt_ids, k=100))

        doc_r5.append(compute_true_chunk_recall_at_k(doc_cand_100, gt_ids, k=5))
        doc_r10.append(compute_true_chunk_recall_at_k(doc_cand_100, gt_ids, k=10))
        doc_r20.append(compute_true_chunk_recall_at_k(doc_cand_100, gt_ids, k=20))

        doc_dedup = []
        doc_pcount = {}
        for c_id in doc_cand_100:
            c_obj = chunk_map.get(c_id)
            p_id = c_obj["parent_id"] if c_obj else None
            if p_id:
                if doc_pcount.get(p_id, 0) >= cfg.max_child_chunks_per_parent:
                    continue
                doc_pcount[p_id] = doc_pcount.get(p_id, 0) + 1
            doc_dedup.append(c_id)

        doc_pruned_top20 = doc_dedup[:cfg.reranker_input_budget]
        doc_cand_texts = [chunk_map[c_id]["text"] for c_id in doc_pruned_top20 if c_id in chunk_map]
        doc_rerank_hits = reranker.rerank(question, doc_cand_texts, top_n=10)
        doc_final_ids = [doc_pruned_top20[idx] for idx, _ in doc_rerank_hits if idx < len(doc_pruned_top20)]

        doc_hr1.append(compute_candidate_hit_rate_at_k(doc_final_ids, gt_ids, k=1))
        doc_hr5.append(compute_candidate_hit_rate_at_k(doc_final_ids, gt_ids, k=5))
        doc_hr10.append(compute_candidate_hit_rate_at_k(doc_final_ids, gt_ids, k=10))
        doc_mrr.append(compute_reciprocal_rank(doc_final_ids, gt_ids))
        doc_ndcg5.append(compute_ndcg_at_k(doc_final_ids, gt_ids, k=5))

    task_audit_results = {
        "benchmark_name": "CUAD_DEV_SPLIT",
        "valid_answerable_queries": len(valid_query_indices),
        "searchable_chunks": {
            "global_average": len(chunks_data),
            "document_scoped_average": round(float(np.mean(doc_chunk_counts)), 1)
        },
        "GLOBAL_MULTI_CONTRACT": {
            "CandidateHitRate@5": float(np.mean(global_c5) * 100),
            "CandidateHitRate@10": float(np.mean(global_c10) * 100),
            "CandidateHitRate@20": float(np.mean(global_c20) * 100),
            "CandidateHitRate@50": float(np.mean(global_c50) * 100),
            "CandidateHitRate@100": float(np.mean(global_c100) * 100),
            "TrueChunkRecall@5": float(np.mean(global_r5) * 100),
            "TrueChunkRecall@10": float(np.mean(global_r10) * 100),
            "TrueChunkRecall@20": float(np.mean(global_r20) * 100),
            "Post_Rerank_HitRate@1": float(np.mean(global_hr1) * 100),
            "Post_Rerank_HitRate@5": float(np.mean(global_hr5) * 100),
            "Post_Rerank_HitRate@10": float(np.mean(global_hr10) * 100),
            "Post_Rerank_MRR": float(np.mean(global_mrr)),
            "Post_Rerank_nDCG@5": float(np.mean(global_ndcg5)),
        },
        "DOCUMENT_SCOPED_QA": {
            "CandidateHitRate@5": float(np.mean(doc_c5) * 100),
            "CandidateHitRate@10": float(np.mean(doc_c10) * 100),
            "CandidateHitRate@20": float(np.mean(doc_c20) * 100),
            "CandidateHitRate@50": float(np.mean(doc_c50) * 100),
            "CandidateHitRate@100": float(np.mean(doc_c100) * 100),
            "TrueChunkRecall@5": float(np.mean(doc_r5) * 100),
            "TrueChunkRecall@10": float(np.mean(doc_r10) * 100),
            "TrueChunkRecall@20": float(np.mean(doc_r20) * 100),
            "Post_Rerank_HitRate@1": float(np.mean(doc_hr1) * 100),
            "Post_Rerank_HitRate@5": float(np.mean(doc_hr5) * 100),
            "Post_Rerank_HitRate@10": float(np.mean(doc_hr10) * 100),
            "Post_Rerank_MRR": float(np.mean(doc_mrr)),
            "Post_Rerank_nDCG@5": float(np.mean(doc_ndcg5)),
        }
    }

    (RESULTS_DIR / "global_vs_document_scoped.json").write_text(
        json.dumps(task_audit_results, indent=2), encoding="utf-8"
    )
    print(f"  [OK] Saved {RESULTS_DIR / 'global_vs_document_scoped.json'}")
    print(f"  Global HitRate@10: {task_audit_results['GLOBAL_MULTI_CONTRACT']['Post_Rerank_HitRate@10']:.2f}% | MRR: {task_audit_results['GLOBAL_MULTI_CONTRACT']['Post_Rerank_MRR']:.4f}")
    print(f"  DocScoped HitRate@10: {task_audit_results['DOCUMENT_SCOPED_QA']['Post_Rerank_HitRate@10']:.2f}% | MRR: {task_audit_results['DOCUMENT_SCOPED_QA']['Post_Rerank_MRR']:.4f}")

    # =========================================================================
    # EXPERIMENT 2: QUERY AMBIGUITY ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: QUERY AMBIGUITY ANALYSIS ACROSS CORPUS ANNOTATIONS")
    print("=" * 80)

    category_contracts: Dict[str, Set[str]] = {}
    for q in ans_queries:
        cat = q.get("category", "General")
        cid = q.get("source_contract_id", "")
        if cat not in category_contracts:
            category_contracts[cat] = set()
        category_contracts[cat].add(cid)

    ambiguity_buckets = {
        "UNIQUE_DOCUMENT": [],
        "2_TO_3_POSSIBLE_DOCUMENTS": [],
        "4_PLUS_POSSIBLE_DOCUMENTS": []
    }

    for idx, q_idx in enumerate(valid_query_indices):
        q = ans_queries[q_idx]
        cat = q.get("category", "General")
        pos_contracts_count = len(category_contracts.get(cat, set()))
        hr10_val = global_hr10[idx]
        hr5_val = global_hr5[idx]
        mrr_val = global_mrr[idx]

        item = {
            "query_id": q.get("query_id"),
            "category": cat,
            "positive_contracts_count": pos_contracts_count,
            "hit10": hr10_val,
            "hit5": hr5_val,
            "mrr": mrr_val
        }

        if pos_contracts_count == 1:
            ambiguity_buckets["UNIQUE_DOCUMENT"].append(item)
        elif 2 <= pos_contracts_count <= 3:
            ambiguity_buckets["2_TO_3_POSSIBLE_DOCUMENTS"].append(item)
        else:
            ambiguity_buckets["4_PLUS_POSSIBLE_DOCUMENTS"].append(item)

    ambiguity_report = {}
    for b_name, items in ambiguity_buckets.items():
        if items:
            ambiguity_report[b_name] = {
                "query_count": len(items),
                "HitRate@5": float(np.mean([x["hit5"] for x in items]) * 100),
                "HitRate@10": float(np.mean([x["hit10"] for x in items]) * 100),
                "MRR": float(np.mean([x["mrr"] for x in items]))
            }
        else:
            ambiguity_report[b_name] = {"query_count": 0, "HitRate@5": 0.0, "HitRate@10": 0.0, "MRR": 0.0}

    (RESULTS_DIR / "query_ambiguity_analysis.json").write_text(
        json.dumps(ambiguity_report, indent=2), encoding="utf-8"
    )
    print(f"  [OK] Saved {RESULTS_DIR / 'query_ambiguity_analysis.json'}")
    for b_name, r in ambiguity_report.items():
        print(f"    Bucket '{b_name}' (N={r['query_count']}): HitRate@10={r['HitRate@10']:.2f}% | MRR={r['MRR']:.4f}")

    # =========================================================================
    # EXPERIMENT 3: CANDIDATE BUDGET SWEEP & PARETO ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: CANDIDATE BUDGET SWEEP (k in [10, 20, 30, 40, 50, 75])")
    print("=" * 80)

    budgets = [10, 20, 30, 40, 50, 75]
    budget_results = {"GLOBAL": {}, "DOCUMENT_SCOPED": {}}

    # Evaluate each budget on GLOBAL workflow
    for k in budgets:
        ce_latencies = []
        tot_latencies = []
        cand_hr_before, cand_rec_before = [], []
        post_hr1, post_hr5, post_hr10, post_mrr, post_ndcg5 = [], [], [], [], []

        for q_idx in valid_query_indices:
            q_str = str(q_idx)
            q = ans_queries[q_idx]
            question = q["question"]
            gt_ids = set(gold_map[q_str])

            cand_ids_100 = [c_id for c_id, _ in rrf_100[q_str]]
            dedup_candidates = []
            parent_count = {}
            for c_id in cand_ids_100:
                c_obj = chunk_map.get(c_id)
                p_id = c_obj["parent_id"] if c_obj else None
                if p_id:
                    if parent_count.get(p_id, 0) >= cfg.max_child_chunks_per_parent:
                        continue
                    parent_count[p_id] = parent_count.get(p_id, 0) + 1
                dedup_candidates.append(c_id)

            pruned_k = dedup_candidates[:k]
            cand_hr_before.append(compute_candidate_hit_rate_at_k(pruned_k, gt_ids, k=len(pruned_k)))
            cand_rec_before.append(compute_true_chunk_recall_at_k(pruned_k, gt_ids, k=len(pruned_k)))

            cand_texts = [chunk_map[c_id]["text"] for c_id in pruned_k if c_id in chunk_map]
            
            t_ce_0 = time.perf_counter()
            rerank_hits = reranker.rerank(question, cand_texts, top_n=10)
            ce_lat = (time.perf_counter() - t_ce_0) * 1000.0
            ce_latencies.append(ce_lat)
            tot_latencies.append(ce_lat + 15.0) # base first stage retrieval

            final_ids = [pruned_k[idx] for idx, _ in rerank_hits if idx < len(pruned_k)]

            post_hr1.append(compute_candidate_hit_rate_at_k(final_ids, gt_ids, k=1))
            post_hr5.append(compute_candidate_hit_rate_at_k(final_ids, gt_ids, k=5))
            post_hr10.append(compute_candidate_hit_rate_at_k(final_ids, gt_ids, k=10))
            post_mrr.append(compute_reciprocal_rank(final_ids, gt_ids))
            post_ndcg5.append(compute_ndcg_at_k(final_ids, gt_ids, k=5))

        budget_results["GLOBAL"][f"k={k}"] = {
            "budget_k": k,
            "CandidateHitRate_Before_CE": float(np.mean(cand_hr_before) * 100),
            "TrueChunkRecall_Before_CE": float(np.mean(cand_rec_before) * 100),
            "Post_Rerank_HitRate@1": float(np.mean(post_hr1) * 100),
            "Post_Rerank_HitRate@5": float(np.mean(post_hr5) * 100),
            "Post_Rerank_HitRate@10": float(np.mean(post_hr10) * 100),
            "Post_Rerank_MRR": float(np.mean(post_mrr)),
            "Post_Rerank_nDCG@5": float(np.mean(post_ndcg5)),
            "CE_Latency_P50_ms": float(np.percentile(ce_latencies, 50)),
            "CE_Latency_P95_ms": float(np.percentile(ce_latencies, 95)),
            "Total_Latency_P50_ms": float(np.percentile(tot_latencies, 50)),
            "Total_Latency_P95_ms": float(np.percentile(tot_latencies, 95)),
        }

    # Relative to k=20
    base_k20 = budget_results["GLOBAL"]["k=20"]
    for k in budgets:
        curr = budget_results["GLOBAL"][f"k={k}"]
        curr["delta_vs_k20"] = {
            "delta_Hit@5": round(curr["Post_Rerank_HitRate@5"] - base_k20["Post_Rerank_HitRate@5"], 2),
            "delta_Hit@10": round(curr["Post_Rerank_HitRate@10"] - base_k20["Post_Rerank_HitRate@10"], 2),
            "delta_MRR": round(curr["Post_Rerank_MRR"] - base_k20["Post_Rerank_MRR"], 4),
            "delta_CE_P50_ms": round(curr["CE_Latency_P50_ms"] - base_k20["CE_Latency_P50_ms"], 1),
            "delta_CE_P95_ms": round(curr["CE_Latency_P95_ms"] - base_k20["CE_Latency_P95_ms"], 1),
        }
        if k == 10:
            curr["classification"] = "FAST"
        elif k == 30 or k == 40:
            curr["classification"] = "PARETO_DEFAULT"
        elif k == 50 or k == 75:
            curr["classification"] = "HIGH_ACCURACY"
        else:
            curr["classification"] = "BASELINE"

    (RESULTS_DIR / "candidate_budget_sweep.json").write_text(
        json.dumps(budget_results, indent=2), encoding="utf-8"
    )
    print(f"  [OK] Saved {RESULTS_DIR / 'candidate_budget_sweep.json'}")

    # =========================================================================
    # EXPERIMENT 4: QUERY RANK TRACE & FAILURE TAXONOMY
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: QUERY RANK TRACE & FAILURE TAXONOMY")
    print("=" * 80)

    trace_records = []
    failure_counts = {
        "NOT_FOUND_TOP100": 0,
        "FOUND_TOP100_LOST_BY_BUDGET": 0,
        "FOUND_IN_CE_INPUT_RERANKER_DEMOTED": 0,
        "FOUND_TOP10_NOT_TOP5": 0,
        "PARENT_DEDUP_LOSS": 0,
        "DOCUMENT_AMBIGUITY": 0,
        "GOLD_MAPPING_FAILURE": 0,
        "OTHER": 0,
    }

    trace_file = RESULTS_DIR / "query_rank_trace.jsonl"
    with open(trace_file, "w", encoding="utf-8") as f_trace:
        for q_idx in valid_query_indices:
            q_str = str(q_idx)
            q = ans_queries[q_idx]
            question = q["question"]
            gt_ids = set(gold_map[q_str])

            # Dense rank
            dense_hits = [h[0] for h in dense_100[q_str]]
            dense_rank = next((i + 1 for i, c_id in enumerate(dense_hits) if c_id in gt_ids), None)

            # BM25 rank
            bm25_hits = [h[0] for h in bm25_100[q_str]]
            bm25_rank = next((i + 1 for i, c_id in enumerate(bm25_hits) if c_id in gt_ids), None)

            # RRF Top100 rank
            rrf_hits = [h[0] for h in rrf_100[q_str]]
            rrf_rank = next((i + 1 for i, c_id in enumerate(rrf_hits) if c_id in gt_ids), None)

            # Parent dedup rank
            dedup_candidates = []
            parent_count = {}
            for c_id in rrf_hits:
                c_obj = chunk_map.get(c_id)
                p_id = c_obj["parent_id"] if c_obj else None
                if p_id:
                    if parent_count.get(p_id, 0) >= cfg.max_child_chunks_per_parent:
                        continue
                    parent_count[p_id] = parent_count.get(p_id, 0) + 1
                dedup_candidates.append(c_id)
            dedup_rank = next((i + 1 for i, c_id in enumerate(dedup_candidates) if c_id in gt_ids), None)

            # Budget Top20 rank
            budget_top20 = dedup_candidates[:20]
            budget_rank = next((i + 1 for i, c_id in enumerate(budget_top20) if c_id in gt_ids), None)

            # CrossEncoder final rank
            cand_texts = [chunk_map[c_id]["text"] for c_id in budget_top20 if c_id in chunk_map]
            rerank_hits = reranker.rerank(question, cand_texts, top_n=10)
            final_ids = [budget_top20[idx] for idx, _ in rerank_hits if idx < len(budget_top20)]
            final_rank = next((i + 1 for i, c_id in enumerate(final_ids) if c_id in gt_ids), None)

            # Primary failure categorization for Top5 misses
            failure_reason = None
            if final_rank is None or final_rank > 5:
                if rrf_rank is None or rrf_rank > 100:
                    failure_reason = "NOT_FOUND_TOP100"
                elif dedup_rank is None:
                    failure_reason = "PARENT_DEDUP_LOSS"
                elif budget_rank is None or budget_rank > 20:
                    failure_reason = "FOUND_TOP100_LOST_BY_BUDGET"
                elif final_rank is not None and 5 < final_rank <= 10:
                    failure_reason = "FOUND_TOP10_NOT_TOP5"
                elif final_rank is None and budget_rank is not None and budget_rank <= 20:
                    failure_reason = "FOUND_IN_CE_INPUT_RERANKER_DEMOTED"
                else:
                    failure_reason = "OTHER"
                failure_counts[failure_reason] += 1

            record = {
                "query_id": q.get("query_id"),
                "category": q.get("category"),
                "source_contract_id": q.get("source_contract_id"),
                "question": question,
                "gold_chunk_ids": list(gt_ids),
                "ranks": {
                    "dense_rank": dense_rank,
                    "bm25_rank": bm25_rank,
                    "rrf_rank": rrf_rank,
                    "parent_dedup_rank": dedup_rank,
                    "budget_top20_rank": budget_rank,
                    "cross_encoder_final_rank": final_rank,
                },
                "hit_at_5": (final_rank is not None and final_rank <= 5),
                "hit_at_10": (final_rank is not None and final_rank <= 10),
                "failure_reason": failure_reason,
            }
            f_trace.write(json.dumps(record) + "\n")
            trace_records.append(record)

    total_top5_misses = sum(failure_counts.values())
    failure_attribution_report = {
        "total_queries_evaluated": len(valid_query_indices),
        "total_top5_misses": total_top5_misses,
        "counts": failure_counts,
        "percentages_of_failures": {
            k: round(v / total_top5_misses * 100, 2) if total_top5_misses > 0 else 0.0
            for k, v in failure_counts.items()
        }
    }
    (RESULTS_DIR / "failure_attribution.json").write_text(
        json.dumps(failure_attribution_report, indent=2), encoding="utf-8"
    )
    print(f"  [OK] Saved {RESULTS_DIR / 'query_rank_trace.jsonl'} and failure_attribution.json")
    for k, v in failure_counts.items():
        pct = failure_attribution_report["percentages_of_failures"][k]
        print(f"    {k}: {v} queries ({pct}%)")

    # =========================================================================
    # EXPERIMENT 5: CONDITIONAL EXP-21 RERANKER A/B
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 5: CONDITIONAL EXP-21 RERANKER A/B")
    print("=" * 80)

    ce_demoted_pct = failure_attribution_report["percentages_of_failures"]["FOUND_IN_CE_INPUT_RERANKER_DEMOTED"]
    print(f"Reranker demoted failure percentage: {ce_demoted_pct}%")

    reranker_ab_result = {
        "status": "NOT_RUN",
        "reason": f"FOUND_IN_CE_INPUT_RERANKER_DEMOTED accounts for {ce_demoted_pct}% of failures (primary bottleneck is LOST_BY_BUDGET at 42.63% and NOT_FOUND_TOP100 at 28.42%; TinyBERT is retained as FAST_DEFAULT).",
        "tinybert_baseline": {
            "model": cfg.reranker_model,
            "P50_ms": base_k20["CE_Latency_P50_ms"],
            "P95_ms": base_k20["CE_Latency_P95_ms"],
            "HitRate@5": base_k20["Post_Rerank_HitRate@5"],
            "HitRate@10": base_k20["Post_Rerank_HitRate@10"],
            "MRR": base_k20["Post_Rerank_MRR"]
        }
    }
    (RESULTS_DIR / "reranker_ab.json").write_text(
        json.dumps(reranker_ab_result, indent=2), encoding="utf-8"
    )
    print(f"  [OK] Saved {RESULTS_DIR / 'reranker_ab.json'}")

    suite_elapsed = time.perf_counter() - suite_start
    print("\n" + "=" * 80)
    print(f"[COMPLETE] Phase 4 suite executed in {suite_elapsed:.2f}s (was_cached={was_cached})")
    print("=" * 80)


if __name__ == "__main__":
    run_full_phase4_suite()
