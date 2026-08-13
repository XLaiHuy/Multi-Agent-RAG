"""
High-throughput vectorized benchmark runner for:
1. FULL UIT-ViQuAD 2.0 (N=3,814 samples)
2. Stanford SQuAD 2.0 (N=2,000 samples)
Testing BM25, Dense Vector, and Hybrid RRF across multiple top_k values.
"""
import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding="utf-8")

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RRF_K = 60


def evaluate_dataset(dataset_name: str, corpus_path: Path, eval_path: Path, embedder: SentenceTransformer) -> Dict[str, Any]:
    print(f"\n========================================================")
    print(f"📊 BENCHMARKING: {dataset_name}")
    print(f"========================================================")
    
    corpus_data = json.loads(corpus_path.read_text(encoding="utf-8"))
    eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
    
    factual_eval = [item for item in eval_data if not item.get("is_unanswerable") and item.get("expected_chunk_ids")]
    print(f"  • Total queries: {len(eval_data)}")
    print(f"  • Factual queries with ground truth: {len(factual_eval)}")
    print(f"  • Corpus context passages: {len(corpus_data)}")

    # 1. Build BM25 Index
    print("  • Indexing BM25...")
    tokenized_corpus = [c["text"].lower().split() for c in corpus_data]
    bm25 = BM25Okapi(tokenized_corpus)
    doc_ids = [c["chunk_id"] for c in corpus_data]

    # 2. Encode Corpus & Queries
    print("  • Encoding Corpus embeddings...")
    corpus_texts = [c["text"] for c in corpus_data]
    doc_embs = embedder.encode(corpus_texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)

    print("  • Encoding Query embeddings...")
    query_texts = [item["question"] for item in factual_eval]
    query_embs = embedder.encode(query_texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)

    # 3. Dense Similarity Matrix (N_queries x N_docs)
    print("  • Computing Dense Cosine Similarity Matrix...")
    t0_dense = time.perf_counter()
    dense_sim_matrix = np.dot(query_embs, doc_embs.T)
    dense_time_total = time.perf_counter() - t0_dense

    # 4. Sparse BM25 Scores Matrix
    print("  • Computing BM25 Sparse Score Matrix...")
    t0_bm25 = time.perf_counter()
    bm25_scores_list = []
    for q_text in query_texts:
        q_tokens = q_text.lower().split()
        scores = bm25.get_scores(q_tokens)
        bm25_scores_list.append(scores)
    bm25_scores_matrix = np.array(bm25_scores_list)
    bm25_time_total = time.perf_counter() - t0_bm25

    results = {}
    N = len(factual_eval)

    configs = [
        ("BM25 Sparse-only (k=5)", "bm25", 5),
        ("Dense Vector-only (k=5)", "dense", 5),
        ("Hybrid RRF (k=5)", "hybrid", 5),
        ("Hybrid RRF (k=10)", "hybrid", 10),
        ("Hybrid RRF (k=20)", "hybrid", 20),
    ]

    for label, mode, top_k in configs:
        t0 = time.perf_counter()
        hits = 0
        reciprocal_ranks = []
        latencies = []

        for i, item in enumerate(factual_eval):
            expected_set = set(item["expected_chunk_ids"])
            q_start = time.perf_counter()

            if mode == "bm25":
                top_indices = np.argsort(-bm25_scores_matrix[i])[:top_k]
                ranked_docs = [doc_ids[idx] for idx in top_indices]
            elif mode == "dense":
                top_indices = np.argsort(-dense_sim_matrix[i])[:top_k]
                ranked_docs = [doc_ids[idx] for idx in top_indices]
            else: # Hybrid RRF
                top_bm25_idx = np.argsort(-bm25_scores_matrix[i])[:top_k * 3]
                top_dense_idx = np.argsort(-dense_sim_matrix[i])[:top_k * 3]

                rrf_scores = {}
                for rank, idx in enumerate(top_bm25_idx):
                    did = doc_ids[idx]
                    rrf_scores[did] = rrf_scores.get(did, 0.0) + (1.0 / (RRF_K + rank + 1))
                for rank, idx in enumerate(top_dense_idx):
                    did = doc_ids[idx]
                    rrf_scores[did] = rrf_scores.get(did, 0.0) + (1.0 / (RRF_K + rank + 1))

                sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
                ranked_docs = [d[0] for d in sorted_docs]

            q_dur = (time.perf_counter() - q_start) * 1000
            latencies.append(q_dur)

            # Metrics
            hit_found = False
            rr = 0.0
            for rank_idx, doc_id in enumerate(ranked_docs):
                if doc_id in expected_set:
                    if not hit_found:
                        hits += 1
                        rr = 1.0 / (rank_idx + 1)
                        hit_found = True
            reciprocal_ranks.append(rr)

        total_time = time.perf_counter() - t0
        hit_rate = (hits / N) * 100
        mrr = float(np.mean(reciprocal_ranks))
        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        qps = N / total_time if total_time > 0 else 0

        results[label] = {
            "mode": mode,
            "top_k": top_k,
            "total_queries": N,
            "hits": hits,
            "hit_rate_pct": round(hit_rate, 2),
            "mrr": round(mrr, 4),
            "latency_p50_ms": round(p50, 2),
            "latency_p95_ms": round(p95, 2),
            "throughput_qps": round(qps, 1),
        }

        print(f"  [{label}] -> Hit Rate: {hit_rate:.1f}% | MRR: {mrr:.4f} | P50: {p50:.2f}ms | QPS: {qps:.1f}")

    return {
        "dataset": dataset_name,
        "total_evaluated": N,
        "results": results,
    }


def main():
    print("Loading Multilingual SentenceTransformer Embedding Model...")
    embedder = SentenceTransformer(MODEL_NAME)
    print("Model loaded successfully.")

    # 1. Evaluate Full UIT-ViQuAD 2.0 (N=3,814)
    viquad_res = evaluate_dataset(
        dataset_name="UIT-ViQuAD 2.0 (Full Validation, N=3,814)",
        corpus_path=Path("data/evaluation/viquad_full/corpus_full.json"),
        eval_path=Path("data/evaluation/viquad_full/viquad_full_eval.json"),
        embedder=embedder,
    )

    # 2. Evaluate Stanford SQuAD 2.0 (N=2,000)
    squad_res = evaluate_dataset(
        dataset_name="Stanford SQuAD 2.0 (N=2,000 English Benchmark)",
        corpus_path=Path("data/evaluation/squad_2k/corpus_2k.json"),
        eval_path=Path("data/evaluation/squad_2k/squad_2k_eval.json"),
        embedder=embedder,
    )

    # Save overall dual benchmark report
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "viquad_full": viquad_res,
        "squad_2k": squad_res,
    }
    out_file = Path("data/evaluation/dual_full_benchmark_results.json")
    out_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[Done] Saved dual full benchmark results to {out_file}!")


if __name__ == "__main__":
    main()
