"""
Benchmark SOTA Legal RAG Engine on Public Legal RAG Benchmark (isaacus/legal-rag-bench):
Evaluates 150 commercial legal contract passages across:
1. Baseline BM25 Sparse-only
2. Baseline Dense Vector-only
3. Baseline Hybrid RRF
4. SOTA 1: Hierarchical Parent-Child + Hybrid RRF
5. SOTA 2: Legal HyDE + Hierarchical Parent-Child + Hybrid RRF
6. SOTA 3: Legal HyDE + Hierarchical Parent-Child + Hybrid RRF + Cross-Encoder Reranker
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chunking.parent_child import ParentChildChunker
from app.retrieval.hyde import LegalHyDETransformer
from app.ingestion.loader import RawDocument

sys.stdout.reconfigure(encoding="utf-8")

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANKER_NAME = "BAAI/bge-reranker-base"
RRF_K = 60
DATA_DIR = Path("data/evaluation/sota_legal_benchmark")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def run_sota_legal_benchmark():
    print("========================================================")
    print("⚖️ BENCHMARKING SOTA LEGAL RAG ENGINE ON LEGAL RAG BENCHMARK")
    print("========================================================")

    # 1. Load Dataset (150 samples for instant CPU execution)
    print("\n[1/5] Loading public Legal RAG Benchmark dataset (isaacus/legal-rag-bench)...")
    ds = load_dataset("isaacus/legal-rag-bench", split="test[:150]")
    
    parent_child_chunker = ParentChildChunker(parent_size=1500, child_size=300, child_overlap=50)
    hyde_transformer = LegalHyDETransformer()

    raw_corpus = []
    eval_queries = []

    for i, item in enumerate(ds):
        txt = (item.get("text") or "").strip()
        title = (item.get("title") or "").strip()
        cid = item.get("id") or f"leg_{i:04d}"
        if len(txt) < 30:
            continue
        raw_corpus.append({"doc_id": cid, "text": txt, "title": title})
        eval_queries.append({
            "question": f"What is the regulation concerning {title}?",
            "expected_doc_id": cid
        })

    print(f"  -> Total legal contract passages: {len(raw_corpus)}")
    print(f"  -> Total evaluation queries: {len(eval_queries)}")

    # 2. Build Standard Flat Chunks vs Parent-Child Chunks
    print("\n[2/5] Building Parent-Child Hierarchical Chunks...")
    child_corpus = []
    child_to_parent_map = {}

    for doc_item in raw_corpus:
        doc = RawDocument(doc_id=doc_item["doc_id"], text=doc_item["text"], source=f"{doc_item['doc_id']}.txt")
        child_chunks = parent_child_chunker.chunk_document(doc)
        for cc in child_chunks:
            child_corpus.append({"chunk_id": cc.chunk_id, "text": cc.text, "doc_id": cc.doc_id})
            child_to_parent_map[cc.chunk_id] = cc.doc_id

    print(f"  -> Generated {len(child_corpus)} granular child chunks from {len(raw_corpus)} parent sections.")

    # 3. Index & Encode Embeddings
    print("\n[3/5] Initializing Vector Embeddings & BM25 Indices...")
    embedder = SentenceTransformer(MODEL_NAME)
    
    # BM25 on child corpus
    child_tokenized = [c["text"].lower().split() for c in child_corpus]
    bm25_child = BM25Okapi(child_tokenized)
    child_ids = [c["chunk_id"] for c in child_corpus]

    # BM25 on raw parent corpus
    parent_tokenized = [d["text"].lower().split() for d in raw_corpus]
    bm25_parent = BM25Okapi(parent_tokenized)
    parent_ids = [d["doc_id"] for d in raw_corpus]

    print("  • Encoding Parent Vector Embeddings...")
    parent_texts = [d["text"] for d in raw_corpus]
    parent_embs = embedder.encode(parent_texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)

    print("  • Encoding Child Vector Embeddings...")
    child_texts = [c["text"] for c in child_corpus]
    child_embs = embedder.encode(child_texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)

    print("  • Encoding Query Embeddings...")
    q_texts = [q["question"] for q in eval_queries]
    q_embs = embedder.encode(q_texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)

    print("  • Encoding HyDE Transformed Query Embeddings...")
    q_hyde_texts = [hyde_transformer.transform(q["question"]) for q in eval_queries]
    q_hyde_embs = embedder.encode(q_hyde_texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)

    # Cross-Encoder Reranker
    print("\n[4/5] Loading Cross-Encoder Reranker model...")
    cross_encoder = CrossEncoder(RERANKER_NAME)

    # 4. Run Benchmark Configurations
    print("\n[5/5] Running SOTA Evaluation Pipelines...")
    N = len(eval_queries)
    report = {}

    pipelines = [
        ("Baseline BM25 Sparse (k=5)", "bm25_base", 5),
        ("Baseline Dense Vector (k=5)", "dense_base", 5),
        ("Baseline Hybrid RRF (k=5)", "hybrid_base", 5),
        ("SOTA 1: Parent-Child + Hybrid RRF (k=5)", "parent_child", 5),
        ("SOTA 2: Legal HyDE + Parent-Child + Hybrid RRF (k=5)", "hyde_parent_child", 5),
        ("SOTA 3: Legal HyDE + Parent-Child + Hybrid RRF + Cross-Encoder Reranker (k=5)", "sota_full", 5),
    ]

    for label, mode, top_k in pipelines:
        t0 = time.perf_counter()
        hits = 0
        reciprocal_ranks = []
        latencies = []

        for i, q_item in enumerate(eval_queries):
            expected_doc = q_item["expected_doc_id"]
            q_str = q_item["question"]
            q_start = time.perf_counter()

            if mode == "bm25_base":
                q_toks = q_str.lower().split()
                scores = bm25_parent.get_scores(q_toks)
                top_idx = np.argsort(-scores)[:top_k]
                ranked_docs = [parent_ids[idx] for idx in top_idx]

            elif mode == "dense_base":
                sims = np.dot(q_embs[i], parent_embs.T)
                top_idx = np.argsort(-sims)[:top_k]
                ranked_docs = [parent_ids[idx] for idx in top_idx]

            elif mode == "hybrid_base":
                q_toks = q_str.lower().split()
                scores_bm25 = bm25_parent.get_scores(q_toks)
                top_bm25 = np.argsort(-scores_bm25)[:top_k * 3]

                sims = np.dot(q_embs[i], parent_embs.T)
                top_dense = np.argsort(-sims)[:top_k * 3]

                rrf = {}
                for rank, idx in enumerate(top_bm25):
                    did = parent_ids[idx]
                    rrf[did] = rrf.get(did, 0.0) + (1.0 / (RRF_K + rank + 1))
                for rank, idx in enumerate(top_dense):
                    did = parent_ids[idx]
                    rrf[did] = rrf.get(did, 0.0) + (1.0 / (RRF_K + rank + 1))

                sorted_rrf = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]
                ranked_docs = [d[0] for d in sorted_rrf]

            elif mode == "parent_child":
                q_toks = q_str.lower().split()
                scores_bm25 = bm25_child.get_scores(q_toks)
                top_bm25 = np.argsort(-scores_bm25)[:top_k * 3]

                sims = np.dot(q_embs[i], child_embs.T)
                top_dense = np.argsort(-sims)[:top_k * 3]

                rrf = {}
                for rank, idx in enumerate(top_bm25):
                    cid = child_ids[idx]
                    pid = child_to_parent_map[cid]
                    rrf[pid] = rrf.get(pid, 0.0) + (1.0 / (RRF_K + rank + 1))
                for rank, idx in enumerate(top_dense):
                    cid = child_ids[idx]
                    pid = child_to_parent_map[cid]
                    rrf[pid] = rrf.get(pid, 0.0) + (1.0 / (RRF_K + rank + 1))

                sorted_rrf = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]
                ranked_docs = [d[0] for d in sorted_rrf]

            elif mode == "hyde_parent_child":
                q_toks = q_hyde_texts[i].lower().split()
                scores_bm25 = bm25_child.get_scores(q_toks)
                top_bm25 = np.argsort(-scores_bm25)[:top_k * 3]

                sims = np.dot(q_hyde_embs[i], child_embs.T)
                top_dense = np.argsort(-sims)[:top_k * 3]

                rrf = {}
                for rank, idx in enumerate(top_bm25):
                    cid = child_ids[idx]
                    pid = child_to_parent_map[cid]
                    rrf[pid] = rrf.get(pid, 0.0) + (1.0 / (RRF_K + rank + 1))
                for rank, idx in enumerate(top_dense):
                    cid = child_ids[idx]
                    pid = child_to_parent_map[cid]
                    rrf[pid] = rrf.get(pid, 0.0) + (1.0 / (RRF_K + rank + 1))

                sorted_rrf = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]
                ranked_docs = [d[0] for d in sorted_rrf]

            elif mode == "sota_full":
                # Step A: HyDE + Parent-Child RRF top 8 candidate fetch
                q_toks = q_hyde_texts[i].lower().split()
                scores_bm25 = bm25_child.get_scores(q_toks)
                top_bm25 = np.argsort(-scores_bm25)[:24]

                sims = np.dot(q_hyde_embs[i], child_embs.T)
                top_dense = np.argsort(-sims)[:24]

                rrf = {}
                for rank, idx in enumerate(top_bm25):
                    cid = child_ids[idx]
                    pid = child_to_parent_map[cid]
                    rrf[pid] = rrf.get(pid, 0.0) + (1.0 / (RRF_K + rank + 1))
                for rank, idx in enumerate(top_dense):
                    cid = child_ids[idx]
                    pid = child_to_parent_map[cid]
                    rrf[pid] = rrf.get(pid, 0.0) + (1.0 / (RRF_K + rank + 1))

                candidate_parents = [d[0] for d in sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:8]]
                
                # Step B: Fast Cross-Encoder Reranking
                candidate_pairs = []
                for pid in candidate_parents:
                    p_idx = parent_ids.index(pid)
                    candidate_pairs.append((q_str, parent_texts[p_idx]))

                rerank_scores = cross_encoder.predict(candidate_pairs)
                reranked_idx = np.argsort(-rerank_scores)[:top_k]
                ranked_docs = [candidate_parents[idx] for idx in reranked_idx]

            q_dur = (time.perf_counter() - q_start) * 1000
            latencies.append(q_dur)

            # Hit Rate & MRR
            hit_found = False
            rr = 0.0
            for rank_idx, did in enumerate(ranked_docs):
                if did == expected_doc:
                    if not hit_found:
                        hits += 1
                        rr = 1.0 / (rank_idx + 1)
                        hit_found = True
            reciprocal_ranks.append(rr)

        tot_dur = time.perf_counter() - t0
        hit_rate = (hits / N) * 100
        mrr = float(np.mean(reciprocal_ranks))
        p50 = float(np.percentile(latencies, 50))
        qps = N / tot_dur if tot_dur > 0 else 0

        report[label] = {
            "hit_rate_pct": round(hit_rate, 2),
            "mrr": round(mrr, 4),
            "latency_p50_ms": round(p50, 2),
            "qps": round(qps, 1)
        }
        print(f"  [{label}] -> Hit Rate: {hit_rate:.1f}% ({hits}/{N}) | MRR: {mrr:.4f} | QPS: {qps:.1f}")

    # Save benchmark report
    out_file = DATA_DIR / "sota_legal_benchmark_report.json"
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[DONE] Saved SOTA Legal RAG report to {out_file}")


if __name__ == "__main__":
    run_sota_legal_benchmark()
