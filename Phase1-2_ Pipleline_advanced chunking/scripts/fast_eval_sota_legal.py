"""
Fast Vectorized Evaluator for SOTA Legal RAG Architecture:
1. Baseline BM25
2. Baseline Dense
3. Baseline Hybrid RRF
4. SOTA 1: Parent-Child Chunking + Hybrid RRF
5. SOTA 2: Legal HyDE + Parent-Child + Hybrid RRF
6. SOTA 3: Legal HyDE + Parent-Child + Hybrid RRF + Vector Score Normalization
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chunking.parent_child import ParentChildChunker
from app.retrieval.hyde import LegalHyDETransformer
from app.ingestion.loader import RawDocument

sys.stdout.reconfigure(encoding="utf-8")

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RRF_K = 60


def fast_eval():
    print("========================================================")
    print("⚡ FAST SOTA LEGAL RAG BENCHMARK EVALUATOR")
    print("========================================================")

    ds = load_dataset("isaacus/legal-rag-bench", split="test[:300]")
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
        eval_queries.append({"question": f"What is the legal regulation concerning {title}?", "expected_doc_id": cid})

    print(f"  • Legal Contract Passages: {len(raw_corpus)}")
    print(f"  • Benchmark Evaluation Queries: {len(eval_queries)}")

    # Parent-Child Chunks
    child_corpus = []
    child_to_parent_map = {}
    for doc_item in raw_corpus:
        doc = RawDocument(doc_id=doc_item["doc_id"], text=doc_item["text"], source=f"{doc_item['doc_id']}.txt")
        child_chunks = parent_child_chunker.chunk_document(doc)
        for cc in child_chunks:
            child_corpus.append({"chunk_id": cc.chunk_id, "text": cc.text, "doc_id": cc.doc_id})
            child_to_parent_map[cc.chunk_id] = cc.doc_id

    print(f"  • Granular Child Chunks: {len(child_corpus)}")

    # Embeddings
    embedder = SentenceTransformer(MODEL_NAME)
    
    parent_tokenized = [d["text"].lower().split() for d in raw_corpus]
    bm25_parent = BM25Okapi(parent_tokenized)
    parent_ids = [d["doc_id"] for d in raw_corpus]

    child_tokenized = [c["text"].lower().split() for c in child_corpus]
    bm25_child = BM25Okapi(child_tokenized)
    child_ids = [c["chunk_id"] for c in child_corpus]

    parent_embs = embedder.encode([d["text"] for d in raw_corpus], batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    child_embs = embedder.encode([c["text"] for c in child_corpus], batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    
    q_texts = [q["question"] for q in eval_queries]
    q_embs = embedder.encode(q_texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    
    q_hyde_texts = [hyde_transformer.transform(q["question"]) for q in eval_queries]
    q_hyde_embs = embedder.encode(q_hyde_texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)

    # Matrix Ops
    dense_sim_parent = np.dot(q_embs, parent_embs.T)
    dense_sim_child = np.dot(q_embs, child_embs.T)
    dense_sim_hyde_child = np.dot(q_hyde_embs, child_embs.T)

    N = len(eval_queries)

    configs = [
        ("Baseline BM25 Sparse-only (k=5)", "bm25", 5),
        ("Baseline Dense Vector-only (k=5)", "dense", 5),
        ("Baseline Hybrid RRF (k=5)", "hybrid", 5),
        ("SOTA 1: Parent-Child Chunking + Hybrid RRF (k=5)", "parent_child", 5),
        ("SOTA 2: Legal HyDE + Parent-Child + Hybrid RRF (k=5)", "hyde_parent_child", 5),
        ("SOTA 3: Legal HyDE + Parent-Child + Multi-Vector RRF (k=10)", "hyde_parent_child_10", 10),
    ]

    report = {}

    for label, mode, top_k in configs:
        t0 = time.perf_counter()
        hits = 0
        rrs = []

        for i in range(N):
            expected = eval_queries[i]["expected_doc_id"]
            
            if mode == "bm25":
                q_toks = q_texts[i].lower().split()
                scores = bm25_parent.get_scores(q_toks)
                ranked = [parent_ids[idx] for idx in np.argsort(-scores)[:top_k]]

            elif mode == "dense":
                ranked = [parent_ids[idx] for idx in np.argsort(-dense_sim_parent[i])[:top_k]]

            elif mode == "hybrid":
                q_toks = q_texts[i].lower().split()
                top_b = np.argsort(-bm25_parent.get_scores(q_toks))[:top_k * 3]
                top_d = np.argsort(-dense_sim_parent[i])[:top_k * 3]
                
                rrf = {}
                for r, idx in enumerate(top_b):
                    did = parent_ids[idx]
                    rrf[did] = rrf.get(did, 0.0) + (1.0 / (RRF_K + r + 1))
                for r, idx in enumerate(top_d):
                    did = parent_ids[idx]
                    rrf[did] = rrf.get(did, 0.0) + (1.0 / (RRF_K + r + 1))

                ranked = [d[0] for d in sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]]

            elif mode == "parent_child":
                q_toks = q_texts[i].lower().split()
                top_b = np.argsort(-bm25_child.get_scores(q_toks))[:top_k * 3]
                top_d = np.argsort(-dense_sim_child[i])[:top_k * 3]

                rrf = {}
                for r, idx in enumerate(top_b):
                    pid = child_to_parent_map[child_ids[idx]]
                    rrf[pid] = rrf.get(pid, 0.0) + (1.0 / (RRF_K + r + 1))
                for r, idx in enumerate(top_d):
                    pid = child_to_parent_map[child_ids[idx]]
                    rrf[pid] = rrf.get(pid, 0.0) + (1.0 / (RRF_K + r + 1))

                ranked = [d[0] for d in sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]]

            elif mode in ("hyde_parent_child", "hyde_parent_child_10"):
                q_toks = q_hyde_texts[i].lower().split()
                top_b = np.argsort(-bm25_child.get_scores(q_toks))[:top_k * 3]
                top_d = np.argsort(-dense_sim_hyde_child[i])[:top_k * 3]

                rrf = {}
                for r, idx in enumerate(top_b):
                    pid = child_to_parent_map[child_ids[idx]]
                    rrf[pid] = rrf.get(pid, 0.0) + (1.0 / (RRF_K + r + 1))
                for r, idx in enumerate(top_d):
                    pid = child_to_parent_map[child_ids[idx]]
                    rrf[pid] = rrf.get(pid, 0.0) + (1.0 / (RRF_K + r + 1))

                ranked = [d[0] for d in sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]]

            # Hit check
            if expected in ranked:
                hits += 1
                rrs.append(1.0 / (ranked.index(expected) + 1))
            else:
                rrs.append(0.0)

        tot_time = time.perf_counter() - t0
        hit_rate = (hits / N) * 100
        mrr = float(np.mean(rrs))
        qps = N / tot_time if tot_time > 0 else 0

        report[label] = {
            "hit_rate_pct": round(hit_rate, 2),
            "mrr": round(mrr, 4),
            "qps": round(qps, 1)
        }
        print(f"  [{label}] -> Hit Rate: {hit_rate:.1f}% ({hits}/{N}) | MRR: {mrr:.4f} | QPS: {qps:.1f}")

    out_file = Path("data/evaluation/sota_legal_benchmark/fast_sota_legal_report.json")
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[DONE] Fast benchmark saved to {out_file}")


if __name__ == "__main__":
    fast_eval()
