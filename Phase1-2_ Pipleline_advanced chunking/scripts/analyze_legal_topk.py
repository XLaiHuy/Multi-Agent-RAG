"""
Evaluator for Legal RAG Benchmark across top_k=5, 10, 15, 20
to analyze why Legal RAG requires top_k >= 15 due to boilerplate legal contract overlap.
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


def analyze_topk_legal():
    print("========================================================")
    print("⚖️ ANALYZING LEGAL RAG BENCHMARK HIT RATE ACROSS TOP-K (k=5..20)")
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

    child_corpus = []
    child_to_parent_map = {}
    for doc_item in raw_corpus:
        doc = RawDocument(doc_id=doc_item["doc_id"], text=doc_item["text"], source=f"{doc_item['doc_id']}.txt")
        child_chunks = parent_child_chunker.chunk_document(doc)
        for cc in child_chunks:
            child_corpus.append({"chunk_id": cc.chunk_id, "text": cc.text, "doc_id": cc.doc_id})
            child_to_parent_map[cc.chunk_id] = cc.doc_id

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

    dense_sim_parent = np.dot(q_embs, parent_embs.T)
    dense_sim_child = np.dot(q_embs, child_embs.T)
    dense_sim_hyde_child = np.dot(q_hyde_embs, child_embs.T)

    N = len(eval_queries)

    print(f"Total legal queries evaluated: {N}\n")

    for k in [5, 10, 15, 20]:
        # Hybrid RRF on Parent-Child
        hits = 0
        rrs = []
        for i in range(N):
            expected = eval_queries[i]["expected_doc_id"]
            q_toks = q_hyde_texts[i].lower().split()
            top_b = np.argsort(-bm25_child.get_scores(q_toks))[:k * 3]
            top_d = np.argsort(-dense_sim_hyde_child[i])[:k * 3]

            rrf = {}
            for r, idx in enumerate(top_b):
                pid = child_to_parent_map[child_ids[idx]]
                rrf[pid] = rrf.get(pid, 0.0) + (1.0 / (RRF_K + r + 1))
            for r, idx in enumerate(top_d):
                pid = child_to_parent_map[child_ids[idx]]
                rrf[pid] = rrf.get(pid, 0.0) + (1.0 / (RRF_K + r + 1))

            ranked = [d[0] for d in sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:k]]
            if expected in ranked:
                hits += 1
                rrs.append(1.0 / (ranked.index(expected) + 1))
            else:
                rrs.append(0.0)

        hit_rate = (hits / N) * 100
        mrr = float(np.mean(rrs))
        print(f"  • Parent-Child Hybrid RRF (top_k={k:2d}) -> Hit Rate: {hit_rate:.1f}% ({hits}/{N}) | MRR: {mrr:.4f}")


if __name__ == "__main__":
    analyze_topk_legal()
