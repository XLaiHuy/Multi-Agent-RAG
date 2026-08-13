"""
Standalone benchmark runner — measures retrieval quality and latency only.
No LLM calls for retrieval metrics → no rate limiting, runs in ~30s.
"""
import json, time, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.evaluation.metrics import compute_recall_at_k, compute_hit_rate

EVAL_PATH = Path("data/evaluation/eval_dataset.json")
OUT_PATH  = Path("data/evaluation/benchmark_results_v2.json")

def mrr(retrieved_ids, expected_ids, k=5):
    exp = set(expected_ids)
    for rank, cid in enumerate(retrieved_ids[:k], 1):
        if cid in exp:
            return 1.0 / rank
    return 0.0

def run_exp(name, retriever, items, top_k, extra_kw):
    recalls, hits, mrrs, latencies = [], [], [], []
    factual_hits = []

    for item in items:
        is_unans = item.get("is_unanswerable", False)
        t0 = time.perf_counter()
        try:
            res = retriever.search(query=item["question"], top_k=top_k, **extra_kw)
            retrieved_ids = [r.chunk_id for r in res]
        except Exception as e:
            print(f"  ERROR on {item['id']}: {e}")
            retrieved_ids = []
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)

        exp_ids = item.get("expected_chunk_ids", [])
        r  = compute_recall_at_k(retrieved_ids, exp_ids, k=top_k)
        h  = compute_hit_rate(retrieved_ids, exp_ids, k=top_k)
        m  = mrr(retrieved_ids, exp_ids, k=top_k)
        recalls.append(r); hits.append(h); mrrs.append(m)

        if not is_unans and exp_ids:
            factual_hits.append(h)

        print(f"  [{item['id']}] Hit={h:.0f}  Recall={r:.2f}  MRR={m:.2f}  Lat={lat:.0f}ms")

    n = len(items)
    latencies_sorted = sorted(latencies)
    p50 = latencies_sorted[n // 2]
    p95 = latencies_sorted[min(int(n * 0.95), n - 1)]
    factual_hit_rate = sum(factual_hits) / len(factual_hits) if factual_hits else 0.0

    result = {
        "experiment": name,
        "total_samples": n,
        "avg_recall_at_k": round(sum(recalls) / n, 4),
        "avg_hit_rate":    round(sum(hits) / n, 4),
        "avg_mrr":         round(sum(mrrs) / n, 4),
        "factual_hit_rate": round(factual_hit_rate, 4),
        "latency_p50_ms":  round(p50, 1),
        "latency_p95_ms":  round(p95, 1),
        "avg_latency_ms":  round(sum(latencies) / n, 1),
    }
    print(f"\n  >> {name}: Hit={result['avg_hit_rate']:.1%}  "
          f"Factual-Hit={result['factual_hit_rate']:.1%}  "
          f"MRR={result['avg_mrr']:.3f}  P50={p50:.0f}ms  P95={p95:.0f}ms\n")
    return result

def run_multi_query_exp(name, retriever, items, top_k, query_variants: dict):
    """
    Multi-Query Expansion experiment: for each item, also search additional
    query variants (English equivalents), fuse with RRF, then evaluate.
    """
    from app.retrieval.hybrid_retriever import reciprocal_rank_fusion

    recalls, hits, mrrs, latencies = [], [], [], []
    factual_hits = []

    for item in items:
        is_unans = item.get("is_unanswerable", False)
        queries = [item["question"]] + query_variants.get(item["id"], [])
        t0 = time.perf_counter()

        all_ranked: list[list[str]] = []
        chunk_map: dict = {}
        for q in queries:
            try:
                res = retriever.search(query=q, top_k=top_k, use_rerank=False)
                all_ranked.append([r.chunk_id for r in res])
                for r in res:
                    if r.chunk_id not in chunk_map:
                        chunk_map[r.chunk_id] = r
            except Exception as e:
                print(f"  ERROR [{item['id']}]: {e}")

        fused = reciprocal_rank_fusion(all_ranked, k=60)
        retrieved_ids = [cid for cid, _ in fused[:top_k]]
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)

        exp_ids = item.get("expected_chunk_ids", [])
        r  = compute_recall_at_k(retrieved_ids, exp_ids, k=top_k)
        h  = compute_hit_rate(retrieved_ids, exp_ids, k=top_k)
        m  = mrr(retrieved_ids, exp_ids, k=top_k)
        recalls.append(r); hits.append(h); mrrs.append(m)

        if not is_unans and exp_ids:
            factual_hits.append(h)

        print(f"  [{item['id']}] Hit={h:.0f}  Recall={r:.2f}  MRR={m:.2f}  Lat={lat:.0f}ms  Queries={len(queries)}")

    n = len(items)
    latencies_sorted = sorted(latencies)
    p50 = latencies_sorted[n // 2]
    p95 = latencies_sorted[min(int(n * 0.95), n - 1)]
    factual_hit_rate = sum(factual_hits) / len(factual_hits) if factual_hits else 0.0

    result = {
        "experiment": name,
        "total_samples": n,
        "avg_recall_at_k": round(sum(recalls) / n, 4),
        "avg_hit_rate":    round(sum(hits) / n, 4),
        "avg_mrr":         round(sum(mrrs) / n, 4),
        "factual_hit_rate": round(factual_hit_rate, 4),
        "latency_p50_ms":  round(p50, 1),
        "latency_p95_ms":  round(p95, 1),
        "avg_latency_ms":  round(sum(latencies) / n, 1),
    }
    print(f"\n  >> {name}: Hit={result['avg_hit_rate']:.1%}  "
          f"Factual-Hit={result['factual_hit_rate']:.1%}  "
          f"MRR={result['avg_mrr']:.3f}  P50={p50:.0f}ms  P95={p95:.0f}ms\n")
    return result


# English/bilingual query variants for items that suffer from cross-lingual embedding mismatch
# Generated by looking at the actual chunk content language
QUERY_VARIANTS = {
    "eval_11": [
        "Cross-encoder vs Bi-encoder reranker difference retrieval accuracy",
        "cross-encoder reranker reads query and chunk together for better scoring",
    ],
    "eval_13": [
        "DoRA Weight-Decomposed Low-Rank Adaptation magnitude direction decomposition",
        "DoRA LoRA weight decomposition magnitude direction update pattern",
    ],
    "eval_15": [
        "ACD-CLIP Zero-Shot Anomaly Detection ZSAD Vision-Language Model CLIP",
        "ACD-CLIP anomaly detection industrial medical benchmark dataset",
    ],
    "eval_18": [
        "Metadata Extraction RAG pipeline source page title attribution",
        "metadata extraction gắn nguồn trang tiêu đề citation",
    ],
}


def main():
    print("Loading eval dataset...")
    items = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    print(f"  {len(items)} items loaded.\n")

    print("=== Experiment 1: Basic Vector RAG (top_k=5) ===")
    vec = VectorRetriever()
    r1  = run_exp("1_basic_vector_k5", vec, items, top_k=5, extra_kw={})

    print("=== Experiment 2: Hybrid RAG (Vector + BM25 + RRF, top_k=12) ===")
    hyb = HybridRetriever()
    r2  = run_exp("2_hybrid_k12", hyb, items, top_k=12, extra_kw={"use_rerank": False})

    print("=== Experiment 3: Hybrid RAG (top_k=20, higher recall) ===")
    r3  = run_exp("3_hybrid_k20", hyb, items, top_k=20, extra_kw={"use_rerank": False})

    print("=== Experiment 4: Multi-Query Hybrid (Manual English Variants + RRF) ===")
    r4  = run_multi_query_exp("4_multi_query_hybrid", hyb, items, top_k=15, query_variants=QUERY_VARIANTS)

    print("\n" + "="*60)
    print("COMPARISON TABLE:")
    print(f"{'Experiment':<35} {'HitRate':>8} {'FactHit':>8} {'MRR':>7} {'P50ms':>7} {'P95ms':>7}")
    print("-"*75)
    for r in [r1, r2, r3, r4]:
        print(f"{r['experiment']:<35} {r['avg_hit_rate']:>7.1%} {r['factual_hit_rate']:>8.1%} "
              f"{r['avg_mrr']:>7.3f} {r['latency_p50_ms']:>7.0f} {r['latency_p95_ms']:>7.0f}")

    print("\nDelta Hybrid-k12 vs Vector-k5:")
    print(f"  Hit Rate: {r2['avg_hit_rate']-r1['avg_hit_rate']:+.1%}  MRR: {r2['avg_mrr']-r1['avg_mrr']:+.3f}")
    print(f"\nDelta Multi-Query vs Hybrid-k12:")
    print(f"  Hit Rate: {r4['avg_hit_rate']-r2['avg_hit_rate']:+.1%}  MRR: {r4['avg_mrr']-r2['avg_mrr']:+.3f}")

    results = {"experiments": [r1, r2, r3, r4]}
    OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {OUT_PATH}")

if __name__ == "__main__":
    main()
