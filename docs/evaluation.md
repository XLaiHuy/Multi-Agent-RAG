# Evaluation Methodology & Benchmark Provenance

## 1. Dataset Provenance & Split Integrity

The evaluation harness evaluates document-scoped legal question answering using the **Contract Understanding Atticus Dataset (CUAD v1)**.

### Splits
- **DEV Split (`cuad_dev_manifest.json`)**: 20 contracts, 244 answerable queries used for pipeline validation and cache benchmarking.
- **FROZEN HOLDOUT (`cuad_locked_test_v2_manifest.json`)**: 25 completely unseen contracts, 294 answerable queries, and 388 unanswerable queries (`CUSTOM_CUAD_HOLDOUT_V2`).

---

## 2. Strict Child Evidence Protocol (`STRICT_CHILD_EXACT_OR_SPAN_V2`)

In Phase 4.2, evidence mapping was rebuilt from scratch:
- A child chunk (~250 tokens) is marked relevant **only** if the normalized gold evidence is contained within that specific child or has $\ge 30$ chars overlap across boundaries.
- **Sibling child chunks NEVER inherit relevance** merely because they share a parent chunk.
- **Parent Context HitRate** is reported as a separate metric to evaluate the usefulness of the expanded context window (~1,200 tokens) for synthesis.

---

## 3. Frozen Held-Out Benchmark Results ($N=294$)

```json
{
  "benchmark": "CUSTOM_CUAD_HOLDOUT_V2",
  "total_contracts": 25,
  "total_answerable_queries": 294,
  "valid_evaluated_queries": 294,
  "strict_child_retrieval_metrics": {
    "CandidateHitRate@1": 40.48,
    "CandidateHitRate@5": 72.11,
    "CandidateHitRate@10": 85.37,
    "CandidateHitRate@20": 92.86,
    "HitRate@1": 39.12,
    "HitRate@5": 68.71,
    "HitRate@10": 81.97,
    "MRR": 0.5214,
    "nDCG@5": 0.4906,
    "TrueChunkRecall@10": 69.04
  },
  "parent_context_metrics_separate": {
    "ParentHitRate@5": 83.67,
    "ParentHitRate@10": 94.90
  }
}
```

---

## 4. Measured Online Latency Profile

Measured end-to-end including runtime `BAAI/bge-m3` online query embedding:
- **T_query_embedding**: P50 = **437.03 ms**, P95 = **620.21 ms**
- **T_post_embedding_retrieval**: P50 = **166.35 ms**, P95 = **267.04 ms**
- **T_total_online_retrieval_and_rerank**: P50 = **586.11 ms**, P95 = **819.96 ms**, P99 = **902.97 ms**

---

## 5. Cache Speedup Verification

Reproduced via `evaluation/scripts/benchmark_eval_cache.py`:
- **Cold Runtime**: 179.74 s
- **Warm Runtime**: 1.54 s
- **Measured Speedup**: **116.8x**
- **Fingerprint Verification**: `cold_result_hash == warm_result_hash` (Exact Match).
