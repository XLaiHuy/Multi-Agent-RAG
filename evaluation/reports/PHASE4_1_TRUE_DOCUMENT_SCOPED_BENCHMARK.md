# Phase 4.1 True Document-Scoped Retrieval Benchmark Report

**Date**: 2026-08-16  
**Status**: FINAL BENCHMARK RESULTS  
**Dataset**: CUAD DEV ($N=238$) & CUSTOM_CUAD_HOLDOUT_V2 ($N=293$)

---

## 1. Motivation & Task Formulation Distinction

Standard enterprise contract review systems (e.g., analyzing an uploaded NDA, Master Services Agreement, or Lease) operate in **Document-Scoped QA mode**: the user asks a question about a specific identified contract. 

In Phase 3 and Phase 4, queries were evaluated in **Global Multi-Contract mode** (searching across all 20-25 contracts simultaneously in a shared corpus of 1,034–1,348 chunks). When the user asks "What is the governing law?", global retrieval retrieves governing law clauses from 15 different contracts, crowding out the specific target contract's clause within the top-20 candidate budget.

Phase 4.1 establishes the mathematical separation between the two task formulations:

```
Multi-Contract Search Space:  N_chunks = 1,034 (DEV) / 1,348 (Holdout)
Document-Scoped Search Space: N_chunks = 52.0 median (DEV) / 45.0 median (Holdout)
```

---

## 2. DEV Split: Global Multi-Contract vs True Document-Scoped Comparison

| Metric Stage | Global Multi-Contract (1,034 Chunks) | True Document-Scoped (52 Chunks Median) | Absolute Delta |
| :--- | :--- | :--- | :--- |
| **CandidateHitRate@1** | 4.62% | **52.10%** | +47.48% |
| **CandidateHitRate@5** | 18.07% | **82.77%** | +64.70% |
| **CandidateHitRate@10** | 27.73% | **91.18%** | +63.45% |
| **CandidateHitRate@20** | 42.44% | **94.96%** | +52.52% |
| **CandidateHitRate@50** | 62.18% | **99.16%** | +36.98% |
| **Post-Rerank HitRate@1** | 5.46% | **51.26%** | +45.80% |
| **Post-Rerank HitRate@5** | 20.17% | **80.67%** | +60.50% |
| **Post-Rerank HitRate@10** | 31.09% | **90.34%** | +59.25% |
| **MRR** | 0.1173 | **0.6359** | +0.5186 |
| **nDCG@5** | 0.0477 | **0.3425** | +0.2948 |
| **TrueChunkRecall@5** | 4.23% | **25.26%** | +21.03% |
| **TrueChunkRecall@10** | 7.08% | **33.87%** | +26.79% |
| **Measured Total Latency (P50)** | 174.57 ms | **154.58 ms** | -19.99 ms |

---

## 3. Held-Out Evaluation on CUSTOM_CUAD_HOLDOUT_V2 ($N=293$)

| Metric Stage | Frozen Held-Out True Document-Scoped Result |
| :--- | :--- |
| **Evaluated Queries** | 293 valid answerable queries across 25 unseen contracts |
| **Search Space per Query** | Mean: 60.6 chunks, Median: 45.0 chunks, P95: 119.0 chunks |
| **CandidateHitRate@1** | 56.66% |
| **CandidateHitRate@5** | 83.96% |
| **CandidateHitRate@10** | 91.81% |
| **CandidateHitRate@20** | **98.29%** |
| **CandidateHitRate@50** | 99.32% |
| **Post-Rerank HitRate@1** | 50.85% |
| **Post-Rerank HitRate@5** | **82.94%** |
| **Post-Rerank HitRate@10** | **94.54%** |
| **MRR** | **0.6418** |
| **nDCG@5** | **0.3585** |
| **TrueChunkRecall@10** | **34.02%** |
| **Total Measured Latency P50** | **68.89 ms** |
| **Total Measured Latency P95** | **121.82 ms** |

---

## 4. Failure Breakdown on Held-Out Split ($N=293$)

| Failure Category | Count | Percentage | Root Cause & Resolution |
| :--- | :--- | :--- | :--- |
| **NONE_HIT_TOP5** | 243 | 82.94% | Query successfully retrieved gold chunk in top-5 final rank. |
| **TOP10_NOT_TOP5** | 34 | 11.60% | Gold chunk retrieved in top-10, but below top-5 threshold. |
| **LOST_BY_BUDGET** | 1 | 0.34% | Gold chunk ranked between position 21 and 50 before $k=20$ truncation. |
| **NOT_FOUND_SCOPED_FIRST_STAGE** | 2 | 0.68% | Vocabulary mismatch on domain acronyms in both Dense and BM25. |
| **RERANKER_DEMOTED** | 1 | 0.34% | CrossEncoder scored distractor section slightly higher than gold. |
| **OTHER** | 12 | 4.10% | Complex cross-table definition references. |
